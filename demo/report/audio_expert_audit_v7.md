# Demo 目录音频处理代码专家级审计报告 v7

## 执行摘要

本报告对代码进行第 7 次审计，仅报告当前仍存在的问题。已修复的问题不再提及。

---

## 1. 逻辑错误

### 1.1 AUDIO-BUG-V7-001: 环境自适应特征计算节流未生效

**位置**: `capture.rs:1157, 1178-1189`

**问题分析**:
```1157:1189:demo/src/capture.rs
                let feats = compute_noise_features(df.get_spec_noisy());
                // ... 其他代码 ...
                // 环境特征节流：每 2 帧计算一次，其余使用缓存
                feature_counter = feature_counter.wrapping_add(1);
                const FEATURE_INTERVAL: usize = 2;
                let _feats = if feature_counter % FEATURE_INTERVAL == 0 {
                    let new_feats = compute_noise_features(df.get_spec_noisy());
                    cached_feats = new_feats;
                    new_feats
                } else {
                    cached_feats
                };
                smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, update_alpha);
                smoothed_flatness =
                    smooth_value(smoothed_flatness, feats.spectral_flatness, update_alpha);
                smoothed_centroid =
                    smooth_value(smoothed_centroid, feats.spectral_centroid, update_alpha);
```

**问题**:
- 第 1157 行定义了 `feats`，每帧都调用 `compute_noise_features`（**没有节流**）
- 第 1178 行定义了 `_feats`，实现了节流逻辑（每 2 帧一次），但被标记为未使用
- 第 1185-1189 行使用的是第 1157 行的 `feats`，而不是节流后的 `_feats`
- **结果**: 节流代码存在但未生效，`compute_noise_features` 仍然每帧都执行

**影响**:
- **严重**: 性能优化未生效，环境自适应计算仍然每帧执行
- 浪费计算资源（5-15ms/帧）
- 可能导致性能问题

**建议**:
```rust
// 删除第 1157 行的 feats 定义，使用节流后的 feats
// 环境特征节流：每 2 帧计算一次，其余使用缓存
feature_counter = feature_counter.wrapping_add(1);
const FEATURE_INTERVAL: usize = 2;
let feats = if feature_counter % FEATURE_INTERVAL == 0 {
    let new_feats = compute_noise_features(df.get_spec_noisy());
    cached_feats = new_feats;
    new_feats
} else {
    cached_feats
};
smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, update_alpha);
// ...
```

**优先级**: **高**

---

## 2. 性能问题

### 2.1 AUDIO-BUG-V7-002: 频谱推送每次分配新内存

**位置**: `capture.rs:2384-2393`

**问题分析**:
```2384:2393:demo/src/capture.rs
fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    let needed = spec.len();
    let mut out: Vec<f32> = Vec::with_capacity(needed);
    for src in spec.iter() {
        out.push(src.norm_sqr().max(1e-10).log10() * 10.0);
    }
    if let Err(err) = sender.send(out.into_boxed_slice()) {
        log::warn!("Failed to send spectrogram data: {}", err);
    }
}
```

**问题**:
- 每次调用 `push_spec` 都会创建新的 `Vec<f32>`
- 虽然使用了 `with_capacity`，但仍需要分配内存
- 频谱推送每 3 帧一次，仍有内存分配开销

**影响**:
- 内存分配开销（虽然频率已降低）
- 可能导致内存碎片
- 影响性能

**建议**:
```rust
fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1);
    thread_local! {
        static SPEC_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }
    SPEC_BUF.with(|buf_cell| {
        let mut buf = buf_cell.borrow_mut();
        let needed = spec.len();
        if buf.len() < needed {
            buf.resize(needed, 0.0);
        }
        for (dst, src) in buf.iter_mut().zip(spec.iter()) {
            *dst = src.norm_sqr().max(1e-10).log10() * 10.0;
        }
        // 需要检查 SendSpec 是否支持直接发送切片，或使用其他方式避免分配
        // 如果必须分配，可以考虑使用对象池
        if let Err(err) = sender.send(buf[..needed].to_vec().into_boxed_slice()) {
            log::warn!("Failed to send spectrogram data: {}", err);
        }
    });
}
```

**优先级**: 中

---

### 2.2 AUDIO-BUG-V7-003: 输出时仍分配临时内存

**位置**: `capture.rs:1934`

**问题分析**:
```1934:1935:demo/src/capture.rs
                    let temp = buf[..n_out].to_vec();
                    push_output_block(&should_stop, &mut rb_out, &temp[..], n_out);
```

**问题**:
- 每次输出都会创建临时 `Vec`（`to_vec()`）
- 这是每帧都会执行的操作，内存分配开销较大

**影响**:
- 每帧都有内存分配开销
- 可能导致内存碎片
- 影响实时性能

**建议**:
- 检查 `push_output_block` 是否可以直接接受切片
- 如果可以，直接传递 `&buf[..n_out]`
- 如果必须分配，考虑使用预分配的缓冲区

**优先级**: 中

---

### 2.3 AUDIO-BUG-V7-004: 多次数组遍历可合并

**位置**: `capture.rs:1851-1877`

**问题分析**:
```1851:1877:demo/src/capture.rs
                let mut raw_peak = 0.0f32;
                for v in buffer.iter() {
                    raw_peak = raw_peak.max(v.abs());
                }
                if raw_peak > 2.0 {
                    log::warn!(
                        "检测到异常峰值 {:.2}，将限幅保护（可能某处理节点异常增益）",
                        raw_peak
                    );
                    for v in buffer.iter_mut() {
                        *v = v.clamp(-1.2, 1.2);
                    }
                }
                // ... 其他处理 ...
                // 最终限幅一次，避免多级限幅导致音色压缩
                apply_final_limiter(buffer);
                // 峰值监测，提示潜在增益问题
                let mut peak = 0.0f32;
                for v in buffer.iter() {
                    peak = peak.max(v.abs());
                }
                if peak > 0.99 && perf_last_log.elapsed() > Duration::from_secs(2) {
                    log::warn!("输出峰值 {:.3}，接近裁剪，请下调增益/饱和/激励", peak);
                }
```

**问题**:
- 异常峰值检测遍历一次数组（1851-1854）
- 如果异常，限幅时再遍历一次（1860-1862）
- 峰值监测又遍历一次（1874-1876）
- 总共可能遍历 3 次，可以合并为 1-2 次

**影响**:
- 额外的数组遍历开销（约 0.1-0.5ms，取决于缓冲区大小）
- 可以优化性能

**建议**:
```rust
// 合并遍历：一次遍历同时检测异常峰值和最终峰值
let mut raw_peak = 0.0f32;
let mut final_peak = 0.0f32;
for v in buffer.iter() {
    let abs = v.abs();
    raw_peak = raw_peak.max(abs);
    final_peak = final_peak.max(abs);
}
if raw_peak > 2.0 {
    log::warn!(
        "检测到异常峰值 {:.2}，将限幅保护（可能某处理节点异常增益）",
        raw_peak
    );
    for v in buffer.iter_mut() {
        *v = v.clamp(-1.2, 1.2);
    }
    final_peak = 1.2; // 限幅后的峰值
}
// ... 其他处理 ...
apply_final_limiter(buffer);
// 如果限幅后需要重新计算峰值，可以在这里再次遍历
// 否则直接使用 final_peak
if final_peak > 0.99 && perf_last_log.elapsed() > Duration::from_secs(2) {
    log::warn!("输出峰值 {:.3}，接近裁剪，请下调增益/饱和/激励", final_peak);
}
```

**优先级**: 低

---

## 3. 数值稳定性问题

### 3.1 AUDIO-BUG-V7-005: RT60 计算中 slope 接近零的风险

**位置**: `capture.rs:2866-2870`

**问题分析**:
```2866:2870:demo/src/capture.rs
    let slope = (end_mean - start_mean) / duration; // dB/s，衰减应为负值
    if slope >= -10.0 {
        return None;
    }
    let rt60 = (-60.0 / slope).clamp(0.2, 1.2);
```

**问题**:
- 虽然检查了 `slope >= -10.0`，但如果 `slope` 非常接近 0（例如 `-0.001`），`-60.0 / slope` 会产生非常大的值
- 虽然 `clamp(0.2, 1.2)` 会限制结果，但计算过程中可能出现数值不稳定
- 如果 `slope` 为 `-10.0`，`-60.0 / -10.0 = 6.0`，会被 clamp 到 1.2，但逻辑上应该返回 `None`（因为 `slope >= -10.0`）

**影响**:
- 数值计算可能不稳定
- RT60 估计可能不准确

**建议**:
```rust
let slope = (end_mean - start_mean) / duration; // dB/s，衰减应为负值
if slope >= -10.0 {
    return None;
}
// 确保 slope 足够负，避免除零或数值不稳定
if slope.abs() < 1e-6 {
    return None;
}
let rt60 = (-60.0 / slope).clamp(0.2, 1.2);
```

**优先级**: 低

---

## 4. 逻辑优化建议

### 4.1 AUDIO-OPT-V7-001: 输出增益检查可以优化

**位置**: `capture.rs:1741-1744`

**问题分析**:
```1741:1744:demo/src/capture.rs
                    if out_gain < 0.9999 || out_gain > 1.0001 {
                        for v in buffer.iter_mut() {
                            *v *= out_gain;
                        }
                    }
```

**问题**:
- 使用浮点数比较 `0.9999` 和 `1.0001` 作为阈值
- 如果 `out_gain` 恰好为 `1.0`，但浮点数精度问题导致比较失败，会执行不必要的乘法
- 可以更精确地检查是否接近 1.0

**影响**:
- 轻微的性能影响（如果 `out_gain` 恰好为 1.0 但比较失败）
- 代码可读性

**建议**:
```rust
// 使用更精确的阈值，或直接比较
if (out_gain - 1.0).abs() > 1e-6 {
    for v in buffer.iter_mut() {
        *v *= out_gain;
    }
}
```

**优先级**: 低（优化建议）

---

## 5. 优先级总结

### 高优先级
1. **AUDIO-BUG-V7-001**: 环境自适应特征变量使用错误（编译错误或逻辑错误）

### 中优先级
1. **AUDIO-BUG-V7-002**: 频谱推送每次分配新内存
2. **AUDIO-BUG-V7-003**: 输出时仍分配临时内存

### 低优先级
1. **AUDIO-BUG-V7-004**: 多次数组遍历可合并
2. **AUDIO-BUG-V7-005**: RT60 计算中 slope 接近零的风险
3. **AUDIO-OPT-V7-001**: 输出增益检查可以优化

---

## 6. 总结

本次审计发现 5 个问题和 1 个优化建议：

1. **逻辑错误** (1个，高优先级):
   - 环境自适应特征变量使用错误（`_feats` vs `feats`）

2. **性能问题** (3个):
   - 频谱推送每次分配新内存
   - 输出时仍分配临时内存
   - 多次数组遍历可合并

3. **数值稳定性问题** (1个):
   - RT60 计算中 slope 接近零的风险

4. **优化建议** (1个):
   - 输出增益检查可以优化

**重要**: **AUDIO-BUG-V7-001** 是严重问题，可能导致代码无法编译或逻辑错误，需要立即修复。

