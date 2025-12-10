# 仅开启降噪/AGC/高通时的性能问题审计报告

## 执行摘要

用户反馈：**仅开启降噪（DeepFilterNet）、AGC、高通滤波器**，其他模块均未开启，但仍然出现单帧耗时超预算的警告（`单帧耗时 47.57 ms 超预算 30.00 ms，可能导致掉帧`）。

**关键发现**：即使只开启这三个核心模块，仍然存在多个隐藏的性能开销点，这些开销可能累积导致性能问题。

---

## 1. 核心模块性能分析

### 1.1 DeepFilterNet 降噪处理

**位置**: `capture.rs:1164-1170`

**预估耗时**: **15-25ms**（神经网络推理）

**问题**:
- DeepFilterNet 是神经网络模型，每帧都需要执行推理
- 这是**无法避免的核心开销**，但可以通过以下方式优化：
  1. 使用更轻量的模型
  2. 检查模型是否使用了最优优化级别编译
  3. 检查是否有不必要的频谱获取（见下文）

**优先级**: **高**（核心瓶颈，但优化空间有限）

---

### 1.2 AGC 处理

**位置**: `capture.rs:1765-1767`

**预估耗时**: **1-3ms**（WebRTC AGC 处理）

**问题**:
- WebRTC AGC 需要分块处理（10ms 帧长）
- 如果 `hop_size` 不是 10ms 的整数倍，会有额外开销

**代码分析**:
```rust
// capture.rs:1765-1767
if agc_enabled {
    agc.process(buffer);
}
```

**优化方案**:
- 检查 `hop_size` 是否与 AGC 的 `frame_size` 匹配
- 如果 `hop_size` 不是 10ms 的整数倍，考虑调整或优化分块逻辑

**优先级**: **中**

---

### 1.3 高通滤波器

**位置**: `capture.rs:1153-1157`

**预估耗时**: **< 0.1ms**（简单 IIR 滤波器）

**问题**:
- 高通滤波器是简单的 IIR 滤波器，逐样本处理
- 开销很小，**不是性能瓶颈**

**优先级**: **低**

---

## 2. 隐藏的性能开销（即使模块未开启）

### 🔴 **PERF-HIDDEN-001: 环境自适应计算（如果开启）**

**位置**: `capture.rs:1175-1208`

**问题**:
```rust
if env_auto_enabled && !bypass_enabled {
    // ...
    let feats = if feature_counter % FEATURE_INTERVAL == 0 {
        let new_feats = compute_noise_features(df.get_spec_noisy()); // <--- 每2帧执行一次
        cached_feats = new_feats;
        new_feats
    } else {
        cached_feats
    };
    // ...
}
```

**预估耗时**: **3-8ms**（每 2 帧执行一次，平均 **1.5-4ms/帧**）

**关键发现**:
- 即使只开启降噪、AGC、高通，如果 `env_auto_enabled` 开启，仍然会执行 `compute_noise_features`
- `compute_noise_features` 包含大量对数运算（`ln()`, `log10()`, `exp()`），遍历整个频谱
- 虽然已节流（每 2 帧一次），但频率仍可能过高

**优化方案**:
```rust
// 方案 1: 在性能紧张时进一步节流
if elapsed_ms > budget_ms * 1.2 {
    const FEATURE_INTERVAL: usize = 4; // 从每 2 帧改为每 4 帧
}

// 方案 2: 如果只开启降噪/AGC/高通，可以完全禁用环境自适应
if !env_auto_enabled {
    // 跳过环境自适应计算
}
```

**优先级**: **高**（如果 `env_auto_enabled` 开启）

---

### 🟡 **PERF-HIDDEN-002: 频谱推送（如果开启）**

**位置**: `capture.rs:1959-1967`

**问题**:
```rust
// 频谱推送节流：默认每 3 帧一次
spec_push_counter = spec_push_counter.wrapping_add(1);
const SPEC_PUSH_INTERVAL: usize = 3;
if spec_enabled && spec_push_counter % SPEC_PUSH_INTERVAL == 0 {
    if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
        push_spec(df.get_spec_noisy(), s_noisy);
        push_spec(df.get_spec_enh(), s_enh);
    }
}
```

**预估耗时**: **1-3ms**（每 3 帧执行一次，平均 **0.33-1ms/帧**）

**关键发现**:
- 如果 `spec_enabled` 开启（频谱可视化），每 3 帧会调用 `push_spec`
- `push_spec` 会遍历频谱数据并计算对数（`norm_sqr()`, `log10()`）
- 虽然已节流，但仍有开销

**优化方案**:
```rust
// 方案 1: 增加节流间隔（从每 3 帧改为每 6-10 帧）
const SPEC_PUSH_INTERVAL: usize = 6; // 或 10

// 方案 2: 在性能紧张时禁用频谱推送
if elapsed_ms > budget_ms * 1.2 {
    // 临时禁用频谱推送
    spec_enabled = false;
}
```

**优先级**: **中**（如果 `spec_enabled` 开启）

---

### 🟡 **PERF-HIDDEN-003: 录音处理（如果开启）**

**位置**: `capture.rs:1130-1133, 1924-1928`

**问题**:
```rust
// 录音原始信号（设备采样率或重采样后），在任何处理前
if let Some(ref rec) = recording {
    if let Some(buffer) = inframe.as_slice() {
        rec.append_noisy(buffer); // <--- 可能涉及内存分配
    }
}
// ...
// 录音最终输出（限幅后）
if let Some(ref rec) = recording {
    if let Some(buffer) = outframe.as_slice_mut() {
        rec.append_processed(buffer); // <--- 可能涉及内存分配
    }
}
```

**预估耗时**: **0.1-1ms**（取决于内存分配开销）

**关键发现**:
- 如果录音功能开启，每帧都会调用 `rec.append_noisy` 和 `rec.append_processed`
- 这些函数可能涉及内存分配和复制

**优化方案**:
- 检查 `recording` 的实现，确认是否有不必要的内存分配
- 如果录音不是必需的，可以临时禁用

**优先级**: **低**（如果录音开启）

---

### 🟡 **PERF-HIDDEN-004: 输出重采样（如果设备采样率不一致）**

**位置**: `capture.rs:1930-1947`

**问题**:
```rust
if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
    // ...
    if let Err(err) = r.process_into_buffer(&[slice], buf, None) {
        log::error!("输出重采样失败: {:?}", err);
    } else {
        push_output_block(&should_stop, &mut rb_out, &buf[0][..n_out], n_out);
    }
}
```

**预估耗时**: **1-5ms**（取决于重采样算法和采样率比）

**关键发现**:
- 如果设备采样率与模型采样率不一致，会有输出重采样开销
- 重采样是 CPU 密集型操作

**优化方案**:
- 尽量使用与模型采样率一致的设备采样率
- 如果必须重采样，考虑使用更高效的重采样算法

**优先级**: **中**（如果设备采样率不一致）

---

### 🟡 **PERF-HIDDEN-005: 输入缓冲区等待**

**位置**: `capture.rs:1099-1118`

**问题**:
```rust
while filled < n_in {
    let pulled = rb_in.pop_slice(&mut buffer[filled..n_in]);
    if pulled == 0 {
        if should_stop.load(Ordering::Relaxed) {
            log::debug!("停止时输入数据不足，退出处理循环");
            break 'processing;
        }
        if start_fill.elapsed() > input_timeout {
            log::warn!(
                "等待输入数据超时（需要 {}，已获取 {}），丢弃该帧",
                n_in,
                filled
            );
            continue 'processing;
        }
        sleep(input_retry_delay); // <--- 等待数据
        continue;
    }
    filled += pulled;
}
```

**预估耗时**: **0-10ms**（取决于数据到达时间）

**关键发现**:
- 如果输入数据不足，会 sleep 并等待
- 这个等待时间**不计入处理时间**（因为 `frame_start` 在填充完成后才开始计时）
- 但如果数据到达延迟，会导致整体延迟

**优化方案**:
- 检查输入缓冲区大小是否足够
- 检查音频设备驱动是否有延迟问题

**优先级**: **低**（通常不是主要问题）

---

### 🟡 **PERF-HIDDEN-006: 输出缓冲区等待**

**位置**: `capture.rs:2442-2470`

**问题**:
```rust
fn push_output_block(
    should_stop: &Arc<AtomicBool>,
    rb_out: &mut RbProd,
    data: &[f32],
    expected_frames: usize,
) {
    // ...
    while n < expected_frames {
        if should_stop.load(Ordering::Relaxed) {
            log::debug!("停止播放输出（检测到停止信号）");
            break;
        }
        if start_time.elapsed() > timeout {
            log::warn!("播放输出超时，跳过 {} 个样本", expected_frames - n);
            break;
        }
        let pushed = rb_out.push_slice(&data[n..expected_frames]);
        if pushed == 0 {
            sleep(retry_delay); // <--- 等待缓冲区空间
        } else {
            n += pushed;
        }
    }
    rb_out.sync();
}
```

**预估耗时**: **0-20ms**（取决于缓冲区空间，最多等待 20ms）

**关键发现**:
- 如果输出缓冲区满了，会 sleep 并等待
- 这个等待时间**不计入处理时间**（因为 `frame_start` 在输出推送前已结束计时）
- 但如果缓冲区经常满，说明处理速度跟不上，会导致整体延迟

**优化方案**:
- 检查输出缓冲区大小是否足够
- 检查音频设备驱动是否有延迟问题
- 如果缓冲区经常满，说明处理速度需要优化

**优先级**: **低**（通常不是主要问题，但可能是症状）

---

### 🟡 **PERF-HIDDEN-007: sanitize_samples 检查**

**位置**: `capture.rs:1144, 1172, 等`

**问题**:
```rust
sanitize_samples("输入信号", buffer);
// ...
sanitize_samples("降噪输出", buffer);
```

**预估耗时**: **< 0.1ms**（简单检查）

**关键发现**:
- `sanitize_samples` 会遍历整个缓冲区检查 NaN/Inf
- 虽然开销很小，但每帧都会执行多次

**优化方案**:
- 如果确定数据正常，可以在 Release 模式下禁用这些检查
- 或者只在 Debug 模式下启用

**优先级**: **低**（开销很小）

---

### 🟡 **PERF-HIDDEN-008: 峰值检测和限幅**

**位置**: `capture.rs:1869-1892`

**问题**:
```rust
// 一次遍历检测异常峰值并记录峰值
let mut peak = 0.0f32;
for v in buffer.iter() {
    peak = peak.max(v.abs());
}
// ...
// 最终限幅一次，避免多级限幅导致音色压缩
apply_final_limiter(buffer);
```

**预估耗时**: **0.1-0.5ms**（遍历和限幅）

**关键发现**:
- 峰值检测需要遍历整个缓冲区
- `apply_final_limiter` 也需要遍历整个缓冲区
- 可以合并为一次遍历

**优化方案**:
```rust
// 合并峰值检测和限幅为一次遍历
let mut peak = 0.0f32;
for v in buffer.iter_mut() {
    let abs = v.abs();
    peak = peak.max(abs);
    // 限幅逻辑
    *v = v.clamp(-1.0, 1.0);
}
```

**优先级**: **低**（开销很小，但可以优化）

---

## 3. 性能瓶颈汇总

### 3.1 核心模块（必须执行）

| 模块 | 预估耗时 | 优化空间 |
|------|---------|---------|
| DeepFilterNet | 15-25ms | 低（核心算法） |
| AGC | 1-3ms | 中 |
| 高通滤波器 | < 0.1ms | 低（已足够快） |
| **小计** | **16-28ms** | - |

### 3.2 隐藏开销（如果开启）

| 模块 | 预估耗时 | 优化空间 |
|------|---------|---------|
| 环境自适应计算 | 1.5-4ms/帧 | 高（可以节流或禁用） |
| 频谱推送 | 0.33-1ms/帧 | 中（可以节流或禁用） |
| 录音处理 | 0.1-1ms/帧 | 低（可以禁用） |
| 输出重采样 | 1-5ms/帧 | 中（可以避免） |
| 输入缓冲区等待 | 0-10ms | 低（取决于数据到达） |
| 输出缓冲区等待 | 0-20ms | 低（取决于缓冲区空间） |
| sanitize_samples | < 0.1ms | 低（可以禁用） |
| 峰值检测和限幅 | 0.1-0.5ms | 低（可以合并） |
| **小计（如果全部开启）** | **3-32ms** | - |

### 3.3 总计

**最坏情况**（所有隐藏开销都开启）:
- 核心模块: 16-28ms
- 隐藏开销: 3-32ms
- **总计: 19-60ms**

**典型情况**（只开启降噪/AGC/高通，环境自适应开启，频谱推送开启）:
- 核心模块: 16-28ms
- 环境自适应: 1.5-4ms
- 频谱推送: 0.33-1ms
- **总计: 17.83-33ms**

**最佳情况**（只开启降噪/AGC/高通，其他都关闭）:
- 核心模块: 16-28ms
- **总计: 16-28ms**

---

## 4. 优化建议

### 4.1 立即优化（预期减少 2-5ms）

1. **禁用环境自适应计算**（如果不需要）:
   ```rust
   // 如果只开启降噪/AGC/高通，可以禁用环境自适应
   env_auto_enabled = false;
   ```
   **预期效果**: 减少 1.5-4ms/帧

2. **禁用频谱推送**（如果不需要可视化）:
   ```rust
   // 如果不需要频谱可视化，可以禁用
   spec_enabled = false;
   ```
   **预期效果**: 减少 0.33-1ms/帧

3. **检查输出重采样**:
   - 如果设备采样率与模型采样率不一致，尽量使用一致的采样率
   - 如果必须重采样，考虑使用更高效的重采样算法
   **预期效果**: 减少 1-5ms/帧（如果存在重采样）

### 4.2 进一步优化（预期减少 1-2ms）

1. **合并峰值检测和限幅**:
   ```rust
   // 一次遍历同时完成峰值检测和限幅
   let mut peak = 0.0f32;
   for v in buffer.iter_mut() {
       let abs = v.abs();
       peak = peak.max(abs);
       *v = v.clamp(-1.0, 1.0);
   }
   ```
   **预期效果**: 减少 0.1-0.3ms/帧

2. **优化 AGC 分块处理**:
   - 检查 `hop_size` 是否与 AGC 的 `frame_size` 匹配
   - 如果不匹配，优化分块逻辑
   **预期效果**: 减少 0.5-1ms/帧（如果存在不匹配）

3. **禁用 sanitize_samples**（在 Release 模式下）:
   ```rust
   #[cfg(debug_assertions)]
   sanitize_samples("输入信号", buffer);
   ```
   **预期效果**: 减少 < 0.1ms/帧

### 4.3 深度优化（如果仍超预算）

1. **使用更轻量的 DeepFilterNet 模型**:
   - 检查是否有更轻量的模型可用
   - 或者使用量化模型

2. **优化 DeepFilterNet 推理**:
   - 检查模型是否使用了最优优化级别编译
   - 考虑使用 GPU 加速（如果可用）

3. **增加处理节流**:
   - 如果性能仍然紧张，可以考虑降低处理频率（如每 2 帧处理一次）

---

## 5. 诊断建议

### 5.1 添加细粒度性能统计

**位置**: `capture.rs:1897`

**建议**:
```rust
// 添加各阶段性能统计
let mut df_time_ms = 0.0f32;
let mut agc_time_ms = 0.0f32;
let mut env_time_ms = 0.0f32;
let mut spec_time_ms = 0.0f32;
let mut other_time_ms = 0.0f32;

let frame_start = Instant::now();

// DeepFilterNet
let t_df = Instant::now();
lsnr = match df.process(inframe.view(), outframe.view_mut()) {
    Ok(v) => v,
    Err(err) => {
        log::error!("DeepFilterNet 处理失败: {:?}", err);
        continue;
    }
};
df_time_ms = t_df.elapsed().as_secs_f32() * 1000.0;

// 环境自适应
let t_env = Instant::now();
if env_auto_enabled && !bypass_enabled {
    // ... 环境自适应计算 ...
}
env_time_ms = t_env.elapsed().as_secs_f32() * 1000.0;

// AGC
let t_agc = Instant::now();
if agc_enabled {
    agc.process(buffer);
}
agc_time_ms = t_agc.elapsed().as_secs_f32() * 1000.0;

// 频谱推送
let t_spec = Instant::now();
if spec_enabled && spec_push_counter % SPEC_PUSH_INTERVAL == 0 {
    // ... 频谱推送 ...
}
spec_time_ms = t_spec.elapsed().as_secs_f32() * 1000.0;

let total_time = frame_start.elapsed().as_secs_f32() * 1000.0;
other_time_ms = total_time - df_time_ms - agc_time_ms - env_time_ms - spec_time_ms;

// 记录详细性能统计
if elapsed_ms > budget_ms * 1.5 && perf_last_log.elapsed() > Duration::from_millis(500) {
    log::warn!(
        "性能瓶颈分析: 总={:.2}ms, DF={:.2}ms, AGC={:.2}ms, ENV={:.2}ms, SPEC={:.2}ms, Other={:.2}ms",
        total_time,
        df_time_ms,
        agc_time_ms,
        env_time_ms,
        spec_time_ms,
        other_time_ms
    );
}
```

**优先级**: **高**（用于定位具体瓶颈）

---

## 6. 关键发现总结

### 6.1 主要性能瓶颈

1. **DeepFilterNet 处理** (15-25ms) - **核心瓶颈，无法避免**
2. **环境自适应计算** (1.5-4ms/帧) - **如果开启，可以禁用或节流**
3. **输出重采样** (1-5ms/帧) - **如果设备采样率不一致，可以避免**
4. **AGC 处理** (1-3ms) - **开销较小，但可以优化**

### 6.2 最容易优化的点

1. **禁用环境自适应计算** - 简单有效，预期减少 1.5-4ms/帧
2. **禁用频谱推送** - 简单有效，预期减少 0.33-1ms/帧
3. **避免输出重采样** - 使用与模型一致的设备采样率，预期减少 1-5ms/帧

### 6.3 需要进一步调查的点

1. **DeepFilterNet 处理耗时** - 需要实际测量，确认是否真的是 15-25ms
2. **环境自适应是否开启** - 需要确认 `env_auto_enabled` 的状态
3. **频谱推送是否开启** - 需要确认 `spec_enabled` 的状态
4. **设备采样率是否一致** - 需要确认是否存在输出重采样

---

## 7. 实施建议

### 7.1 立即实施（今天）

1. **添加细粒度性能统计**，定位具体瓶颈
2. **检查环境自适应是否开启**，如果不需要，禁用它
3. **检查频谱推送是否开启**，如果不需要可视化，禁用它
4. **检查设备采样率是否一致**，如果存在重采样，尽量使用一致的采样率

### 7.2 短期实施（本周）

1. **合并峰值检测和限幅**为一次遍历
2. **优化 AGC 分块处理**（如果存在不匹配）
3. **禁用 sanitize_samples**（在 Release 模式下）

### 7.3 长期优化（如果需要）

1. **使用更轻量的 DeepFilterNet 模型**
2. **优化 DeepFilterNet 推理**（使用 GPU 或量化）
3. **增加处理节流**（如果性能仍然紧张）

---

## 8. 总结

即使只开启降噪、AGC、高通，仍然存在多个隐藏的性能开销点：

1. **环境自适应计算**（如果开启）: 1.5-4ms/帧
2. **频谱推送**（如果开启）: 0.33-1ms/帧
3. **输出重采样**（如果设备采样率不一致）: 1-5ms/帧
4. **录音处理**（如果开启）: 0.1-1ms/帧

**最有效的优化方案**:
- **禁用环境自适应计算**: 减少 1.5-4ms/帧
- **禁用频谱推送**: 减少 0.33-1ms/帧
- **避免输出重采样**: 减少 1-5ms/帧

**总计预期减少**: 2.83-10ms/帧

**优化后预期耗时**: 14-38ms（最坏情况）或 14-28ms（典型情况）

**建议**: 立即添加细粒度性能统计，定位具体瓶颈，然后针对性优化。

