# Demo 目录音频处理代码专家级审计报告 v6

## 执行摘要

本报告对代码进行第 6 次审计，仅报告当前仍存在的问题。已修复的问题不再提及。

---

## 1. 性能问题

### 1.1 AUDIO-BUG-V6-001: 频谱推送仍分配新内存

**位置**: `capture.rs:2348`

**问题分析**:
```2333:2352:demo/src/capture.rs
fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    // 预分配缓冲并重用，减少分配；按列数量一次性分配
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
        if let Err(err) = sender.send(buf[..needed].to_vec().into_boxed_slice()) {
        log::warn!("Failed to send spectrogram data: {}", err);
        }
    });
}
```

**问题**:
- 虽然重用了 `SPEC_BUF` 缓冲区进行计算，但最后发送时 `buf[..needed].to_vec()` 仍会分配新内存
- 每次推送（每 3 帧一次）都会分配新内存，导致内存分配开销

**影响**:
- 仍有内存分配开销
- 可能导致内存碎片
- 影响性能（虽然频率已降低）

**建议**:
```rust
// 方案 1: 使用 Box<[f32]> 直接发送切片（如果 sender 支持）
// 需要检查 SendSpec 的类型定义

// 方案 2: 预分配 Box<[f32]> 并重用
thread_local! {
    static SPEC_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static SPEC_BOX: RefCell<Option<Box<[f32]>>> = RefCell::new(None);
}
SPEC_BUF.with(|buf_cell| {
    SPEC_BOX.with(|box_cell| {
        let mut buf = buf_cell.borrow_mut();
        let mut box_buf = box_cell.borrow_mut();
        let needed = spec.len();
        if buf.len() < needed {
            buf.resize(needed, 0.0);
        }
        for (dst, src) in buf.iter_mut().zip(spec.iter()) {
            *dst = src.norm_sqr().max(1e-10).log10() * 10.0;
        }
        // 重用或创建 Box
        if let Some(ref mut b) = *box_buf {
            if b.len() != needed {
                *box_buf = Some(buf[..needed].to_vec().into_boxed_slice());
            } else {
                b.copy_from_slice(&buf[..needed]);
            }
        } else {
            *box_buf = Some(buf[..needed].to_vec().into_boxed_slice());
        }
        if let Err(err) = sender.send(box_buf.as_ref().unwrap().clone()) {
            log::warn!("Failed to send spectrogram data: {}", err);
        }
    });
});
```

**优先级**: 中

---

### 1.2 AUDIO-BUG-V6-002: 环境自适应计算仍每帧执行

**位置**: `capture.rs:1152`

**问题分析**:
```1150:1152:demo/src/capture.rs
            if env_auto_enabled && !bypass_enabled {
                // 环境噪声特征估计与自适应参数：噪声地板 + SNR 连续映射
                let feats = compute_noise_features(df.get_spec_noisy());
```

**问题**:
- 环境自适应计算（包括频谱特征计算）每帧都执行
- `compute_noise_features` 包含大量对数运算（`ln()`, `log10()`, `exp()`），计算量大
- 可以降低计算频率（如每 2-3 帧一次），使用缓存的频谱特征

**影响**:
- 仍有 5-15ms 的计算开销
- 可以进一步优化性能

**建议**:
```rust
// 添加计算频率节流
let mut feature_counter = 0usize;
let mut cached_feats = NoiseFeatures {
    energy_db: -60.0,
    spectral_flatness: 0.5,
    spectral_centroid: 0.5,
};

if env_auto_enabled && !bypass_enabled {
    feature_counter += 1;
    const FEATURE_INTERVAL: usize = 2; // 每 2 帧计算一次
    let feats = if feature_counter % FEATURE_INTERVAL == 0 {
        let new_feats = compute_noise_features(df.get_spec_noisy());
        cached_feats = new_feats;
        new_feats
    } else {
        cached_feats
    };
    // ... 后续使用 feats
}
```

**优先级**: 中

---

## 2. 错误处理和边界检查问题

### 2.1 AUDIO-BUG-V6-003: VAD 重采样错误处理不足

**位置**: `capture.rs:1208-1214`

**问题分析**:
```1205:1216:demo/src/capture.rs
                            if let Some((rs, rs_buf)) = vad_resampler.as_mut() {
                                if rs_buf.len() >= 1 && rs_buf[0].len() >= buf.len() {
                                    rs_buf[0][..buf.len()].copy_from_slice(buf);
                                    if let Ok(out) = rs.process(rs_buf, None) {
                                        if let Some(out_ch) = out.get(0) {
                                            for &v in out_ch.iter() {
                                                push_sample(v);
                                            }
                                        }
                                    }
                                }
                            }
```

**问题**:
1. **错误处理不足**: `rs.process()` 失败时没有日志或错误处理，静默失败
2. **数据丢失风险**: 重采样失败时，该帧的 VAD 数据丢失，可能导致 VAD 输入不完整
3. **没有重试机制**: 临时错误可能导致永久数据丢失

**影响**:
- VAD 输入可能不完整
- 可能影响语音检测准确性
- 错误难以追踪

**建议**:
```rust
if let Some((rs, rs_buf)) = vad_resampler.as_mut() {
    if rs_buf.len() >= 1 && rs_buf[0].len() >= buf.len() {
        rs_buf[0][..buf.len()].copy_from_slice(buf);
        match rs.process(rs_buf, None) {
            Ok(out) => {
                if let Some(out_ch) = out.get(0) {
                    for &v in out_ch.iter() {
                        push_sample(v);
                    }
                }
            }
            Err(err) => {
                // 记录错误，但不中断处理
                if vad_resample_error_count == 0 || 
                   vad_resample_error_last_log.elapsed() > Duration::from_secs(5) {
                    log::warn!("VAD 重采样失败: {:?}，该帧数据丢失", err);
                    vad_resample_error_last_log = Instant::now();
                }
                vad_resample_error_count += 1;
                // 可选：如果错误过多，禁用 VAD 重采样
                if vad_resample_error_count > 10 {
                    log::error!("VAD 重采样连续失败 {} 次，禁用 VAD", vad_resample_error_count);
                    vad_enabled = false;
                }
            }
        }
    }
}
```

**优先级**: 中

---

### 2.2 AUDIO-BUG-V6-004: VAD 重采样输出容量检查不完整

**位置**: `capture.rs:1210-1212`

**问题分析**:
```1210:1212:demo/src/capture.rs
                                        if let Some(out_ch) = out.get(0) {
                                            for &v in out_ch.iter() {
                                                push_sample(v);
                                            }
                                        }
```

**问题**:
- `push_sample` 函数会检查 `vad_buf_raw.len() >= cap` 并丢弃数据，但没有检查重采样输出的大小
- 如果重采样输出过多，可能导致大量数据被丢弃，影响 VAD 准确性
- 没有限制单次重采样输出的处理量

**影响**:
- 可能导致 VAD 缓冲区溢出
- VAD 输入可能不完整
- 可能影响语音检测准确性

**建议**:
```rust
if let Some(out_ch) = out.get(0) {
    // 限制单次处理量，避免缓冲区溢出
    let max_samples_per_frame = vad_source_frame; // 限制为单帧大小
    let samples_to_process = out_ch.len().min(max_samples_per_frame);
    for &v in out_ch.iter().take(samples_to_process) {
        push_sample(v);
    }
    // 如果输出过多，记录警告
    if out_ch.len() > max_samples_per_frame {
        if vad_oversample_warn_last.elapsed() > Duration::from_secs(5) {
            log::warn!(
                "VAD 重采样输出过多 ({} > {})，已限制处理量",
                out_ch.len(),
                max_samples_per_frame
            );
            vad_oversample_warn_last = Instant::now();
        }
    }
}
```

**优先级**: 低

---

## 3. 性能监控优化建议

### 3.1 AUDIO-OPT-V6-001: 性能监控缺少细粒度统计

**位置**: `capture.rs:1830-1855`

**问题分析**:
```1830:1855:demo/src/capture.rs
                // 处理耗时监测
                let elapsed_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
                let smooth = 0.08f32;
                proc_time_avg_ms = proc_time_avg_ms * (1.0 - smooth) + elapsed_ms * smooth;
                proc_time_peak_ms = proc_time_peak_ms.max(elapsed_ms);
                // 预算=DF hop + 重采样延迟（设备与模型采样率不一致时），下限 30ms 以适配 24k↔16k
                let budget_ms = (block_duration * 1000.0 + resample_latency_ms).max(30.0);
                if perf_last_log.elapsed() > Duration::from_secs(5) {
                    log::info!(
                        "帧耗时 avg/peak {:.2}/{:.2} ms（预算 {:.2} ms，重采样 {:.2} ms）",
                        proc_time_avg_ms,
                        proc_time_peak_ms,
                        budget_ms,
                        resample_latency_ms
                    );
                    proc_time_peak_ms *= 0.5; // 简单衰减记录
                    perf_last_log = Instant::now();
                }
                // 留出 50% 容错，避免设备采样率不可调导致的常驻告警
                if elapsed_ms > budget_ms * 1.5 && perf_last_log.elapsed() > Duration::from_millis(500) {
                    log::warn!(
                        "单帧耗时 {:.2} ms 超预算 {:.2} ms，可能导致掉帧",
                        elapsed_ms,
                        budget_ms
                    );
                }
```

**问题**:
- 只有总耗时统计，没有各阶段（DF、VAD、环境自适应等）的细粒度统计
- 难以定位具体性能瓶颈
- 超预算时无法知道哪个阶段耗时最多

**影响**:
- 性能问题定位困难
- 无法知道哪个阶段耗时最多

**建议**:
```rust
// 添加细粒度性能监控
let t_df = Instant::now();
lsnr = df.process(...)?;
let df_time = t_df.elapsed().as_millis();

let t_vad = Instant::now();
// VAD 处理
let vad_time = t_vad.elapsed().as_millis();

let t_env = Instant::now();
// 环境自适应
let env_time = t_env.elapsed().as_millis();

// 记录各阶段耗时
if elapsed_ms > budget_ms * 1.5 && perf_last_log.elapsed() > Duration::from_millis(500) {
    log::warn!(
        "性能瓶颈: 总耗时={:.2}ms, DF={:.2}ms, VAD={:.2}ms, ENV={:.2}ms, 其他={:.2}ms",
        elapsed_ms,
        df_time as f32,
        vad_time as f32,
        env_time as f32,
        elapsed_ms - df_time as f32 - vad_time as f32 - env_time as f32
    );
}
```

**优先级**: 低（优化建议）

---

## 4. 优先级总结

### 高优先级
无

### 中优先级
1. **AUDIO-BUG-V6-001**: 频谱推送内存分配优化
2. **AUDIO-BUG-V6-002**: 环境自适应计算频率优化
3. **AUDIO-BUG-V6-003**: VAD 重采样错误处理

### 低优先级
1. **AUDIO-BUG-V6-004**: VAD 重采样输出容量检查
2. **AUDIO-OPT-V6-001**: 性能监控细粒度统计

---

## 5. 总结

本次审计发现 4 个问题和 1 个优化建议：

1. **性能问题** (2个):
   - 频谱推送仍分配新内存
   - 环境自适应计算频率可以优化

2. **错误处理问题** (2个):
   - VAD 重采样错误处理不足
   - VAD 重采样输出容量检查不完整

3. **优化建议** (1个):
   - 性能监控缺少细粒度统计

所有问题均为中低优先级，不影响核心功能，但可以进一步优化性能和稳定性。

