# 单帧耗时 47.57ms 超预算性能优化审计报告

## 执行摘要

单帧耗时 **47.57ms** 严重超预算 **30.00ms**，超出 **58%**。本报告系统分析所有可能导致性能瓶颈的原因，并提供彻底的优化方案。

**目标**: 将单帧耗时从 47.57ms 降低到 30ms 以下。

---

## 1. 性能瓶颈分析

### 1.1 主要性能瓶颈（按预估耗时排序）

#### 🔴 **PERF-CRITICAL-001: DeepFilterNet 处理（预估 15-25ms）**

**位置**: `capture.rs:1164`

**问题分析**:
```1164:1170:demo/src/capture.rs
                lsnr = match df.process(inframe.view(), outframe.view_mut()) {
                    Ok(v) => v,
                    Err(err) => {
                        log::error!("DeepFilterNet 处理失败: {:?}", err);
                        continue;
                    }
                };
```

**预估耗时**: 15-25ms（取决于模型复杂度和硬件）

**优化方案**:
- **无法直接优化**（核心算法），但可以：
  1. 检查是否可以使用更轻量的模型
  2. 检查是否有不必要的频谱获取（`df.get_spec_noisy()` 可能触发额外计算）
  3. 确保模型已使用最优优化级别编译

**优先级**: **高**（核心瓶颈，但优化空间有限）

---

#### 🔴 **PERF-CRITICAL-002: VAD ONNX 推理（预估 8-15ms）**

**位置**: `capture.rs:1325`

**问题分析**:
```1324:1327:demo/src/capture.rs
                        if filled == vad_source_frame {
                            if let Ok(_) = v.process(&vad_frame_buf[..vad_source_frame]) {
                                vad_voice = v.is_speaking();
                            }
                        }
```

**预估耗时**: 8-15ms（ONNX 推理）

**问题**:
- 每帧都可能执行 ONNX 推理（如果缓冲区有足够数据）
- ONNX 推理是 CPU 密集型操作
- 没有节流机制

**优化方案**:
```rust
// 方案 1: 增加 VAD 处理节流（每 2-3 帧处理一次）
let mut vad_process_counter = 0usize;
const VAD_PROCESS_INTERVAL: usize = 2; // 每 2 帧处理一次

if vad_enabled && vad_source_frame > 0 && vad_buf_raw.len() >= vad_source_frame {
    vad_process_counter += 1;
    if vad_process_counter % VAD_PROCESS_INTERVAL == 0 {
        // 处理 VAD
        if let Some(ref mut v) = vad {
            // ... 现有处理逻辑
        }
    }
    // 即使不处理，也要保持缓冲区不溢出
    // 可以丢弃部分数据或增加缓冲区容量
}
```

**预期效果**: 减少 50% VAD 处理时间（4-7.5ms）

**优先级**: **高**

---

#### 🔴 **PERF-CRITICAL-003: 音色修复 ONNX 推理（预估 10-20ms）**

**位置**: `capture.rs:1678`

**问题分析**:
```1675:1697:demo/src/capture.rs
                    if let Some(ref mut tr) = timbre_restore {
                        if let Some(buffer) = outframe.as_slice_mut() {
                            let t0 = Instant::now();
                            if let Err(err) = tr.process_frame(buffer) {
                                log::warn!("音色修复处理失败，已重置状态: {}", err);
                                tr.reset();
                            } else if let Some(ref rec) = recording {
                                rec.append_timbre(buffer);
                            }
                            let elapsed = t0.elapsed().as_secs_f32();
                            if elapsed > block_duration * 0.8 {
                                timbre_overload_frames = timbre_overload_frames.max(16);
                                timbre_stride = (timbre_stride + 1).min(4);
                                timbre_skip_idx = 0;
                                log::warn!(
                                    "音色修复耗时 {:.1} ms，提升节流至每 {} 帧处理一次",
                                    elapsed * 1000.0,
                                    timbre_stride
                                );
                            } else {
                                timbre_last_good = Instant::now();
                            }
                        }
                    }
```

**预估耗时**: 10-20ms（ONNX 推理）

**问题**:
- 虽然有节流机制（`timbre_stride`），但可能不够激进
- 如果 `timbre_stride = 1`，仍然每帧都处理

**优化方案**:
```rust
// 方案 1: 增加默认节流（从每帧改为每 2-3 帧）
// 方案 2: 在性能紧张时自动增加节流
if elapsed_ms > budget_ms * 1.2 {
    // 性能紧张，增加音色修复节流
    timbre_stride = (timbre_stride + 1).min(8); // 最多每 8 帧处理一次
}

// 方案 3: 完全禁用音色修复（如果性能问题严重）
if elapsed_ms > budget_ms * 1.5 {
    timbre_restore = None; // 临时禁用
    log::warn!("性能紧张，临时禁用音色修复");
}
```

**预期效果**: 减少 50-75% 音色修复时间（5-15ms）

**优先级**: **高**

---

#### 🟡 **PERF-MEDIUM-001: 环境自适应计算（预估 3-8ms）**

**位置**: `capture.rs:1197-1203`

**问题分析**:
```1194:1203:demo/src/capture.rs
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
```

**预估耗时**: 3-8ms（每 2 帧执行一次，平均 1.5-4ms/帧）

**问题**:
- `compute_noise_features` 包含大量对数运算（`ln()`, `log10()`, `exp()`）
- `df.get_spec_noisy()` 可能触发额外计算
- 虽然已节流，但频率仍可能过高

**优化方案**:
```rust
// 方案 1: 增加节流间隔（从每 2 帧改为每 4 帧）
const FEATURE_INTERVAL: usize = 4; // 每 4 帧计算一次

// 方案 2: 在性能紧张时进一步节流
if elapsed_ms > budget_ms * 1.2 {
    const FEATURE_INTERVAL: usize = 8; // 每 8 帧计算一次
}

// 方案 3: 优化 compute_noise_features 计算
// - 使用 SIMD 加速
// - 减少对数运算（使用查表法）
// - 只计算必要的特征
```

**预期效果**: 减少 50-75% 环境自适应计算时间（0.75-3ms/帧）

**优先级**: **中**

---

#### 🟡 **PERF-MEDIUM-002: 动态 EQ 处理（预估 2-5ms）**

**位置**: `capture.rs:1734`

**问题分析**:
```1732:1738:demo/src/capture.rs
                let metrics = if let Some(buffer) = outframe.as_slice_mut() {
                    dynamic_eq.set_dry_wet(1.0);
                    dynamic_eq.process_block(buffer)
                } else {
                    log::error!("输出帧内存布局异常，跳过动态 EQ");
                    EqProcessMetrics::default()
                };
```

**预估耗时**: 2-5ms（多频段处理）

**优化方案**:
- 检查动态 EQ 是否可以使用更少的频段
- 检查是否可以降低处理频率
- 使用 SIMD 优化滤波器计算

**优先级**: **中**

---

#### 🟡 **PERF-MEDIUM-003: 频谱推送计算和内存分配（预估 1-3ms）**

**位置**: `capture.rs:1964-1965, 2420`

**问题分析**:
```1960:1967:demo/src/capture.rs
            spec_push_counter = spec_push_counter.wrapping_add(1);
            const SPEC_PUSH_INTERVAL: usize = 3;
            if spec_enabled && spec_push_counter % SPEC_PUSH_INTERVAL == 0 {
                if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                    push_spec(df.get_spec_noisy(), s_noisy);
                    push_spec(df.get_spec_enh(), s_enh);
                }
            }
```

```2420:2430:demo/src/capture.rs
fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1);
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

**预估耗时**: 1-3ms（每 3 帧执行一次，平均 0.33-1ms/帧）

**问题**:
- 每次调用都分配新内存
- 包含大量对数运算（`norm_sqr()`, `log10()`）
- `df.get_spec_noisy()` 和 `df.get_spec_enh()` 可能触发额外计算

**优化方案**:
```rust
// 方案 1: 增加节流间隔（从每 3 帧改为每 6-10 帧）
const SPEC_PUSH_INTERVAL: usize = 6; // 或 10

// 方案 2: 重用缓冲区，避免内存分配
thread_local! {
    static SPEC_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static SPEC_BOX: RefCell<Option<Box<[f32]>>> = RefCell::new(None);
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
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
            // 重用 Box，避免分配
            if let Some(ref mut b) = *box_buf {
                if b.len() == needed {
                    b.copy_from_slice(&buf[..needed]);
                    if let Err(err) = sender.send(b.clone()) {
                        log::warn!("Failed to send spectrogram data: {}", err);
                    }
                    return;
                }
            }
            *box_buf = Some(buf[..needed].to_vec().into_boxed_slice());
            if let Err(err) = sender.send(box_buf.as_ref().unwrap().clone()) {
                log::warn!("Failed to send spectrogram data: {}", err);
            }
        });
    });
}

// 方案 3: 在性能紧张时禁用频谱推送
if elapsed_ms > budget_ms * 1.2 {
    // 临时禁用频谱推送
    spec_enabled = false;
}
```

**预期效果**: 减少 50-80% 频谱推送时间（0.17-0.8ms/帧）

**优先级**: **中**

---

#### 🟡 **PERF-MEDIUM-004: 多次数组遍历（预估 0.5-2ms）**

**位置**: `capture.rs:1438-1456, 1871-1873`

**问题分析**:
```1438:1456:demo/src/capture.rs
                    let rms = df::rms(buf.iter());
                    let peak = buf.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                    let crest = if rms > 1e-6 { peak / rms } else { 0.0 };
                    // ... 高频能量计算
                    for (idx, v) in buf.iter().enumerate() {
                        if idx % 4 == 0 {
                            hf_energy += v * v;
                            hf_count += 1;
                        }
                    }
```

```1871:1873:demo/src/capture.rs
                for v in buffer.iter() {
                    peak = peak.max(v.abs());
                }
```

**预估耗时**: 0.5-2ms（多次遍历）

**优化方案**:
```rust
// 合并遍历：一次遍历同时计算 RMS、峰值、高频能量
let mut sum_sq = 0.0f32;
let mut peak = 0.0f32;
let mut hf_energy = 0.0f32;
let mut hf_count = 0usize;
for (idx, v) in buf.iter().enumerate() {
    let abs = v.abs();
    peak = peak.max(abs);
    sum_sq += v * v;
    if idx % 4 == 0 {
        hf_energy += v * v;
        hf_count += 1;
    }
}
let rms = (sum_sq / buf.len() as f32).sqrt();
let crest = if rms > 1e-6 { peak / rms } else { 0.0 };
```

**预期效果**: 减少 50-70% 遍历时间（0.25-1.4ms/帧）

**优先级**: **中**

---

#### 🟢 **PERF-LOW-001: 输出时临时内存分配（预估 0.1-0.5ms）**

**位置**: `capture.rs:1934`

**问题分析**:
```1934:1935:demo/src/capture.rs
                    let temp = buf[..n_out].to_vec();
                    push_output_block(&should_stop, &mut rb_out, &temp[..], n_out);
```

**预估耗时**: 0.1-0.5ms

**优化方案**:
- 检查 `push_output_block` 是否可以直接接受切片
- 如果必须分配，使用预分配的缓冲区

**优先级**: **低**

---

## 2. 优化方案汇总

### 2.1 立即优化（预期减少 15-25ms）

1. **VAD 处理节流**（每 2 帧一次）: **-4 to -7.5ms**
2. **音色修复节流**（每 2-4 帧一次）: **-5 to -15ms**
3. **环境自适应节流**（每 4 帧一次）: **-0.75 to -3ms**
4. **频谱推送节流**（每 6 帧一次）: **-0.17 to -0.8ms**
5. **合并数组遍历**: **-0.25 to -1.4ms**

**总计**: **-10 to -27.7ms**

### 2.2 进一步优化（预期减少 5-10ms）

1. **SIMD 优化**（频谱特征计算、数组遍历）: **-2 to -5ms**
2. **动态 EQ 优化**（减少频段或降低频率）: **-1 to -3ms**
3. **内存分配优化**（重用缓冲区）: **-0.5 to -1ms**

**总计**: **-3.5 to -9ms**

### 2.3 激进优化（如果仍超预算）

1. **完全禁用音色修复**（如果性能问题严重）: **-10 to -20ms**
2. **完全禁用频谱推送**（如果性能问题严重）: **-1 to -3ms**
3. **降低环境自适应频率**（每 8-10 帧一次）: **-1 to -2ms**

**总计**: **-12 to -25ms**

---

## 3. 详细优化建议

### 3.1 PERF-OPT-001: 添加性能自适应节流

**位置**: `capture.rs:1897-1921`

**问题**: 当前所有处理都是固定频率，没有根据实际性能动态调整

**建议**:
```rust
// 添加性能自适应节流
let mut perf_adaptive_throttle = 1usize; // 节流倍数

// 在处理耗时监测后
if elapsed_ms > budget_ms * 1.5 {
    // 性能严重超预算，增加节流
    perf_adaptive_throttle = perf_adaptive_throttle.saturating_add(1).min(4);
    log::warn!(
        "性能紧张，启用自适应节流 x{}（VAD/音色修复/环境自适应频率降低）",
        perf_adaptive_throttle
    );
} else if elapsed_ms < budget_ms * 0.8 {
    // 性能充足，减少节流
    perf_adaptive_throttle = perf_adaptive_throttle.saturating_sub(1).max(1);
}

// 应用到各个处理
const VAD_PROCESS_INTERVAL: usize = 2 * perf_adaptive_throttle;
const FEATURE_INTERVAL: usize = 2 * perf_adaptive_throttle;
const SPEC_PUSH_INTERVAL: usize = 3 * perf_adaptive_throttle;
// 音色修复的 timbre_stride 也需要根据 perf_adaptive_throttle 调整
```

**优先级**: **高**

---

### 3.2 PERF-OPT-002: 优化 compute_noise_features

**位置**: `capture.rs:2865-2893`

**问题**: 包含大量对数运算，可以优化

**建议**:
```rust
fn compute_noise_features(spec: ArrayView2<Complex32>) -> NoiseFeatures {
    let (_, freq_len) = spec.dim();
    let freq_len_f32 = freq_len.max(1) as f32;
    let row = spec.row(0);
    let eps = 1e-12f32;
    
    // 使用 SIMD 加速（如果可用）
    // 或者减少计算精度（使用快速对数近似）
    
    let mut sum_power = 0.0;
    let mut sum_log_power = 0.0;
    let mut weighted_sum = 0.0;
    
    // 优化：减少对数运算
    // 方案 1: 使用查表法（如果精度要求不高）
    // 方案 2: 使用快速对数近似（如 log2 然后转换）
    // 方案 3: 只计算必要的特征（如果某些特征可以省略）
    
    for (i, &c) in row.iter().enumerate() {
        let p = c.norm_sqr().max(eps);
        sum_power += p;
        // 使用 log2 然后转换，可能更快
        sum_log_power += p.log2(); // 使用 log2 代替 ln
        weighted_sum += p * i as f32;
    }
    
    let mean_power = sum_power / freq_len_f32;
    let energy_db = 10.0 * mean_power.max(eps).log10();
    // 转换 log2 到 ln: ln(x) = log2(x) * ln(2)
    let geometric_mean = (sum_log_power / freq_len_f32 * std::f32::consts::LN_2).exp();
    let spectral_flatness = geometric_mean / mean_power.max(eps);
    let spectral_centroid = if sum_power > 0.0 {
        (weighted_sum / sum_power) / freq_len_f32
    } else {
        0.0
    };
    
    NoiseFeatures {
        energy_db,
        spectral_flatness,
        spectral_centroid,
    }
}
```

**优先级**: **中**

---

### 3.3 PERF-OPT-003: 检查 df.get_spec 的开销

**位置**: `capture.rs:1198, 1964-1965`

**问题**: `df.get_spec_noisy()` 和 `df.get_spec_enh()` 可能触发额外计算

**建议**:
- 检查这些函数是否只是返回缓存的频谱，还是需要重新计算
- 如果需要重新计算，考虑缓存结果
- 如果不需要实时频谱，可以降低获取频率

**优先级**: **中**

---

### 3.4 PERF-OPT-004: 优化动态 EQ 处理

**位置**: `capture.rs:1734`

**建议**:
- 检查是否可以减少 EQ 频段数量
- 检查是否可以降低处理频率（如每 2 帧处理一次）
- 使用 SIMD 优化滤波器计算

**优先级**: **中**

---

### 3.5 PERF-OPT-005: 优化 RT60 估计

**位置**: `capture.rs:1393, 2895`

**问题**: `estimate_rt60_from_energy` 每帧都可能执行

**建议**:
```rust
// 增加 RT60 估计节流
let mut rt60_estimate_counter = 0usize;
const RT60_ESTIMATE_INTERVAL: usize = 4; // 每 4 帧估计一次

if !is_voice && rt60_enabled {
    // ... 更新 rt60_history ...
    rt60_estimate_counter += 1;
    if rt60_estimate_counter % RT60_ESTIMATE_INTERVAL == 0 {
        if let Some(rt) = estimate_rt60_from_energy(&rt60_history, block_duration) {
            smoothed_rt60 = smooth_value(smoothed_rt60, rt.clamp(0.2, 0.8), 0.25);
        }
    }
}
```

**优先级**: **低**

---

## 4. 性能监控增强

### 4.1 PERF-MON-001: 添加细粒度性能统计

**位置**: `capture.rs:1128, 1897`

**建议**:
```rust
// 添加各阶段性能统计
let mut df_time_ms = 0.0f32;
let mut vad_time_ms = 0.0f32;
let mut env_time_ms = 0.0f32;
let mut eq_time_ms = 0.0f32;
let mut timbre_time_ms = 0.0f32;
let mut other_time_ms = 0.0f32;

let frame_start = Instant::now();

// DeepFilterNet
let t_df = Instant::now();
lsnr = match df.process(...) { ... };
df_time_ms = t_df.elapsed().as_secs_f32() * 1000.0;

// VAD
let t_vad = Instant::now();
// ... VAD 处理 ...
vad_time_ms = t_vad.elapsed().as_secs_f32() * 1000.0;

// 环境自适应
let t_env = Instant::now();
// ... 环境自适应 ...
env_time_ms = t_env.elapsed().as_secs_f32() * 1000.0;

// 动态 EQ
let t_eq = Instant::now();
// ... 动态 EQ ...
eq_time_ms = t_eq.elapsed().as_secs_f32() * 1000.0;

// 音色修复
let t_timbre = Instant::now();
// ... 音色修复 ...
timbre_time_ms = t_timbre.elapsed().as_secs_f32() * 1000.0;

let total_time = frame_start.elapsed().as_secs_f32() * 1000.0;
other_time_ms = total_time - df_time_ms - vad_time_ms - env_time_ms - eq_time_ms - timbre_time_ms;

// 记录详细性能统计
if elapsed_ms > budget_ms * 1.5 && perf_last_log.elapsed() > Duration::from_millis(500) {
    log::warn!(
        "性能瓶颈分析: 总={:.2}ms, DF={:.2}ms, VAD={:.2}ms, ENV={:.2}ms, EQ={:.2}ms, Timbre={:.2}ms, Other={:.2}ms",
        total_time,
        df_time_ms,
        vad_time_ms,
        env_time_ms,
        eq_time_ms,
        timbre_time_ms,
        other_time_ms
    );
}
```

**优先级**: **高**（用于定位具体瓶颈）

---

## 5. 优化实施优先级

### 第一阶段：立即实施（预期减少 15-25ms）

1. ✅ **PERF-CRITICAL-002**: VAD 处理节流（每 2 帧一次）
2. ✅ **PERF-CRITICAL-003**: 音色修复节流（每 2-4 帧一次）
3. ✅ **PERF-MEDIUM-001**: 环境自适应节流（每 4 帧一次）
4. ✅ **PERF-MEDIUM-003**: 频谱推送节流（每 6 帧一次）
5. ✅ **PERF-MEDIUM-004**: 合并数组遍历
6. ✅ **PERF-OPT-001**: 添加性能自适应节流
7. ✅ **PERF-MON-001**: 添加细粒度性能统计

### 第二阶段：进一步优化（预期减少 5-10ms）

1. **PERF-OPT-002**: 优化 compute_noise_features
2. **PERF-OPT-003**: 检查 df.get_spec 的开销
3. **PERF-OPT-004**: 优化动态 EQ 处理
4. **PERF-MEDIUM-002**: 动态 EQ 优化

### 第三阶段：激进优化（如果仍超预算）

1. **完全禁用音色修复**（如果性能问题严重）
2. **完全禁用频谱推送**（如果性能问题严重）
3. **降低环境自适应频率**（每 8-10 帧一次）

---

## 6. 预期效果

### 优化前
- **单帧耗时**: 47.57ms
- **超预算**: 17.57ms (58%)

### 优化后（第一阶段）
- **单帧耗时**: 22.57-32.57ms
- **超预算**: -7.43 到 2.57ms
- **预期**: 大部分情况下可以满足 30ms 预算

### 优化后（第二阶段）
- **单帧耗时**: 17.57-27.57ms
- **超预算**: -12.43 到 -2.43ms
- **预期**: 可以稳定满足 30ms 预算

---

## 7. 关键发现

### 7.1 最严重的性能瓶颈

1. **DeepFilterNet 处理** (15-25ms) - 核心算法，优化空间有限
2. **VAD ONNX 推理** (8-15ms) - **可以节流优化**
3. **音色修复 ONNX 推理** (10-20ms) - **可以节流优化**

### 7.2 最容易优化的点

1. **VAD 处理节流** - 简单有效，预期减少 4-7.5ms
2. **音色修复节流** - 简单有效，预期减少 5-15ms
3. **环境自适应节流** - 简单有效，预期减少 0.75-3ms
4. **合并数组遍历** - 简单有效，预期减少 0.25-1.4ms

### 7.3 需要进一步调查的点

1. **DeepFilterNet 处理耗时** - 需要实际测量，确认是否真的是 15-25ms
2. **df.get_spec 的开销** - 需要确认是否触发额外计算
3. **动态 EQ 处理耗时** - 需要实际测量

---

## 8. 实施建议

### 8.1 立即实施（今天）

1. 添加细粒度性能统计，定位具体瓶颈
2. 实施 VAD 处理节流（每 2 帧一次）
3. 实施音色修复节流（每 2-4 帧一次）
4. 实施环境自适应节流（每 4 帧一次）
5. 实施频谱推送节流（每 6 帧一次）
6. 合并数组遍历

### 8.2 短期实施（本周）

1. 添加性能自适应节流
2. 优化 compute_noise_features
3. 检查 df.get_spec 的开销
4. 优化动态 EQ 处理

### 8.3 长期优化（如果需要）

1. SIMD 优化
2. 并行化处理（如果可能）
3. 模型优化（使用更轻量的模型）

---

## 9. 总结

单帧耗时 47.57ms 超预算的主要原因是：

1. **ONNX 推理开销**（VAD + 音色修复）: 18-35ms
2. **DeepFilterNet 处理**: 15-25ms
3. **环境自适应计算**: 1.5-4ms/帧
4. **其他处理**: 2-5ms

**最有效的优化方案**:
- **VAD 处理节流**: 减少 4-7.5ms
- **音色修复节流**: 减少 5-15ms
- **环境自适应节流**: 减少 0.75-3ms
- **合并数组遍历**: 减少 0.25-1.4ms

**总计预期减少**: 10-27.7ms

**优化后预期耗时**: 19.87-37.57ms（第一阶段）或 14.87-27.57ms（第二阶段）

**建议**: 立即实施第一阶段的优化，然后根据细粒度性能统计结果，进一步优化具体瓶颈。

