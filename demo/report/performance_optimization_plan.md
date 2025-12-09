# 音频处理性能优化方案

## 执行摘要

本报告提供系统性的性能优化方案，涵盖算法优化、内存优化、SIMD 优化、并行化等多个方面。

---

## 1. 算法层面优化

### 1.1 OPT-ALGO-001: 合并多次数组遍历

**位置**: `capture.rs:1798-1812, 1823-1826`

**问题分析**:
```1798:1812:demo/src/capture.rs
            if let Some(buffer) = outframe.as_slice_mut() {
                // 异常信号检测，避免上游节点输出过大数值
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
```

```1823:1826:demo/src/capture.rs
                let mut peak = 0.0f32;
                for v in buffer.iter() {
                    peak = peak.max(v.abs());
                }
```

**问题**:
- 多次遍历同一数组：异常检测遍历一次，峰值监测又遍历一次
- 可以合并为一次遍历

**优化方案**:
```rust
if let Some(buffer) = outframe.as_slice_mut() {
    let mut raw_peak = 0.0f32;
    let mut peak = 0.0f32;
    // 一次遍历同时检测异常和峰值
    for v in buffer.iter() {
        let abs = v.abs();
        raw_peak = raw_peak.max(abs);
        peak = peak.max(abs);
    }
    if raw_peak > 2.0 {
        log::warn!("检测到异常峰值 {:.2}，将限幅保护", raw_peak);
        for v in buffer.iter_mut() {
            *v = v.clamp(-1.2, 1.2);
        }
        peak = 1.2; // 限幅后峰值已知
    }
    if peak > 0.99 && perf_last_log.elapsed() > Duration::from_secs(2) {
        log::warn!("输出峰值 {:.3}，接近裁剪", peak);
    }
}
```

**预期效果**: 减少一次数组遍历，节省 0.5-1ms

---

### 1.2 OPT-ALGO-002: 优化 RMS 计算

**位置**: `capture.rs:1159, 1419`

**问题分析**:
```1159:1159:demo/src/capture.rs
                    let rms = df::rms(buf.iter());
```

**问题**:
- RMS 计算需要遍历整个数组并计算平方和
- 如果只需要判断阈值，可以使用更快的近似方法

**优化方案**:
```rust
// 方案 1: 使用峰值近似 RMS（更快）
fn fast_rms_approx(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    peak * 0.707 // 峰值到 RMS 的近似转换（正弦波）
}

// 方案 2: 采样计算（降低精度但更快）
fn sampled_rms(samples: &[f32], step: usize) -> f32 {
    let mut sum_sq = 0.0f32;
    let mut count = 0usize;
    for (i, &v) in samples.iter().enumerate() {
        if i % step == 0 {
            sum_sq += v * v;
            count += 1;
        }
    }
    (sum_sq / count.max(1) as f32).sqrt()
}
```

**预期效果**: 减少计算时间 30-50%

---

### 1.3 OPT-ALGO-003: 动态 EQ 频段处理优化

**位置**: `audio/eq/dynamic_eq.rs:189-198`

**问题分析**:
```189:198:demo/src/audio/eq/dynamic_eq.rs
        for (idx, band) in self.bands.iter_mut().enumerate() {
            let rms = band.analyze(&self.analysis_buf[..len]);
            band.update(rms, len);
            if idx < MAX_EQ_BANDS {
                metrics.gain_db[idx] = band.gain_db();
            }
        }
        for band in self.bands.iter_mut() {
            band.apply(samples);
        }
```

**问题**:
- 每个频段都要遍历整个数组两次（analyze 和 apply）
- 如果有 5 个频段，就是 10 次遍历

**优化方案**:
```rust
// 方案 1: 合并 analyze 和 apply（如果可能）
// 方案 2: 使用 SIMD 并行处理多个频段
// 方案 3: 减少频段数量（从 5 个减少到 3 个）

// 方案 4: 条件处理（只在需要时处理）
for (idx, band) in self.bands.iter_mut().enumerate() {
    // 如果增益变化很小，跳过处理
    let old_gain = band.gain_db();
    let rms = band.analyze(&self.analysis_buf[..len]);
    band.update(rms, len);
    let new_gain = band.gain_db();
    
    // 只有增益变化超过阈值才应用
    if (new_gain - old_gain).abs() > 0.1 {
        band.apply(samples);
    }
    if idx < MAX_EQ_BANDS {
        metrics.gain_db[idx] = new_gain;
    }
}
```

**预期效果**: 减少 30-50% 的处理时间

---

### 1.4 OPT-ALGO-004: 频谱特征计算优化

**位置**: `capture.rs:2750-2777` (compute_noise_features)

**问题分析**:
```2750:2777:demo/src/capture.rs
fn compute_noise_features(spec: ArrayView2<Complex32>) -> NoiseFeatures {
    // ... 遍历整个频谱，计算能量、平坦度、重心
    for (i, &c) in row.iter().enumerate() {
        let p = c.norm_sqr().max(eps);
        sum_power += p;
        sum_log_power += p.ln();  // 对数计算耗时
        weighted_sum += p * i as f32;
    }
    // ... 更多计算
}
```

**问题**:
- 大量对数运算（`ln()`, `log10()`, `exp()`）计算量大
- 可以降低计算频率或使用查表法

**优化方案**:
```rust
// 方案 1: 使用查表法替代对数计算
const LOG_TABLE_SIZE: usize = 1024;
static LOG_TABLE: [f32; LOG_TABLE_SIZE] = {
    let mut table = [0.0f32; LOG_TABLE_SIZE];
    let mut i = 0;
    while i < LOG_TABLE_SIZE {
        table[i] = (i as f32 / LOG_TABLE_SIZE as f32).ln();
        i += 1;
    }
    table
};

fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return -100.0;
    }
    let idx = ((x * LOG_TABLE_SIZE as f32) as usize).min(LOG_TABLE_SIZE - 1);
    LOG_TABLE[idx]
}

// 方案 2: 降低计算频率（每 2-3 帧一次）
// 方案 3: 使用 SIMD 并行计算
```

**预期效果**: 减少计算时间 40-60%

---

## 2. SIMD 向量化优化

### 2.1 OPT-SIMD-001: 峰值检测 SIMD 优化

**位置**: `capture.rs:1798-1812`

**优化方案**:
```rust
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn find_peak_avx2(samples: &[f32]) -> f32 {
    let mut max = _mm256_setzero_ps();
    let chunks = samples.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        let vec = _mm256_loadu_ps(chunk.as_ptr());
        let abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0), vec); // abs
        max = _mm256_max_ps(max, abs);
    }
    
    // 水平最大值
    let max_array = std::mem::transmute::<__m256, [f32; 8]>(max);
    let mut result = max_array[0];
    for &v in &max_array[1..] {
        result = result.max(v);
    }
    
    // 处理剩余元素
    for &v in remainder {
        result = result.max(v.abs());
    }
    
    result
}
```

**预期效果**: 加速 4-8 倍（取决于数组大小）

---

### 2.2 OPT-SIMD-002: RMS 计算 SIMD 优化

**位置**: 多处使用 `df::rms()`

**优化方案**:
```rust
#[target_feature(enable = "avx2")]
unsafe fn rms_avx2(samples: &[f32]) -> f32 {
    let mut sum_sq = _mm256_setzero_ps();
    let chunks = samples.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        let vec = _mm256_loadu_ps(chunk.as_ptr());
        let sq = _mm256_mul_ps(vec, vec);
        sum_sq = _mm256_add_ps(sum_sq, sq);
    }
    
    // 水平求和
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum_sq);
    let mut sum = sum_array.iter().sum::<f32>();
    
    // 处理剩余元素
    for &v in remainder {
        sum += v * v;
    }
    
    (sum / samples.len() as f32).sqrt()
}
```

**预期效果**: 加速 4-8 倍

---

### 2.3 OPT-SIMD-003: 动态 EQ 滤波器 SIMD 优化

**位置**: `audio/eq/dynamic_band.rs:165-169`

**优化方案**:
```rust
// Biquad 滤波器 SIMD 优化
// 需要重新设计滤波器状态存储，使用 SIMD 寄存器
// 可以并行处理多个样本
```

**预期效果**: 加速 2-4 倍

---

## 3. 内存优化

### 3.1 OPT-MEM-001: 频谱推送内存分配优化（已部分优化）

**位置**: `capture.rs:2348`

**当前状态**: 已使用 `thread_local` 重用缓冲区，但发送时仍分配

**进一步优化**:
```rust
// 使用预分配的 Box<[f32]> 并重用
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
        
        // 重用或调整 Box
        match box_buf.as_mut() {
            Some(b) if b.len() == needed => {
                b.copy_from_slice(&buf[..needed]);
            }
            _ => {
                *box_buf = Some(buf[..needed].to_vec().into_boxed_slice());
            }
        }
        
        if let Err(err) = sender.send(box_buf.as_ref().unwrap().clone()) {
            log::warn!("Failed to send spectrogram data: {}", err);
        }
    });
});
```

**预期效果**: 完全消除发送时的内存分配

---

### 3.2 OPT-MEM-002: 动态 EQ 分析缓冲区优化

**位置**: `audio/eq/dynamic_eq.rs:185-188`

**优化方案**:
```rust
// 使用预分配的缓冲区，避免每次 resize
if self.analysis_buf.capacity() < len {
    self.analysis_buf.reserve(len - self.analysis_buf.capacity());
}
if self.analysis_buf.len() < len {
    self.analysis_buf.resize(len, 0.0);
}
```

**预期效果**: 减少内存分配次数

---

## 4. 计算频率优化

### 4.1 OPT-FREQ-001: 环境自适应计算频率优化

**位置**: `capture.rs:1152`

**优化方案**:
```rust
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
    // ... 使用 feats
}
```

**预期效果**: 减少 50% 的计算时间

---

### 4.2 OPT-FREQ-002: 峰值监测频率优化

**位置**: `capture.rs:1827-1829`

**优化方案**:
```rust
// 降低峰值监测频率
static mut PEAK_CHECK_COUNTER: usize = 0;
unsafe {
    PEAK_CHECK_COUNTER += 1;
    if PEAK_CHECK_COUNTER % 3 == 0 {
        if peak > 0.99 && perf_last_log.elapsed() > Duration::from_secs(2) {
            log::warn!("输出峰值 {:.3}，接近裁剪", peak);
        }
    }
}
```

**预期效果**: 减少峰值检测开销

---

## 5. 并行化优化

### 5.1 OPT-PAR-001: 动态 EQ 频段并行处理

**位置**: `audio/eq/dynamic_eq.rs:189-198`

**优化方案**:
```rust
use rayon::prelude::*;

// 并行分析所有频段
let rms_values: Vec<f32> = self.bands
    .par_iter_mut()
    .map(|band| band.analyze(&self.analysis_buf[..len]))
    .collect();

// 并行更新所有频段
self.bands
    .par_iter_mut()
    .zip(rms_values.par_iter())
    .for_each(|(band, &rms)| {
        band.update(rms, len);
    });

// 串行应用（因为需要修改同一个 samples 数组）
for band in self.bands.iter_mut() {
    band.apply(samples);
}
```

**预期效果**: 加速 2-4 倍（取决于 CPU 核心数）

---

### 5.2 OPT-PAR-002: 频谱特征计算并行化

**位置**: `capture.rs:2750-2777`

**优化方案**:
```rust
use rayon::prelude::*;

// 并行计算各个特征
let (sum_power, sum_log_power, weighted_sum) = row
    .par_iter()
    .enumerate()
    .map(|(i, &c)| {
        let p = c.norm_sqr().max(eps);
        (p, p.ln(), p * i as f32)
    })
    .reduce(
        || (0.0, 0.0, 0.0),
        |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3)
    );
```

**预期效果**: 加速 2-3 倍

---

## 6. 缓存优化

### 6.1 OPT-CACHE-001: 频谱数据缓存

**位置**: `capture.rs:1152, 1899-1900`

**优化方案**:
```rust
// 缓存频谱数据，避免重复获取
struct CachedSpec {
    noisy: Array2<Complex32>,
    enh: Array2<Complex32>,
    frame_counter: usize,
}

let mut spec_cache = CachedSpec {
    noisy: Array2::zeros((1, 512)),
    enh: Array2::zeros((1, 512)),
    frame_counter: 0,
};

// 每帧更新缓存
spec_cache.frame_counter += 1;
if spec_cache.frame_counter % 2 == 0 {
    // 每 2 帧更新一次
    spec_cache.noisy = df.get_spec_noisy().to_owned();
    spec_cache.enh = df.get_spec_enh().to_owned();
}
```

**预期效果**: 减少频谱获取开销

---

## 7. 数据结构优化

### 7.1 OPT-DATA-001: 使用更高效的数据结构

**位置**: 多处使用 `VecDeque`

**优化方案**:
```rust
// 对于固定大小的缓冲区，使用固定大小的数组 + 循环索引
struct CircularBuffer<T, const N: usize> {
    data: [T; N],
    head: usize,
    len: usize,
}

// 避免 VecDeque 的动态分配
```

**预期效果**: 减少内存分配和碎片

---

## 8. 编译器优化

### 8.1 OPT-COMP-001: 启用 LTO 和优化选项

**Cargo.toml**:
```toml
[profile.release]
lto = "thin"  # 或 "fat"
codegen-units = 1
opt-level = 3
```

**预期效果**: 提升 10-20% 性能

---

## 9. 优先级和预期效果总结

### 高优先级（立即实施）

1. **OPT-ALGO-001**: 合并多次数组遍历 - 预期减少 0.5-1ms
2. **OPT-FREQ-001**: 环境自适应计算频率优化 - 预期减少 3-8ms
3. **OPT-MEM-001**: 频谱推送内存分配优化 - 预期减少内存分配开销

**总预期**: 减少 4-10ms 耗时

---

### 中优先级（近期优化）

1. **OPT-ALGO-003**: 动态 EQ 频段处理优化 - 预期减少 1-3ms
2. **OPT-ALGO-002**: RMS 计算优化 - 预期减少 0.5-1ms
3. **OPT-SIMD-001**: 峰值检测 SIMD - 预期加速 4-8 倍
4. **OPT-SIMD-002**: RMS 计算 SIMD - 预期加速 4-8 倍

**总预期**: 减少 2-5ms 耗时

---

### 低优先级（长期优化）

1. **OPT-PAR-001**: 动态 EQ 并行处理 - 预期加速 2-4 倍
2. **OPT-PAR-002**: 频谱特征计算并行化 - 预期加速 2-3 倍
3. **OPT-SIMD-003**: 动态 EQ 滤波器 SIMD - 预期加速 2-4 倍
4. **OPT-ALGO-004**: 频谱特征计算优化 - 预期减少 40-60% 计算时间

**总预期**: 减少 3-8ms 耗时

---

## 10. 实施建议

### 第一阶段（快速优化）
1. 合并多次数组遍历
2. 环境自适应计算频率优化
3. 频谱推送内存分配优化

**预期效果**: 总耗时从 25-50ms 降低到 20-40ms

---

### 第二阶段（算法优化）
1. 动态 EQ 频段处理优化
2. RMS 计算优化
3. 峰值检测 SIMD 优化

**预期效果**: 总耗时降低到 15-30ms

---

### 第三阶段（深度优化）
1. 并行化优化
2. 更多 SIMD 优化
3. 数据结构优化

**预期效果**: 总耗时降低到 10-25ms，大部分情况下可以满足 30ms 预算

---

## 11. 注意事项

1. **SIMD 优化**: 需要检查 CPU 支持（AVX2, SSE4.1 等）
2. **并行化**: 需要注意线程安全和数据竞争
3. **缓存优化**: 需要注意缓存一致性
4. **测试**: 每次优化后需要充分测试，确保功能正确性

---

## 12. 总结

通过系统性的性能优化，预期可以将总处理时间从当前的 25-50ms 降低到 10-25ms，大部分情况下可以满足 30ms 预算。

优化重点：
1. **算法层面**: 减少遍历次数，优化计算
2. **SIMD 优化**: 利用 CPU 向量指令
3. **并行化**: 利用多核 CPU
4. **内存优化**: 减少分配和碎片
5. **计算频率**: 降低不必要的计算

建议按阶段实施，逐步优化。

