# 输出重采样说明

## 1. 输出一般需要重采样么？

### 1.1 重采样触发条件

**输出重采样在以下情况下会被触发**：

```rust
// capture.rs:978-998
let (mut output_resampler, n_out) = if output_sr != df.sr {
    // 设备采样率 != 内部处理采样率，需要重采样
    match FftFixedIn::<f32>::new(df.sr, output_sr, df.hop_size, 1, 1) {
        Ok(r) => {
            // 创建重采样器：内部 48kHz -> 设备采样率
            (Some((r, buf)), n_out)
        }
        Err(err) => {
            log::error!("输出重采样初始化失败: {:?}", err);
            return;
        }
    }
} else {
    // 设备采样率 == 内部处理采样率，无需重采样
    (None, df.hop_size)
};
```

**关键点**：
- **内部处理采样率固定为 48kHz** (`PROCESS_SR = 48_000`)
- **如果设备采样率 ≠ 48kHz，就需要重采样**
- **如果设备采样率 = 48kHz，无需重采样**

### 1.2 设备采样率如何确定？

**设备采样率的确定流程**：

```rust
// capture.rs:3026-3027
let mut source = AudioSource::new(sr as u32, frame_size, input_device)?;
let mut sink = AudioSink::new(sr as u32, frame_size, output_device)?;
// sr 是模型采样率，固定为 48kHz
```

```rust
// capture.rs:589-596
let selection = get_stream_config(&device, sample_rate, StreamDirection::Output, frame_size)
    .with_context(|| {
        format!(
            "No suitable audio output config found for device {}",
            device.name().unwrap_or_else(|_| "unknown".into())
        )
    })?;
```

```rust
// capture.rs:404-418
let select_with_format = |format: cpal::SampleFormat| -> Option<StreamSelection> {
    for c in configs.iter() {
        if c.sample_format() != format {
            continue;
        }
        // 检查设备是否支持请求的采样率（48kHz）
        if sr >= c.min_sample_rate() && sr <= c.max_sample_rate() {
            let mut cfg: StreamConfig = (*c).with_sample_rate(sr).into();
            cfg.buffer_size = BufferSize::Fixed(frame_size as u32);
            return Some(StreamSelection {
                config: cfg,
                format,
            });
        }
    }
    None
};
```

**如果设备不支持 48kHz**：

```rust
// capture.rs:430-448
for format in [
    cpal::SampleFormat::F32,
    cpal::SampleFormat::I16,
    cpal::SampleFormat::U16,
] {
    if let Some(range) = configs.iter().find(|c| c.sample_format() == format) {
        // 使用设备支持的最大采样率
        let mut cfg: StreamConfig = (*range).with_max_sample_rate().into();
        // 调整 buffer_size 以适应实际采样率
        cfg.buffer_size =
            BufferSize::Fixed(frame_size as u32 * cfg.sample_rate.0 / sample_rate);
        log::warn!(
            "Using best matching config {:?} with sample format {:?}",
            cfg,
            format
        );
        return Ok(StreamSelection {
            config: cfg,
            format,
        });
    }
}
```

**总结**：
1. 首先尝试使用 48kHz 配置设备
2. 如果设备支持 48kHz，直接使用，**无需重采样**
3. 如果设备不支持 48kHz，使用设备支持的最大采样率（如 24kHz），**需要重采样**

---

## 2. 如果把 48kHz 的送给了只支持 24kHz 的会怎么样？

### 2.1 代码会自动处理

**如果设备只支持 24kHz，代码会自动处理**：

1. **设备配置阶段**：
   ```rust
   // capture.rs:436
   let mut cfg: StreamConfig = (*range).with_max_sample_rate().into();
   // 设备会被配置为 24kHz（设备支持的最大采样率）
   ```

2. **重采样器创建阶段**：
   ```rust
   // capture.rs:978-998
   let (mut output_resampler, n_out) = if output_sr != df.sr {
       // output_sr = 24kHz, df.sr = 48kHz
       // 创建重采样器：48kHz -> 24kHz
       match FftFixedIn::<f32>::new(df.sr, output_sr, df.hop_size, 1, 1) {
           Ok(r) => {
               log::info!(
                   "输出重采样: 内部 {} Hz -> 设备 {} Hz，块长 {}",
                   df.sr,      // 48kHz
                   output_sr,  // 24kHz
                   n_out
               );
               (Some((r, buf)), n_out)
           }
           // ...
       }
   }
   ```

3. **运行时重采样**：
   ```rust
   // capture.rs:1930-1947
   if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
       if let Some(slice) = outframe.as_slice() {
           // 将 48kHz 的内部处理结果重采样到 24kHz
           if let Err(err) = r.process_into_buffer(&[slice], buf, None) {
               log::error!("输出重采样失败: {:?}", err);
           } else {
               // 推送重采样后的 24kHz 数据到输出缓冲区
               push_output_block(&should_stop, &mut rb_out, &buf[0][..n_out], n_out);
           }
       }
   }
   ```

### 2.2 会发生什么？

#### ✅ **正常情况**（代码会自动处理）

1. **设备会被配置为 24kHz**
   - 代码会检测到设备不支持 48kHz
   - 自动使用设备支持的最大采样率（24kHz）

2. **自动创建重采样器**
   - 48kHz 内部处理 → 24kHz 设备输出
   - 使用 `rubato` 库的 `FftFixedIn` 重采样器

3. **运行时自动重采样**
   - 每帧处理时，将 48kHz 数据重采样到 24kHz
   - 重采样后的数据推送到输出缓冲区

4. **日志会提示**：
   ```
   输出重采样: 内部 48000 Hz -> 设备 24000 Hz，块长 XXX
   ```

#### ⚠️ **潜在问题**

1. **性能开销**：
   - 重采样是 CPU 密集型操作
   - **预估耗时：1-5ms/帧**（取决于重采样算法和采样率比）
   - 如果性能已经紧张，重采样会进一步增加延迟

2. **音质损失**：
   - 从 48kHz 降采样到 24kHz 会丢失高频信息（> 12kHz）
   - 虽然人耳对 > 12kHz 的感知有限，但理论上会有音质损失

3. **延迟增加**：
   ```rust
   // capture.rs:1003-1005
   if output_sr != df.sr {
       resample_latency_ms += (n_out as f32 / output_sr as f32) * 1000.0;
   }
   ```
   - 重采样会增加处理延迟
   - 延迟 = 重采样缓冲区大小 / 输出采样率

4. **内存开销**：
   - 重采样器需要额外的缓冲区
   ```rust
   // capture.rs:982
   let buf = r.output_buffer_allocate(true);
   ```

---

## 3. 重采样算法

### 3.1 使用的重采样库

**代码使用 `rubato` 库进行重采样**：

```rust
use rubato::{FftFixedIn, FftFixedOut, Resampler};
```

- **`FftFixedIn`**: 固定输入采样率，可变输出采样率（用于输出重采样）
- **`FftFixedOut`**: 固定输出采样率，可变输入采样率（用于输入重采样）

### 3.2 重采样质量

- **`rubato` 使用 FFT 重采样**，质量较高
- **支持抗混叠滤波**，避免混叠失真
- **适合实时处理**，延迟可控

---

## 4. 性能影响

### 4.1 重采样耗时

**预估耗时**：
- **48kHz → 24kHz**: 1-3ms/帧
- **48kHz → 16kHz**: 2-5ms/帧
- **48kHz → 44.1kHz**: 1-2ms/帧

**影响因素**：
- 采样率比（比例越大，耗时越长）
- 重采样算法复杂度
- CPU 性能

### 4.2 对整体性能的影响

**如果只开启降噪/AGC/高通**：
- 核心处理：16-28ms
- 输出重采样（如果存在）：1-5ms
- **总计：17-33ms**

**如果性能已经紧张**（如 47.57ms 超预算）：
- 重采样会进一步增加延迟
- 可能导致更频繁的掉帧

---

## 5. 优化建议

### 5.1 避免重采样（最佳方案）

**使用支持 48kHz 的设备**：
- 大多数现代音频设备都支持 48kHz
- 如果设备支持，代码会自动使用 48kHz，无需重采样

**检查设备采样率**：
```rust
// 代码会打印实际设备采样率
log::info!(
    "Output device '{}' @ {} Hz (internal processing {} Hz)",
    out_name,
    sink.sr(),  // 实际设备采样率
    PROCESS_SR // 48kHz
);
```

### 5.2 如果必须重采样

**优化重采样性能**：
1. **使用更高效的重采样算法**（如果 `rubato` 支持）
2. **降低重采样质量**（如果音质要求不高）
3. **使用 SIMD 优化**（如果 `rubato` 支持）

**减少重采样频率**：
- 如果性能紧张，可以考虑降低处理频率（如每 2 帧处理一次）
- 但这会影响实时性，不推荐

### 5.3 性能监控

**添加重采样性能统计**：
```rust
let mut resample_time_ms = 0.0f32;

if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
    let t_resample = Instant::now();
    if let Err(err) = r.process_into_buffer(&[slice], buf, None) {
        log::error!("输出重采样失败: {:?}", err);
    } else {
        push_output_block(&should_stop, &mut rb_out, &buf[0][..n_out], n_out);
    }
    resample_time_ms = t_resample.elapsed().as_secs_f32() * 1000.0;
}

// 记录重采样耗时
if elapsed_ms > budget_ms * 1.5 {
    log::warn!(
        "输出重采样耗时: {:.2} ms",
        resample_time_ms
    );
}
```

---

## 6. 总结

### 6.1 输出重采样是否必要？

**答案**：取决于设备采样率

- **如果设备支持 48kHz**：**无需重采样**（代码会自动使用 48kHz）
- **如果设备不支持 48kHz**：**需要重采样**（代码会自动创建重采样器）

### 6.2 48kHz → 24kHz 会发生什么？

**答案**：代码会自动处理，但会有性能开销

1. ✅ **设备会被配置为 24kHz**
2. ✅ **自动创建 48kHz → 24kHz 重采样器**
3. ✅ **运行时自动重采样**
4. ⚠️ **性能开销：1-5ms/帧**
5. ⚠️ **音质损失：丢失 > 12kHz 高频信息**
6. ⚠️ **延迟增加：重采样缓冲区延迟**

### 6.3 最佳实践

1. **优先使用支持 48kHz 的设备**（避免重采样）
2. **如果必须重采样，监控性能影响**
3. **如果性能紧张，考虑优化重采样或使用更高效的重采样算法**

---

## 7. 相关代码位置

- **重采样器创建**: `capture.rs:978-998`
- **重采样执行**: `capture.rs:1930-1947`
- **设备采样率配置**: `capture.rs:379-455`
- **延迟计算**: `capture.rs:1003-1005`

