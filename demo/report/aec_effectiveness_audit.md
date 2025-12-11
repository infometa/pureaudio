# AEC（回声消除）效果不佳问题审计报告

## 问题列表

### 🔴 **AEC-BUG-001: 延迟估算不准确**

**位置**: `capture.rs:1028-1030`

**问题**:
```rust
let auto_aec_delay =
    ((block_duration * 1000.0) + resample_latency_ms + 5.0).round().clamp(0.0, 200.0);
```

**分析**:
- 延迟估算只考虑了 `block_duration`（10ms）、`resample_latency_ms` 和固定偏移 5.0ms
- **没有考虑**：
  - 音频设备延迟（输入/输出缓冲区）
  - 系统延迟（驱动、内核）
  - DeepFilterNet 处理延迟
  - 其他处理延迟
- 固定偏移 5.0ms 可能不够，实际延迟通常更大

**影响**:
- 如果实际延迟 > 估算延迟，AEC 无法正确对齐 render 和 capture 信号
- 导致回声消除效果不佳

**修复建议**:
```rust
// 改进延迟估算，考虑更多因素
let base_delay = block_duration * 1000.0;  // 10ms
let resample_delay = resample_latency_ms;
let processing_delay = 5.0;  // 处理延迟估算
let device_delay = 10.0;  // 设备延迟估算（输入+输出缓冲区）
let system_delay = 5.0;  // 系统延迟估算（驱动、内核）
let auto_aec_delay = (base_delay + resample_delay + processing_delay + device_delay + system_delay)
    .round()
    .clamp(0.0, 500.0);  // 提高上限到 500ms
```

**或者更简单**:
```rust
// 使用更保守的估算
let auto_aec_delay = ((block_duration * 1000.0) + resample_latency_ms + 20.0)
    .round()
    .clamp(0.0, 500.0);  // 固定偏移从 5ms 增加到 20ms，上限提高到 500ms
```

**优先级**: **高**

---

### 🔴 **AEC-BUG-002: 帧长不匹配导致 AEC 旁路**

**位置**: `audio/aec.rs:19-20`

**问题**:
```rust
let frame_size = (sample_rate / 100.0).round() as usize; // 10ms 帧
let frame_ok = frame_size > 0 && hop_size % frame_size == 0;
```

**分析**:
- AEC 需要 10ms 帧长（`frame_size = sample_rate / 100`）
- 如果 `hop_size` 不是 `frame_size` 的整数倍，AEC 会旁路
- 在 48kHz 下，`frame_size = 480`，`hop_size` 必须是 480 的倍数
- 如果模型使用其他 `hop_size`（如 960），AEC 会旁路，回声消除完全不起作用

**影响**:
- 如果 AEC 旁路，回声消除完全不起作用

**修复建议**:
```rust
let frame_size = (sample_rate / 100.0).round() as usize; // 10ms 帧
let frame_ok = frame_size > 0 && hop_size % frame_size == 0;
if !frame_ok {
    error!(
        "AEC3 无法初始化：hop_size={} 不是 frame_size={} 的整数倍。请使用 10ms 整数倍的 hop_size。",
        hop_size, frame_size
    );
}
```

**优先级**: **高**

---

### 🟡 **AEC-BUG-003: process_render 在最终限幅器前**

**位置**: `capture.rs:1960-1970`

**问题**:
```rust
if aec_enabled {
    aec.process_render(buffer);  // <--- 在最终限幅器前
}
// 最终限幅一次，避免多级限幅导致音色压缩
if final_limiter_enabled {
    apply_final_limiter(buffer);  // <--- 在 AEC 后
}
```

**分析**:
- `process_render` 在最终限幅器前调用
- 这意味着 AEC 收到的 render 信号可能包含峰值
- 如果最终限幅器改变了信号形状，AEC 收到的 render 信号可能与实际播放的信号不同
- 导致 AEC 无法正确匹配回声

**影响**:
- 如果最终限幅器改变了信号，AEC 可能无法正确匹配回声

**修复建议**:
```rust
// 先应用最终限幅器
if final_limiter_enabled {
    apply_final_limiter(buffer);
}
// 然后发送给 AEC（与实际播放的信号一致）
if aec_enabled {
    aec.process_render(buffer);  // 在最终限幅器后
}
```

**注意**: 需要确保 `process_render` 不会修改 buffer（当前实现是只读的，应该没问题）

**优先级**: **中**

---

### 🟡 **AEC-BUG-004: 延迟设置上限 200ms 可能不够**

**位置**: `capture.rs:1029`

**问题**:
```rust
let auto_aec_delay =
    ((block_duration * 1000.0) + resample_latency_ms + 5.0).round().clamp(0.0, 200.0);
```

**分析**:
- 延迟上限设置为 200ms
- 在某些系统中，实际延迟可能 > 200ms（特别是使用蓝牙设备时）
- 如果实际延迟 > 200ms，AEC 无法正确对齐信号

**影响**:
- 如果实际延迟 > 200ms，AEC 效果会变差

**修复建议**:
```rust
let auto_aec_delay =
    ((block_duration * 1000.0) + resample_latency_ms + 20.0).round().clamp(0.0, 500.0);
```

**优先级**: **中**

---

### 🟢 **AEC-BUG-005: 没有延迟自适应调整**

**位置**: `capture.rs:1028-1030`

**问题**:
- 延迟只在初始化时计算一次
- 如果实际延迟在运行时变化（如设备切换、缓冲区变化），AEC 无法适应

**分析**:
- WebRTC AEC3 支持 `enable_delay_agnostic: true`，可以自动适应延迟
- 但初始延迟设置不准确，可能影响自适应效果

**修复建议**:
```rust
// 定期重新估算延迟（如每 10 秒）
let mut last_delay_update = Instant::now();
const DELAY_UPDATE_INTERVAL: Duration = Duration::from_secs(10);

if last_delay_update.elapsed() > DELAY_UPDATE_INTERVAL {
    let new_delay = ((block_duration * 1000.0) + resample_latency_ms + 20.0)
        .round()
        .clamp(0.0, 500.0);
    if (new_delay - aec_delay_ms as f32).abs() > 5.0 {
        aec_delay_ms = new_delay as i32;
        aec.set_delay_ms(aec_delay_ms);
        log::info!("AEC 延迟自适应调整: {} ms", aec_delay_ms);
    }
    last_delay_update = Instant::now();
}
```

**优先级**: **低**

---

### 🟢 **AEC-BUG-006: 延迟估算没有考虑 DeepFilterNet 处理延迟**

**位置**: `capture.rs:1028-1030`

**问题**:
- 延迟估算没有考虑 DeepFilterNet 的处理延迟
- DeepFilterNet 处理需要时间（通常 10-20ms），这会影响 render 和 capture 信号的对齐

**分析**:
- DeepFilterNet 在 capture 路径上，会增加 capture 信号的延迟
- 如果 render 信号没有相应的延迟，AEC 无法正确对齐

**修复建议**:
```rust
// 考虑 DeepFilterNet 处理延迟（估算 10-15ms）
let df_processing_delay = 12.0;  // DeepFilterNet 处理延迟估算
let auto_aec_delay = ((block_duration * 1000.0) + resample_latency_ms + df_processing_delay + 20.0)
    .round()
    .clamp(0.0, 500.0);
```

**优先级**: **低**

---

## 问题汇总

### 高优先级问题

1. **AEC-BUG-001**: 延迟估算不准确（只考虑了部分延迟）
2. **AEC-BUG-002**: 帧长不匹配导致 AEC 旁路（如果 `hop_size` 不是 10ms 的整数倍）

### 中优先级问题

3. **AEC-BUG-003**: process_render 在最终限幅器前（可能导致信号不匹配）
4. **AEC-BUG-004**: 延迟设置上限 200ms 可能不够（蓝牙设备延迟可能更大）

### 低优先级问题

5. **AEC-BUG-005**: 没有延迟自适应调整（延迟只在初始化时计算一次）
6. **AEC-BUG-006**: 延迟估算没有考虑 DeepFilterNet 处理延迟

---

## 诊断步骤

### 步骤 1: 检查 AEC 是否激活

**查看日志**，确认是否有以下警告：
```
AEC3 未激活（检查帧长/初始化），当前旁路
```

**如果看到这个警告**:
- AEC 没有激活，回声消除完全不起作用
- 检查 `hop_size` 是否是 10ms 的整数倍（在 48kHz 下，必须是 480 的倍数）

### 步骤 2: 检查延迟设置

**查看日志**，确认延迟估算值：
```
估算链路延迟 XX.XX ms (DF hop XX.XX ms, AEC 延迟 XX ms, 重采样 XX.XX ms)
```

**如果延迟估算不准确**:
- 尝试手动调整延迟（通过 UI）
- 测试不同延迟值，找到最佳效果

### 步骤 3: 测试不同延迟值

1. 从估算值开始
2. 逐步增加（+10ms、+20ms、+30ms...）
3. 找到最佳效果

---

## 相关代码位置

- **延迟估算**: `capture.rs:1028-1030`
- **帧长检查**: `audio/aec.rs:19-20`
- **process_capture**: `capture.rs:1219-1224`
- **process_render**: `capture.rs:1960-1966`
- **AEC 初始化**: `audio/aec.rs:18-65`
