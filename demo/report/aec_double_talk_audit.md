# AEC 双讲吞音问题审计报告

## 问题描述

用户反馈：**AEC 功能 OK，但双讲时会吞音（本地语音断断续续）**

需要检查：
1. 双讲检测逻辑是否正确
2. 双讲时的抑制级别是否合适
3. 双讲保护窗口是否足够
4. 可能导致吞音的其他原因

---

## 问题列表

### 🔴 **AEC-DT-BUG-001: 双讲保护窗口过短**

**位置**: `capture.rs:1986`

**问题**:
```rust
if vad_state {
    dt_bypass_frames = 4; // 约 40ms 双讲保护窗口
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**分析**:
- 双讲保护窗口只有 **4 帧 = 40ms**（10ms/帧）
- 这个窗口太短，无法覆盖完整的语音段
- 语音通常持续 100-500ms，40ms 的保护窗口只能覆盖语音的一小部分
- 当保护窗口结束后，AEC 会立即恢复高抑制，导致后续语音被吞

**影响**:
- 双讲时，本地语音会被频繁地吞音
- 语音断断续续

**修复建议**:
```rust
// 增加双讲保护窗口
if vad_state {
    dt_bypass_frames = 20; // 约 200ms 双讲保护窗口（更长的保护）
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**或者使用更智能的衰减**:
```rust
// 使用指数衰减，而不是线性衰减
if vad_state {
    dt_bypass_frames = 30; // 约 300ms 双讲保护窗口
} else if dt_bypass_frames > 0 {
    // 指数衰减：前 10 帧快速衰减，后续慢速衰减
    if dt_bypass_frames > 10 {
        dt_bypass_frames -= 1; // 快速衰减
    } else {
        // 慢速衰减，每 2 帧减 1
        if frame_counter % 2 == 0 {
            dt_bypass_frames -= 1;
        }
    }
}
```

**优先级**: **高**

---

### 🟡 **AEC-DT-BUG-002: 双讲检测依赖 VAD，但 VAD 可能未开启**

**位置**: `capture.rs:1985`

**问题**:
```rust
if vad_state {
    dt_bypass_frames = 4;
}
```

**分析**:
- 双讲检测完全依赖 `vad_state`
- 如果 VAD 未开启（`vad_enabled = false`），`vad_state` 始终为 `false`
- 双讲保护永远不会触发，AEC 会一直使用高抑制级别
- 导致双讲时本地语音被吞

**影响**:
- 如果 VAD 未开启，双讲保护完全失效
- 双讲时本地语音会被吞

**修复建议**:
```rust
// 添加基于能量/SNR 的备用双讲检测
let energy_based_voice = rms_db > -40.0 && snr_db > 10.0;
let dt_detected = vad_state || (vad_enabled == false && energy_based_voice);

if dt_detected {
    dt_bypass_frames = 20; // 约 200ms 双讲保护窗口
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**优先级**: **中**

---

### 🟡 **AEC-DT-BUG-003: 双讲时抑制级别可能仍然过高**

**位置**: `audio/aec.rs:72-79`

**问题**:
```rust
let suppression = if self.double_talk {
    // 双讲时使用更温和的抑制，保护近端语音
    EchoCancellationSuppressionLevel::Low
} else if self.aggressive_base {
    EchoCancellationSuppressionLevel::High
} else {
    EchoCancellationSuppressionLevel::Moderate
};
```

**分析**:
- 双讲时使用 `Low` 抑制级别，这是正确的
- 但如果 `aggressive_base = true`（强力模式），在非双讲时使用 `High` 抑制级别
- 当双讲保护窗口结束后，AEC 会立即恢复到 `High` 抑制级别
- 这可能导致语音被突然吞掉

**影响**:
- 双讲保护窗口结束后，语音可能被突然吞掉

**修复建议**:
```rust
// 双讲保护窗口结束后，使用更温和的过渡
let suppression = if self.double_talk {
    EchoCancellationSuppressionLevel::Low
} else if self.aggressive_base {
    // 如果刚从双讲状态退出，使用 Moderate 而不是 High
    EchoCancellationSuppressionLevel::Moderate  // 改为 Moderate
} else {
    EchoCancellationSuppressionLevel::Moderate
};
```

**或者添加过渡期**:
```rust
// 添加双讲退出后的过渡期
let mut double_talk_exit_frames: u16 = 0;

// 在双讲退出时
if self.double_talk && !new_double_talk {
    double_talk_exit_frames = 10; // 约 100ms 过渡期
}

let suppression = if self.double_talk || double_talk_exit_frames > 0 {
    EchoCancellationSuppressionLevel::Low
} else if self.aggressive_base {
    EchoCancellationSuppressionLevel::High
} else {
    EchoCancellationSuppressionLevel::Moderate
};
```

**优先级**: **中**

---

### 🟡 **AEC-DT-BUG-004: VAD 检测延迟导致双讲检测滞后**

**位置**: `capture.rs:1391-1400`

**问题**:
```rust
if vad_enabled && vad_source_frame > 0 && vad_buf_raw.len() >= vad_source_frame {
    // VAD 处理
    // ...
    vad_voice = v.is_speaking();
}
```

**分析**:
- VAD 需要积累 `vad_source_frame`（480 样本，30ms @16kHz）才能处理
- VAD 处理本身需要时间（ONNX 推理，约 8-15ms）
- 双讲检测有延迟，可能导致语音已经开始但双讲保护还未触发
- 语音的前几个帧可能被吞掉

**影响**:
- 语音起音时可能被吞掉
- 双讲检测滞后

**修复建议**:
```rust
// 添加基于能量的快速双讲检测（不依赖 VAD）
let energy_voice = rms_db > -35.0 && snr_db > 8.0;
let fast_dt_detected = energy_voice && render_active;

// 快速触发双讲保护
if fast_dt_detected {
    dt_bypass_frames = 20;
} else if vad_state {
    dt_bypass_frames = 20.max(dt_bypass_frames); // 保持或延长保护窗口
}
```

**优先级**: **中**

---

### 🟢 **AEC-DT-BUG-005: 双讲检测没有考虑 render 信号的能量**

**位置**: `capture.rs:1985`

**问题**:
```rust
if vad_state {
    dt_bypass_frames = 4;
}
```

**分析**:
- 双讲检测只检查了本地语音（`vad_state`）
- 没有检查 render 信号的能量
- 如果 render 信号很弱，可能不是真正的双讲
- 如果 render 信号很强，应该更积极地保护本地语音

**修复建议**:
```rust
// 检查 render 信号的能量
let render_energy = calculate_render_energy(buffer); // 需要实现
let render_active = render_energy > -50.0; // render 信号存在

// 只有在 render 信号存在时才触发双讲保护
if vad_state && render_active {
    dt_bypass_frames = 20;
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**优先级**: **低**

---

### 🟢 **AEC-DT-BUG-006: 双讲保护窗口固定，没有根据语音特征调整**

**位置**: `capture.rs:1986`

**问题**:
```rust
if vad_state {
    dt_bypass_frames = 4; // 固定 40ms
}
```

**分析**:
- 双讲保护窗口固定为 40ms
- 不同语音特征（能量、持续时间等）可能需要不同的保护窗口
- 高能量语音可能需要更长的保护窗口

**修复建议**:
```rust
// 根据语音能量调整保护窗口
let energy_factor = (rms_db + 60.0) / 30.0; // 归一化到 0-1
let base_window = 20; // 基础窗口 200ms
let adaptive_window = (base_window as f32 * (1.0 + energy_factor * 0.5)) as u16; // 最多 300ms

if vad_state {
    dt_bypass_frames = adaptive_window.max(dt_bypass_frames); // 保持或延长
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**优先级**: **低**

---

## 问题汇总

### 高优先级问题

1. **AEC-DT-BUG-001**: 双讲保护窗口过短（只有 40ms，应该至少 200ms）

### 中优先级问题

2. **AEC-DT-BUG-002**: 双讲检测依赖 VAD，但 VAD 可能未开启
3. **AEC-DT-BUG-003**: 双讲时抑制级别可能仍然过高（强力模式下）
4. **AEC-DT-BUG-004**: VAD 检测延迟导致双讲检测滞后

### 低优先级问题

5. **AEC-DT-BUG-005**: 双讲检测没有考虑 render 信号的能量
6. **AEC-DT-BUG-006**: 双讲保护窗口固定，没有根据语音特征调整

---

## 优化方向

### 方向 1: 增加双讲保护窗口（立即修复）

**修改**: `capture.rs:1986`

**当前**:
```rust
dt_bypass_frames = 4; // 40ms
```

**修复**:
```rust
dt_bypass_frames = 20; // 200ms（推荐）
// 或
dt_bypass_frames = 30; // 300ms（更保守）
```

**预期效果**: 显著减少双讲时的吞音

---

### 方向 2: 添加基于能量的备用双讲检测

**修改**: `capture.rs:1985`

**当前**:
```rust
if vad_state {
    dt_bypass_frames = 4;
}
```

**修复**:
```rust
// 添加基于能量的备用检测
let energy_voice = rms_db > -40.0 && snr_db > 10.0;
let dt_detected = vad_state || (vad_enabled == false && energy_voice);

if dt_detected {
    dt_bypass_frames = 20;
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1;
}
```

**预期效果**: 即使 VAD 未开启，也能检测双讲

---

### 方向 3: 优化双讲退出后的过渡

**修改**: `audio/aec.rs:72-79`

**当前**:
```rust
let suppression = if self.double_talk {
    EchoCancellationSuppressionLevel::Low
} else if self.aggressive_base {
    EchoCancellationSuppressionLevel::High  // 立即恢复到 High
} else {
    EchoCancellationSuppressionLevel::Moderate
};
```

**修复**:
```rust
// 添加过渡期
let suppression = if self.double_talk {
    EchoCancellationSuppressionLevel::Low
} else if self.aggressive_base {
    // 如果刚从双讲退出，使用 Moderate 而不是 High
    EchoCancellationSuppressionLevel::Moderate
} else {
    EchoCancellationSuppressionLevel::Moderate
};
```

**预期效果**: 双讲保护窗口结束后，不会突然吞音

---

### 方向 4: 使用指数衰减延长保护窗口

**修改**: `capture.rs:1987-1989`

**当前**:
```rust
} else if dt_bypass_frames > 0 {
    dt_bypass_frames -= 1; // 线性衰减
}
```

**修复**:
```rust
} else if dt_bypass_frames > 0 {
    // 指数衰减：前 10 帧快速衰减，后续慢速衰减
    if dt_bypass_frames > 10 {
        dt_bypass_frames -= 1; // 快速衰减
    } else {
        // 慢速衰减，每 2 帧减 1
        if frame_counter % 2 == 0 {
            dt_bypass_frames -= 1;
        }
    }
}
```

**预期效果**: 保护窗口更平滑地退出，减少突然吞音

---

## 诊断步骤

### 步骤 1: 检查双讲保护窗口

**查看日志**，确认双讲保护窗口的大小：
```
双讲旁路中，剩余 XX ms（参考 存在/缺失）
```

**如果窗口只有 40ms**:
- 这是问题所在，需要增加到 200-300ms

### 步骤 2: 检查 VAD 是否开启

**查看日志**，确认 VAD 状态：
```
Silero VAD: 开启/关闭
```

**如果 VAD 未开启**:
- 双讲保护完全失效
- 需要添加基于能量的备用检测

### 步骤 3: 检查 AEC 抑制级别

**查看日志**，确认 AEC 状态：
```
AEC3 状态: 开启 (render参考 存在，双讲旁路 XXms)
```

**如果强力模式开启**:
- 双讲保护窗口结束后，会立即恢复到 High 抑制级别
- 可能导致突然吞音

---

## 相关代码位置

- **双讲检测**: `capture.rs:1985-1989`
- **双讲保护窗口**: `capture.rs:1986`
- **AEC 抑制级别**: `audio/aec.rs:72-79`
- **VAD 检测**: `capture.rs:1391-1400`
- **双讲状态设置**: `capture.rs:2009-2011`

---

## 总结

**主要问题**: 双讲保护窗口过短（只有 40ms），无法覆盖完整的语音段。

**优化方向**:
1. **增加双讲保护窗口**（从 40ms 增加到 200-300ms）
2. **添加基于能量的备用双讲检测**（不依赖 VAD）
3. **优化双讲退出后的过渡**（避免突然吞音）
4. **使用指数衰减**（更平滑地退出保护窗口）

**预期效果**: 修复后，双讲时的吞音问题应该显著改善。



