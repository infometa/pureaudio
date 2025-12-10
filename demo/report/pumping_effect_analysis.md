# 抽吸感（Pumping Effect）问题分析报告

## 问题描述

用户反馈：**当前代码有抽吸感，声音一会高一会低**

这是典型的音频处理中的 **"pumping"** 或 **"breathing"** 效果，表现为音量快速、周期性地变化。

---

## 1. 根本原因分析

### 🔴 **PUMP-001: 环境自适应参数变化过快**

**位置**: `capture.rs:1602-1609`

**问题**:
```rust
// 更快的参数平滑，兼顾响应与平顺
let alpha_fast = 0.5;  // <--- 每帧移动 50%，非常快！
let alpha_hp = 0.15;  // 高通调节更平滑，避免可闻跳变
let new_atten = smooth_value(current_atten, target_atten, alpha_fast);
df.set_atten_lim(new_atten);
df.min_db_thresh = smooth_value(df.min_db_thresh, target_min_thresh, alpha_fast);
df.max_db_df_thresh = smooth_value(df.max_db_df_thresh, target_max_thresh, alpha_fast);
```

**分析**:
- `alpha_fast = 0.5` 意味着每帧会移动 **50%** 的距离到目标值
- 在 48kHz、10ms 帧长的情况下，**每 10ms 移动 50%**
- 如果 `target_atten` 从 40dB 变化到 70dB，只需要 **约 30ms** 就能完成
- 这种快速变化会直接导致 DeepFilterNet 的增益快速变化，产生抽吸感

**`smooth_value` 函数**:
```rust
// capture.rs:2841-2843
fn smooth_value(current: f32, target: f32, alpha: f32) -> f32 {
    current + (target - current) * alpha.clamp(0.0, 1.0)
}
```

**影响**:
- `target_atten` 快速变化 → DeepFilterNet 降噪强度快速变化 → 输出电平快速变化
- `target_min_thresh` 和 `target_max_thresh` 快速变化 → DeepFilterNet 阈值快速变化 → 增益快速变化

**优先级**: **高**（主要原因）

---

### 🔴 **PUMP-002: 环境特征平滑系数动态变化**

**位置**: `capture.rs:1162-1178`

**问题**:
```rust
let (rms_db, update_alpha) = if let Some(buf) = inframe.as_slice() {
    let rms = df::rms(buf.iter());
    let db = 20.0 * rms.max(1e-9).log10();
    // 提高自适应平滑系数，缩短响应但避免突变
    let alpha = if db < -50.0 {
        0.55  // <--- 低电平，快速响应
    } else if db < -30.0 {
        0.45
    } else if db < -20.0 {
        0.35
    } else {
        0.25  // <--- 高电平，慢速响应
    };
    (db, alpha)
} else {
    (-60.0, 0.3)
};
```

**分析**:
- `update_alpha` 根据输入电平动态调整（0.25-0.55）
- 当输入电平在阈值附近波动时，`update_alpha` 会在不同值之间切换
- 这导致 `smoothed_energy`、`smoothed_flatness`、`smoothed_centroid` 的平滑速度不一致
- 进而导致 `target_atten` 等参数的变化速度不一致，产生抽吸感

**优先级**: **高**（主要原因）

---

### 🟡 **PUMP-003: WebRTC AGC 响应过快**

**位置**: `audio/agc.rs:18-23`

**问题**:
```rust
let cfg = GainControl {
    mode: GainControlMode::AdaptiveDigital,
    target_level_dbfs: 6,      // -6 dBFS 目标，更响亮
    compression_gain_db: 20,   // 更强补偿低音量
    enable_limiter: false,     // 内限幅关闭，交给最终限幅器
};
```

**分析**:
- WebRTC AGC 的 `compression_gain_db: 20` 很强，会快速补偿低音量
- `target_level_dbfs: 6` 目标电平较高（-6 dBFS）
- **无法调整攻击/释放时间**（`set_attack_release` 是空的）
- 当 DeepFilterNet 的输出电平快速变化时，AGC 会快速响应，进一步加剧抽吸感

**优先级**: **中**

---

### 🟡 **PUMP-004: 冲击检测导致的突变**

**位置**: `capture.rs:1575-1594`

**问题**:
```rust
if impact_hold > 0 {
    // 键盘/点击/关门：瞬时提高抑制和高通，关闭激励
    target_atten = (target_atten + 12.0).min(75.0);  // <--- 突然增加 12dB
    target_hp = target_hp.max(180.0);
    target_exciter_mix = 0.0;
    transient_shaper.set_attack_gain(-6.0);
} else if breath_hold > 0 {
    // 急促呼吸/摩擦：提高抑制和高通，去除高频激励
    target_atten = (target_atten + 18.0).min(95.0);  // <--- 突然增加 18dB
    target_hp = target_hp.max(200.0);
    // ...
} else {
    // 恢复用户设定的瞬态增益
    transient_shaper.set_attack_gain(transient_attack_db);
}
```

**分析**:
- 当检测到冲击或呼吸时，`target_atten` 会突然增加 12-18dB
- 即使有 `alpha_fast = 0.5` 的平滑，这种突变仍然会导致音量快速变化
- 当冲击/呼吸结束后，`target_atten` 又会快速恢复，产生抽吸感

**优先级**: **中**

---

### 🟡 **PUMP-005: 多级增益控制冲突**

**位置**: `capture.rs:1737-1752`

**问题**:
```rust
// 输出增益在 AGC 前统一设置，然后交给 AGC 控制最终电平
if let Some(buffer) = outframe.as_slice_mut() {
    let mut out_gain = post_trim_gain * headroom_gain;
    if out_gain > 1.0 {
        out_gain = 1.0;
    }
    if (out_gain - 1.0).abs() > 1e-6 {
        for v in buffer.iter_mut() {
            *v *= out_gain;  // <--- 第一级增益
        }
    }
    if agc_enabled {
        agc.process(buffer);  // <--- 第二级增益（AGC）
    }
}
```

**分析**:
- 第一级：`out_gain = post_trim_gain * headroom_gain`（固定增益）
- 第二级：AGC（动态增益）
- 如果环境自适应导致 DeepFilterNet 的输出电平快速变化，AGC 会快速响应
- 两级增益控制叠加，可能导致增益变化过快，产生抽吸感

**优先级**: **中**

---

### 🟢 **PUMP-006: 动态 EQ 增益变化**

**位置**: `audio/eq/dynamic_band.rs:142-163`

**问题**:
```rust
pub fn update(&mut self, rms: f32, block_len: usize) {
    // 块级 RMS 做 EMA 平滑，降低对块长的敏感度
    let alpha_rms = smoothing_coeff(50.0, block_len, self.sample_rate); // ~50ms 平滑
    // ...
    let target = self.target_gain(env_db);
    let coeff = if target > self.current_gain_db {
        smoothing_coeff(self.settings.attack_ms, block_len, self.sample_rate)
    } else {
        smoothing_coeff(self.settings.release_ms, block_len, self.sample_rate)
    };
    self.current_gain_db += coeff * (target - self.current_gain_db);
}
```

**分析**:
- 动态 EQ 的每个频段都有独立的增益控制
- 如果攻击/释放时间设置不当，可能导致增益变化过快
- 多个频段同时变化，可能产生抽吸感

**优先级**: **低**（如果动态 EQ 未开启，不影响）

---

## 2. 解决方案

### 2.1 立即修复（高优先级）

#### **FIX-001: 降低环境自适应参数变化速度**

**位置**: `capture.rs:1602-1609`

**当前代码**:
```rust
let alpha_fast = 0.5;  // 每帧移动 50%，非常快
```

**修复方案**:
```rust
// 降低参数变化速度，避免抽吸感
let alpha_fast = 0.15;  // 每帧移动 15%，更平滑（约 67ms 完成 90% 变化）
let alpha_hp = 0.10;    // 高通调节更平滑
```

**预期效果**: 参数变化速度降低 **70%**，显著减少抽吸感

**优先级**: **高**

---

#### **FIX-002: 固定环境特征平滑系数**

**位置**: `capture.rs:1162-1178`

**当前代码**:
```rust
let alpha = if db < -50.0 {
    0.55
} else if db < -30.0 {
    0.45
} else if db < -20.0 {
    0.35
} else {
    0.25
};
```

**修复方案**:
```rust
// 固定平滑系数，避免因输入电平波动导致平滑速度不一致
let alpha = 0.25;  // 统一使用较慢的平滑速度
```

**或者**:
```rust
// 使用更平滑的过渡，避免突然切换
let alpha = if db < -50.0 {
    0.30  // 降低快速响应
} else if db < -30.0 {
    0.25
} else if db < -20.0 {
    0.20
} else {
    0.15  // 降低慢速响应
};
```

**预期效果**: 平滑速度一致，避免因输入电平波动导致的抽吸感

**优先级**: **高**

---

### 2.2 进一步优化（中优先级）

#### **FIX-003: 平滑冲击检测的突变**

**位置**: `capture.rs:1575-1594`

**当前代码**:
```rust
if impact_hold > 0 {
    target_atten = (target_atten + 12.0).min(75.0);  // 突然增加 12dB
}
```

**修复方案**:
```rust
// 平滑冲击检测的突变，避免突然变化
if impact_hold > 0 {
    // 使用平滑的目标值，而不是直接增加
    let impact_atten = (target_atten + 12.0).min(75.0);
    target_atten = smooth_value(target_atten, impact_atten, 0.3);  // 平滑过渡
    target_hp = smooth_value(target_hp, target_hp.max(180.0), 0.3);
    target_exciter_mix = smooth_value(target_exciter_mix, 0.0, 0.3);
    transient_shaper.set_attack_gain(-6.0);
} else if breath_hold > 0 {
    let breath_atten = (target_atten + 18.0).min(95.0);
    target_atten = smooth_value(target_atten, breath_atten, 0.3);  // 平滑过渡
    target_hp = smooth_value(target_hp, target_hp.max(200.0), 0.3);
    // ...
}
```

**预期效果**: 冲击/呼吸检测导致的突变更平滑，减少抽吸感

**优先级**: **中**

---

#### **FIX-004: 降低 WebRTC AGC 的响应强度**

**位置**: `audio/agc.rs:18-23`

**当前代码**:
```rust
let cfg = GainControl {
    target_level_dbfs: 6,      // -6 dBFS 目标
    compression_gain_db: 20,   // 更强补偿低音量
};
```

**修复方案**:
```rust
let cfg = GainControl {
    target_level_dbfs: 3,      // 降低目标电平到 -3 dBFS，减少过度补偿
    compression_gain_db: 12,   // 降低补偿强度，减少快速响应
    enable_limiter: false,
};
```

**预期效果**: AGC 响应更温和，减少抽吸感

**优先级**: **中**

---

### 2.3 长期优化（低优先级）

#### **FIX-005: 添加增益变化速率限制**

**位置**: `capture.rs:1605-1609`

**修复方案**:
```rust
// 限制增益变化速率，避免过快变化
let max_atten_change = 3.0;  // 每帧最多变化 3dB
let new_atten = smooth_value(current_atten, target_atten, alpha_fast);
let delta = (new_atten - current_atten).clamp(-max_atten_change, max_atten_change);
df.set_atten_lim(current_atten + delta);
```

**预期效果**: 强制限制增益变化速率，避免过快变化

**优先级**: **低**

---

## 3. 诊断建议

### 3.1 添加增益变化监控

**位置**: `capture.rs:1605-1609`

**建议**:
```rust
// 监控增益变化，便于诊断抽吸感
let new_atten = smooth_value(current_atten, target_atten, alpha_fast);
let atten_change = (new_atten - current_atten).abs();
if atten_change > 5.0 && perf_last_log.elapsed() > Duration::from_millis(500) {
    log::warn!(
        "降噪强度快速变化: {:.1} dB -> {:.1} dB (变化 {:.1} dB)",
        current_atten,
        new_atten,
        atten_change
    );
}
df.set_atten_lim(new_atten);
```

---

### 3.2 检查环境自适应是否开启

**建议**: 检查 `env_auto_enabled` 是否开启。如果未开启，问题可能在其他地方。

---

## 4. 实施优先级

### 第一阶段：立即修复（预期减少 70-80% 抽吸感）

1. ✅ **FIX-001**: 降低环境自适应参数变化速度（`alpha_fast: 0.5 → 0.15`）
2. ✅ **FIX-002**: 固定环境特征平滑系数（统一使用 0.25 或更慢）

### 第二阶段：进一步优化（预期减少 10-20% 抽吸感）

3. **FIX-003**: 平滑冲击检测的突变
4. **FIX-004**: 降低 WebRTC AGC 的响应强度

### 第三阶段：长期优化（预期减少 5-10% 抽吸感）

5. **FIX-005**: 添加增益变化速率限制

---

## 5. 关键发现总结

### 5.1 主要原因

1. **环境自适应参数变化过快** (`alpha_fast = 0.5`)
   - 每帧移动 50%，导致增益快速变化
   - **这是最主要的原因**

2. **环境特征平滑系数动态变化**
   - 根据输入电平动态调整，导致平滑速度不一致
   - **这是次要原因**

### 5.2 次要原因

3. **WebRTC AGC 响应过快**
   - 无法调整攻击/释放时间，响应可能过快

4. **冲击检测导致的突变**
   - 突然增加 12-18dB，即使有平滑也会产生抽吸感

5. **多级增益控制冲突**
   - 两级增益控制叠加，可能导致变化过快

### 5.3 最容易修复的点

1. **降低 `alpha_fast`** - 简单有效，预期减少 70-80% 抽吸感
2. **固定环境特征平滑系数** - 简单有效，预期减少 10-20% 抽吸感

---

## 6. 总结

**抽吸感的主要原因是环境自适应参数变化过快**：

- `alpha_fast = 0.5` 每帧移动 50%，导致增益快速变化
- 环境特征平滑系数动态变化，导致平滑速度不一致
- 这些快速变化导致 DeepFilterNet 的增益快速变化，产生抽吸感

**最有效的修复方案**：

1. **降低 `alpha_fast` 从 0.5 到 0.15** - 参数变化速度降低 70%
2. **固定环境特征平滑系数** - 统一使用 0.25 或更慢

**预期效果**：修复后，抽吸感应该减少 **70-90%**。

---

## 7. 相关代码位置

- **环境自适应参数平滑**: `capture.rs:1602-1609`
- **环境特征平滑系数**: `capture.rs:1162-1178`
- **冲击检测突变**: `capture.rs:1575-1594`
- **WebRTC AGC 配置**: `audio/agc.rs:18-23`
- **多级增益控制**: `capture.rs:1737-1752`
- **smooth_value 函数**: `capture.rs:2841-2843`

