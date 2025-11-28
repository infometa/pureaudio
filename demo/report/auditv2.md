# 环境自适应功能审计报告

## 问题总结：功能基本无效

你的直觉是对的，环境自适应功能确实**几乎不起作用**。我发现了 **6 个关键 Bug**。

---

## Bug 1: 默认关闭，用户可能不知道要开启

**位置**: `capture.rs:822` 和 `main.rs:675`

```rust
// capture.rs:822
let mut env_auto_enabled = false;  // 默认关闭！

// main.rs:675
env_auto_enabled: false,  // UI 也默认关闭
```

**问题**: 用户如果不手动开启，功能永远不会生效。

**建议**: 默认开启，或在 UI 上更明显地提示用户。

---

## Bug 2: 只在"静默"时才更新环境特征（最严重！）

**位置**: `capture.rs:922-935`

```rust
if env_auto_enabled && !bypass_enabled {
    if let Some(buf) = inframe.as_slice() {
        let rms = df::rms(buf.iter());
        let rms_db = 20.0 * rms.max(1e-9).log10();
        
        // ❌ 只有当信号电平 < -35dB 时才更新环境特征！
        const NOISE_ONLY_DB: f32 = -35.0;
        if rms_db < NOISE_ONLY_DB {
            let feats = compute_noise_features(df.get_spec_noisy());
            smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, 0.1);
            // ...
        }
    }
    // ...后面的分类和参数调整代码...
}
```

**问题**: 
- -35dB 是非常安静的电平
- 只要有人说话或有明显噪声，就**永远不会更新**环境特征
- 初始值会一直保持，导致永远是 `EnvClass::Quiet`

**实际场景**:
```
用户在嘈杂办公室说话:
RMS 通常在 -20dB 到 -10dB
-20dB > -35dB → 条件不满足 → 不更新特征 → 永远判定为 Quiet
```

---

## Bug 3: soft_mode 条件永远无法满足

**位置**: `capture.rs:938`

```rust
let soft_candidate = smoothed_energy < -60.0 && smoothed_centroid < -15.0;
//                                              ^^^^^^^^^^^^^^^^^^^^^^^^
//                                              这个条件永远不可能满足！
```

**问题**: 
- `spectral_centroid` 是归一化到 0~1 范围的
- 条件要求 `centroid < -15.0`，这是不可能的
- `soft_mode` 永远不会被触发

**代码证据** (`capture.rs:1861-1864`):
```rust
let spectral_centroid = if sum_power > 0.0 {
    (weighted_sum / sum_power) / freq_len_f32  // 结果在 0~1 之间
} else {
    0.0
};
```

---

## Bug 4: 平滑系数太小，响应极慢

**位置**: `capture.rs:930-934`

```rust
smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, 0.1);
smoothed_flatness = smooth_value(smoothed_flatness, feats.spectral_flatness, 0.1);
smoothed_centroid = smooth_value(smoothed_centroid, feats.spectral_centroid, 0.1);
```

**问题**:
- `alpha = 0.1` 意味着每次只更新 10%
- 需要约 23 次更新才能达到目标值的 90%
- 假设每 20ms 处理一帧，需要 **460ms** 才能响应一次变化
- 而且由于 Bug 2，更新机会很少，实际需要**几秒甚至几十秒**

---

## Bug 5: 初始值导致冷启动问题

**位置**: `capture.rs:768-770`

```rust
let mut smoothed_energy = -80.0f32;    // 极低能量
let mut smoothed_flatness = 0.0f32;     // 零平坦度
let mut smoothed_centroid = 0.0f32;     // 零重心
```

**问题**:
- `energy_db = -80` 远低于 `-40`，所以 `classify_env` 会判定为 `Quiet`
- 由于 Bug 2，这些值几乎不会被更新
- 系统**永远停留在 Quiet 模式**

**classify_env 逻辑** (`capture.rs:1873-1880`):
```rust
fn classify_env(energy_db: f32, flatness: f32, centroid: f32) -> EnvClass {
    if energy_db > -40.0 {
        EnvClass::Noisy
    } else if flatness > 0.45 || centroid > 0.5 {
        EnvClass::Office
    } else {
        EnvClass::Quiet  // ← 初始值会一直走这里
    }
}
```

---

## Bug 6: 环境切换没有日志，无法调试

**位置**: `capture.rs:964-967`

```rust
let target_env = classify_env(smoothed_energy, smoothed_flatness, smoothed_centroid);
if target_env != env_class {
    env_class = target_env;
    // ❌ 没有 log 输出！不知道是否真的在切换
}
```

**问题**: 你无法知道环境分类是否在工作。

---

## 修复方案

### 修复 1: 改为始终更新特征，只是权重不同

```rust
// 替换原来的逻辑
if env_auto_enabled && !bypass_enabled {
    if let Some(buf) = inframe.as_slice() {
        let rms = df::rms(buf.iter());
        let rms_db = 20.0 * rms.max(1e-9).log10();
        
        let feats = compute_noise_features(df.get_spec_noisy());
        
        // 根据是否有语音，使用不同的平滑系数
        // 有语音时更新慢，静默时更新快
        let alpha = if rms_db < -35.0 {
            0.3   // 静默时快速更新
        } else if rms_db < -20.0 {
            0.1   // 有轻微语音时中速更新
        } else {
            0.02  // 有明显语音时慢速更新（但仍然更新！）
        };
        
        smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, alpha);
        smoothed_flatness = smooth_value(smoothed_flatness, feats.spectral_flatness, alpha);
        smoothed_centroid = smooth_value(smoothed_centroid, feats.spectral_centroid, alpha);
    }
    // ... 后续逻辑
}
```

### 修复 2: 修正 soft_mode 条件

```rust
// 原来的（错误）
let soft_candidate = smoothed_energy < -60.0 && smoothed_centroid < -15.0;

// 修正后
let soft_candidate = smoothed_energy < -55.0 && smoothed_centroid < 0.3;
//                                               ^^^^^^^^^^^^^^^^^
//                                               centroid 是 0~1 范围，低值表示低频为主
```

### 修复 3: 添加环境切换日志

```rust
let target_env = classify_env(smoothed_energy, smoothed_flatness, smoothed_centroid);
if target_env != env_class {
    log::info!(
        "环境自适应: {} → {} (energy={:.1}dB, flatness={:.2}, centroid={:.2})",
        env_class_name(env_class),
        env_class_name(target_env),
        smoothed_energy,
        smoothed_flatness,
        smoothed_centroid
    );
    env_class = target_env;
}

// 辅助函数
fn env_class_name(c: EnvClass) -> &'static str {
    match c {
        EnvClass::Quiet => "安静",
        EnvClass::Office => "办公室",
        EnvClass::Noisy => "嘈杂",
    }
}
```

### 修复 4: 合理的初始值

```rust
// 使用更接近"中等环境"的初始值
let mut smoothed_energy = -50.0f32;    // 中等能量
let mut smoothed_flatness = 0.3f32;    // 中等平坦度
let mut smoothed_centroid = 0.4f32;    // 中等重心
let mut env_class = EnvClass::Office;  // 从 Office 开始，而不是 Quiet
```

### 修复 5: 提高更新频率的权重

```rust
// 原来
smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, 0.1);

// 修改为更快的响应
smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, 0.25);
```

---

## 完整修复代码

```rust
// capture.rs 中的环境自适应部分重写

if env_auto_enabled && !bypass_enabled {
    // 始终计算环境特征（而不是只在静默时）
    if let Some(buf) = inframe.as_slice() {
        let rms = df::rms(buf.iter());
        let rms_db = 20.0 * rms.max(1e-9).log10();
        
        // 获取频谱特征
        let feats = compute_noise_features(df.get_spec_noisy());
        
        // 自适应平滑系数：静默时快，有语音时慢
        let alpha = if rms_db < -40.0 {
            0.4   // 非常安静，快速适应
        } else if rms_db < -30.0 {
            0.2   // 轻微背景音
        } else if rms_db < -20.0 {
            0.1   // 有语音
        } else {
            0.03  // 大声说话，非常慢地更新
        };
        
        smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, alpha);
        smoothed_flatness = smooth_value(smoothed_flatness, feats.spectral_flatness, alpha);
        smoothed_centroid = smooth_value(smoothed_centroid, feats.spectral_centroid, alpha);
        
        // 调试日志（可选，正式版本可以用 debug 级别）
        if frame_counter % 100 == 0 {  // 每 100 帧输出一次
            log::debug!(
                "环境特征: energy={:.1}dB, flatness={:.2}, centroid={:.2}, rms={:.1}dB",
                smoothed_energy, smoothed_flatness, smoothed_centroid, rms_db
            );
        }
    }
    
    // 柔和模式检测（修正后的条件）
    let soft_candidate = smoothed_energy < -55.0 
                      && smoothed_flatness < 0.2 
                      && smoothed_centroid < 0.35;
    
    if soft_candidate {
        soft_mode_hold = soft_mode_hold.saturating_add(1);
    } else {
        soft_mode_hold = soft_mode_hold.saturating_sub(2);  // 退出更快
    }
    
    if soft_mode_hold > SOFT_MODE_HOLD_FRAMES {
        soft_mode = true;
    } else if soft_mode_hold < SOFT_MODE_HOLD_FRAMES / 4 {  // 滞后退出
        soft_mode = false;
    }
    
    if soft_mode != last_soft_mode {
        last_soft_mode = soft_mode;
        if soft_mode {
            log::info!("环境自适应: 切换到柔和模式");
        } else {
            log::info!("环境自适应: 切换到正常模式");
        }
        if let Some(ref sender) = s_env_status {
            let status = if soft_mode { EnvStatus::Soft } else { EnvStatus::Normal };
            let _ = sender.try_send(status);
        }
    }

    // 环境分类
    let target_env = classify_env(smoothed_energy, smoothed_flatness, smoothed_centroid);
    if target_env != env_class {
        log::info!(
            "环境自适应: {} → {} (energy={:.1}dB, flat={:.2}, cent={:.2})",
            format!("{:?}", env_class),
            format!("{:?}", target_env),
            smoothed_energy, smoothed_flatness, smoothed_centroid
        );
        env_class = target_env;
    }

    // 参数调整逻辑保持不变...
}
```

---

## 验证修复效果的方法

修复后，你应该能在日志中看到类似这样的输出：

```
2024-01-01 12:00:00 | INFO | 环境自适应: Quiet → Office (energy=-42.5dB, flat=0.52, cent=0.48)
2024-01-01 12:00:05 | INFO | 环境自适应: Office → Noisy (energy=-35.2dB, flat=0.61, cent=0.55)
2024-01-01 12:00:30 | INFO | 环境自适应: Noisy → Office (energy=-45.1dB, flat=0.48, cent=0.41)
```

如果修复正确，你应该能：
1. 看到环境分类在不同场景下切换
2. 看到降噪参数随之变化
3. 听到降噪强度的明显差异

---

## 总结

| Bug | 严重程度 | 影响 |
|-----|---------|------|
| 只在静默时更新 | 🔴 致命 | 功能几乎完全失效 |
| soft_mode 条件错误 | 🔴 致命 | 柔和模式永远不会触发 |
| 默认关闭 | 🟡 中等 | 用户可能不知道开启 |
| 平滑系数太小 | 🟡 中等 | 响应太慢 |
| 初始值问题 | 🟡 中等 | 冷启动时判断错误 |
| 缺少日志 | 🟢 轻微 | 调试困难 |

核心问题就是 **Bug 2**：只在 RMS < -35dB 时才更新环境特征，但正常使用时几乎不可能达到这个条件。