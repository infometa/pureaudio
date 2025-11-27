# 语音增强项目分析报告

## 一、项目架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        音频处理流水线                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  麦克风输入                                                      │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────┐                                            │
│  │ HighpassFilter  │  高通滤波 (60Hz)                           │
│  │   前处理        │  去除低频隆隆声                             │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  DeepFilterNet  │  深度学习降噪                              │
│  │   核心降噪      │  去除环境噪声                               │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │TransientShaper  │  瞬态整形                                  │
│  │   动态塑形      │  增强起音清晰度                             │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   Saturation    │  温暖饱和                                  │
│  │   谐波染色      │  增加谐波丰富度                             │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │HarmonicExciter  │  高频激励                                  │
│  │   空气感补偿    │  恢复高频细节                               │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   DynamicEq     │  动态均衡                                  │
│  │   频段塑形      │  5段动态EQ + 压缩/扩展                      │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │      AGC        │  自动增益控制                              │
│  │   音量归一化    │  稳定输出电平                               │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   BusLimiter    │  总线限制器                                │
│  │   峰值保护      │  防止过载                                   │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│      输出信号                                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、各模块分析

### 2.1 DeepFilterNet 降噪

**当前实现**：使用 DeepFilterNet 模型进行降噪，有环境自适应逻辑。

**优点**：
- 深度学习降噪效果好
- 有环境分类（Quiet/Office/Noisy）
- 有柔和模式检测

**问题**：
1. 降噪后高频损失严重（这是你想解决的核心问题）
2. 环境自适应参数切换可能不够平滑
3. `df_mix` 参数变化可能引入相位问题

---

### 2.2 HarmonicExciter 激励器

**当前实现**：
```rust
// 7500Hz 高通 + 轻度饱和
cutoff_hz: 7500.0
drive: 1.6
mix: 0.25
```

**问题**：
1. **截止频率过高**：7500Hz 只处理极高频，错过了 3-7kHz 的"存在感"区域
2. **一阶高通斜率太缓**：只有 -6dB/oct，分离度不够
3. **没有多频段处理**：语音的不同频段需要不同程度的激励

---

### 2.3 DynamicEq 动态均衡

**当前实现**：5 段动态 EQ，支持上行扩展和下行压缩。

**优点**：
- 预设丰富（7种场景）
- 动态处理可以自适应

**问题**：
1. **齿音控制频率不够精准**：6800-7200Hz，实际齿音峰值常在 5-6kHz
2. **空气感频段偏高**：8000-14000Hz，可能已经被降噪损失了
3. **缺少 2-4kHz 的精细控制**：这是语音清晰度最关键的区域

---

### 2.4 Saturation 饱和器

**当前实现**：
```rust
drive: 1.2
makeup_db: -0.5
```

**问题**：
1. **全频段饱和**：低频饱和会让声音变浑浊
2. **没有频段分离**：应该只对中高频做饱和

---

### 2.5 TransientShaper 瞬态整形

**当前实现**：检测瞬态，增强起音。

**问题**：
1. **阈值固定**：`threshold_db: -30.0` 可能不适合所有场景
2. **hold 时间较短**：8ms 可能无法完整捕捉辅音

---

### 2.6 AGC 自动增益

**当前实现**：RMS 检测 + 软限幅。

**问题**：
1. **RMS 窗口 0.6 秒**：对于语音来说偏长，反应较慢
2. **没有 VAD 配合**：静默时可能放大噪音

---

## 三、核心问题：音色还原

降噪后音色损失的主要原因：

```
DeepFilterNet 降噪的副作用：

频率
 ↑
 │                    ░░░░░░░░░░░░░░
 │                  ░░░░░░░░░░░░░░░░
 │ ████████████████░░░░░░░░░░░░░░░░░   ← 高频被"连带"削弱
 │ ████████████████████░░░░░░░░░░░░
 │ ████████████████████████░░░░░░░░
 └──────────────────────────────────→ 频率
        语音主体        高频细节
                       (存在感/空气感)

████ = 保留良好
░░░░ = 损失区域
```

### 损失的关键频段

| 频段 | 作用 | 损失后的听感 |
|------|------|-------------|
| 2-4 kHz | 语音存在感、辅音清晰度 | 声音发闷、距离感远 |
| 4-6 kHz | 齿音、唇音 | s/f/sh 不清楚 |
| 6-10 kHz | 空气感、明亮度 | 声音暗淡、缺乏活力 |
| 10+ kHz | 泛音、临场感 | 缺乏"高保真"感 |

---

## 四、优化建议

### 4.1 改进 HarmonicExciter

```rust
// 建议：多频段激励器
pub struct MultibandExciter {
    // 存在感激励 (2.5-5 kHz)
    presence_exciter: BandExciter,
    // 空气感激励 (6-12 kHz)  
    air_exciter: BandExciter,
    // 超高频激励 (12+ kHz)
    brilliance_exciter: BandExciter,
}

struct BandExciter {
    highpass: Biquad,      // 二阶高通，-12dB/oct
    lowpass: Biquad,       // 二阶低通，限制频段
    drive: f32,
    mix: f32,
}
```

**参数建议**：
```
存在感段: 2500-5000Hz, drive=1.3, mix=0.15
空气感段: 6000-12000Hz, drive=1.5, mix=0.20
超高频段: 12000+Hz, drive=1.8, mix=0.10
```

---

### 4.2 添加频谱倾斜补偿

降噪后常见的问题是高频整体衰减，可以用简单的高频搁架提升来补偿：

```rust
pub struct SpectralTiltCompensator {
    // 检测降噪前后的频谱倾斜差异
    pre_tilt: f32,
    post_tilt: f32,
    // 补偿滤波器
    compensator: Biquad,  // HighShelf
}

impl SpectralTiltCompensator {
    pub fn process(&mut self, noisy: &[f32], denoised: &mut [f32]) {
        // 1. 计算原始信号的谱重心
        let pre_centroid = compute_spectral_centroid(noisy);
        // 2. 计算降噪后的谱重心
        let post_centroid = compute_spectral_centroid(denoised);
        // 3. 如果谱重心下降太多，提升高频
        let centroid_drop = pre_centroid - post_centroid;
        if centroid_drop > threshold {
            // 动态调整高频搁架增益
            let boost_db = (centroid_drop * scale).clamp(0.0, 6.0);
            self.compensator.set_gain_db(boost_db);
        }
        // 4. 应用补偿
        for sample in denoised {
            *sample = self.compensator.process(*sample);
        }
    }
}
```

---

### 4.3 优化 DynamicEQ 预设

**针对"音色还原"的专用预设**：

```rust
const RESTORATION: EqPreset = EqPreset {
    name: "音色还原",
    default_mix: 0.65,
    bands: [
        // Band 1: 低频收紧，避免浑浊
        BandSettings {
            label: "低频控制",
            frequency_hz: 120.0,
            q: 0.8,
            threshold_db: -32.0,
            ratio: 3.0,
            mode: BandMode::Downward,
            filter: FilterKind::LowShelf,
            static_gain_db: -1.0,
            ..
        },
        // Band 2: 中低频保持自然
        BandSettings {
            label: "温暖度",
            frequency_hz: 250.0,
            q: 1.0,
            threshold_db: -40.0,
            ratio: 1.5,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            static_gain_db: 1.0,
            ..
        },
        // Band 3: 存在感补偿（关键！）
        BandSettings {
            label: "存在感",
            frequency_hz: 3000.0,
            q: 0.9,
            threshold_db: -35.0,
            ratio: 2.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            static_gain_db: 2.5,  // 较强提升
            max_gain_db: 6.0,
            ..
        },
        // Band 4: 清晰度补偿
        BandSettings {
            label: "清晰度",
            frequency_hz: 5000.0,
            q: 1.2,
            threshold_db: -30.0,
            ratio: 2.5,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            static_gain_db: 2.0,
            ..
        },
        // Band 5: 空气感恢复
        BandSettings {
            label: "空气感",
            frequency_hz: 10000.0,
            q: 0.7,
            threshold_db: -38.0,
            ratio: 1.8,
            mode: BandMode::Upward,
            filter: FilterKind::HighShelf,
            static_gain_db: 3.5,  // 明显提升
            max_gain_db: 8.0,
            ..
        },
    ],
};
```

---

### 4.4 改进 Saturation

**建议：只对中高频做饱和**

```rust
pub struct MidHighSaturation {
    // 分频点
    crossover: Biquad,  // ~800Hz 分频
    crossover_lp: Biquad,
    crossover_hp: Biquad,
    // 只对高频部分饱和
    drive: f32,
    mix: f32,
}

impl MidHighSaturation {
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples {
            // 分频
            let low = self.crossover_lp.process(*sample);
            let high = self.crossover_hp.process(*sample);
            // 只饱和高频部分
            let saturated_high = (high * self.drive).tanh() / self.drive;
            // 混合
            *sample = low + (high * (1.0 - self.mix) + saturated_high * self.mix);
        }
    }
}
```

---

### 4.5 添加自适应高频恢复

根据降噪强度自动调整高频补偿：

```rust
// 在处理循环中
let df_attenuation = df.atten_lim.unwrap_or(30.0);

// 降噪越强，高频补偿越多
let hf_compensation = match df_attenuation {
    0.0..=20.0 => 0.0,      // 轻度降噪，不补偿
    20.0..=35.0 => 1.5,     // 中度降噪，轻度补偿
    35.0..=50.0 => 3.0,     // 重度降噪，中度补偿
    _ => 4.5,               // 极重降噪，强补偿
};

// 应用到 exciter 和 EQ
exciter.set_mix(base_mix + hf_compensation * 0.05);
dynamic_eq.set_band_gain(4, base_air_gain + hf_compensation);
```

---

### 4.6 处理链顺序优化

当前顺序：
```
Highpass → DF降噪 → Transient → Saturation → Exciter → EQ → AGC → Limiter
```

建议优化为：
```
Highpass → DF降噪 → 频谱补偿 → Transient → EQ → Exciter → Saturation → AGC → Limiter
                      ↑
                  新增模块
```

**调整理由**：
1. 频谱补偿紧跟降噪，先恢复整体平衡
2. EQ 在 Exciter 之前，先塑形再激励
3. Saturation 靠后，避免影响动态 EQ 的检测

---

## 五、适配性优化

### 5.1 输入信号自适应

```rust
struct InputAnalyzer {
    // 实时分析输入特征
    spectral_centroid: f32,
    spectral_spread: f32,
    harmonic_ratio: f32,  // 谐波/噪声比
    
    // 根据特征选择参数
    pub fn suggest_params(&self) -> ProcessingParams {
        if self.harmonic_ratio > 0.8 {
            // 高质量麦克风，轻处理
            ProcessingParams::light()
        } else if self.spectral_centroid < 2000.0 {
            // 声音偏暗，需要更多高频补偿
            ProcessingParams::bright_boost()
        } else {
            ProcessingParams::default()
        }
    }
}
```

### 5.2 设备特性适配

```rust
// 检测采样率，调整滤波器参数
fn adapt_to_sample_rate(sr: f32) -> FilterParams {
    match sr as u32 {
        ..=16000 => {
            // 窄带语音，集中处理 300-3400Hz
            FilterParams::narrowband()
        }
        16001..=32000 => {
            // 宽带语音
            FilterParams::wideband()
        }
        _ => {
            // 超宽带/全频带
            FilterParams::fullband()
        }
    }
}
```

### 5.3 场景自动检测增强

```rust
// 增强环境分类逻辑
fn classify_env_v2(features: &NoiseFeatures) -> EnvClass {
    // 使用更多特征
    let energy = features.energy_db;
    let flatness = features.spectral_flatness;
    let centroid = features.spectral_centroid;
    let temporal_var = features.temporal_variance;  // 新增：时间变化性
    
    // 检测特殊场景
    if flatness > 0.8 && temporal_var < 0.1 {
        // 稳态噪声（空调、风扇）
        return EnvClass::StationaryNoise;
    }
    
    if centroid > 3000.0 && temporal_var > 0.5 {
        // 非稳态噪声（键盘、说话）
        return EnvClass::TransientNoise;
    }
    
    // 原有分类逻辑...
}
```

---

## 六、实施优先级

| 优先级 | 改进项 | 预期效果 | 工作量 |
|--------|--------|----------|--------|
| 🔴 高 | 多频段激励器 | 大幅改善高频细节 | 中 |
| 🔴 高 | 存在感 EQ 预设 | 改善"距离感" | 低 |
| 🟡 中 | 频谱倾斜补偿 | 自动平衡频谱 | 中 |
| 🟡 中 | 中高频饱和 | 更自然的温暖感 | 低 |
| 🟢 低 | 自适应补偿 | 更智能的适配 | 高 |
| 🟢 低 | 处理链重排 | 更合理的信号流 | 低 |

---

## 七、代码修改建议

详细的代码修改请参见 `OPTIMIZATION_CODE.md` 文件。