# 代码优化实现

## 一、多频段激励器实现

创建新文件 `src/audio/multiband_exciter.rs`：

```rust
use super::eq::biquad::{Biquad, BiquadType};

/// 单频段激励器
struct BandExciter {
    hp: Biquad,           // 高通滤波器
    lp: Biquad,           // 低通滤波器  
    drive: f32,           // 驱动量
    mix: f32,             // 干湿比
    prev_hp: f32,
    prev_lp: f32,
}

impl BandExciter {
    fn new(sample_rate: f32, low_freq: f32, high_freq: f32, drive: f32, mix: f32) -> Self {
        Self {
            hp: Biquad::new(BiquadType::HighPass, sample_rate, low_freq, 0.707, 0.0),
            lp: Biquad::new(BiquadType::LowPass, sample_rate, high_freq, 0.707, 0.0),
            drive: drive.clamp(1.0, 4.0),
            mix: mix.clamp(0.0, 1.0),
            prev_hp: 0.0,
            prev_lp: 0.0,
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        // 带通滤波提取目标频段
        let hp_out = self.hp.process(sample);
        let band = self.lp.process(hp_out);
        
        // 谐波生成：使用 tanh 软饱和
        let driven = band * self.drive;
        let saturated = driven.tanh();
        
        // 归一化补偿
        let normalized = saturated / self.drive.tanh();
        
        // 返回激励后的信号（仅返回新增的谐波成分）
        (normalized - band) * self.mix
    }
}

/// 多频段谐波激励器
pub struct MultibandExciter {
    sample_rate: f32,
    
    // 存在感频段 (2.5-5 kHz)
    presence: BandExciter,
    presence_enabled: bool,
    
    // 空气感频段 (5-10 kHz)
    air: BandExciter,
    air_enabled: bool,
    
    // 亮度频段 (10+ kHz)
    brilliance: BandExciter,
    brilliance_enabled: bool,
    
    // 总混合量
    master_mix: f32,
}

impl MultibandExciter {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            presence: BandExciter::new(sample_rate, 2500.0, 5000.0, 1.4, 0.18),
            presence_enabled: true,
            air: BandExciter::new(sample_rate, 5000.0, 10000.0, 1.6, 0.22),
            air_enabled: true,
            brilliance: BandExciter::new(sample_rate, 10000.0, 18000.0, 1.8, 0.12),
            brilliance_enabled: true,
            master_mix: 1.0,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if self.master_mix <= 0.0 {
            return;
        }
        
        for sample in samples.iter_mut() {
            let mut excitation = 0.0;
            
            if self.presence_enabled {
                excitation += self.presence.process(*sample);
            }
            if self.air_enabled {
                excitation += self.air.process(*sample);
            }
            if self.brilliance_enabled {
                excitation += self.brilliance.process(*sample);
            }
            
            *sample += excitation * self.master_mix;
        }
    }

    pub fn set_presence(&mut self, drive: f32, mix: f32) {
        self.presence.drive = drive.clamp(1.0, 3.0);
        self.presence.mix = mix.clamp(0.0, 0.5);
    }

    pub fn set_air(&mut self, drive: f32, mix: f32) {
        self.air.drive = drive.clamp(1.0, 3.0);
        self.air.mix = mix.clamp(0.0, 0.5);
    }

    pub fn set_brilliance(&mut self, drive: f32, mix: f32) {
        self.brilliance.drive = drive.clamp(1.0, 4.0);
        self.brilliance.mix = mix.clamp(0.0, 0.3);
    }

    pub fn set_master_mix(&mut self, mix: f32) {
        self.master_mix = mix.clamp(0.0, 1.5);
    }

    /// 根据降噪强度自动调整激励参数
    pub fn adapt_to_denoising(&mut self, attenuation_db: f32) {
        // 降噪越强，激励越强
        let factor = (attenuation_db / 40.0).clamp(0.5, 1.5);
        
        self.presence.mix = (0.15 * factor).clamp(0.0, 0.4);
        self.air.mix = (0.18 * factor).clamp(0.0, 0.45);
        self.brilliance.mix = (0.10 * factor).clamp(0.0, 0.25);
    }
}
```

---

## 二、频谱倾斜补偿器

创建新文件 `src/audio/spectral_compensator.rs`：

```rust
use super::eq::biquad::{Biquad, BiquadType};
use std::collections::VecDeque;

/// 频谱倾斜补偿器
/// 检测降噪前后的频谱变化，自动补偿高频损失
pub struct SpectralCompensator {
    sample_rate: f32,
    
    // 频谱分析
    analysis_hp: Biquad,  // 高频能量检测
    analysis_lp: Biquad,  // 低频能量检测
    
    // 补偿滤波器
    compensator: Biquad,  // 高频搁架
    
    // 平滑状态
    pre_ratio_smooth: f32,
    post_ratio_smooth: f32,
    target_boost: f32,
    current_boost: f32,
    
    // 配置
    enabled: bool,
    max_boost_db: f32,
    smoothing: f32,
}

impl SpectralCompensator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            analysis_hp: Biquad::new(BiquadType::HighPass, sample_rate, 4000.0, 0.707, 0.0),
            analysis_lp: Biquad::new(BiquadType::LowPass, sample_rate, 1000.0, 0.707, 0.0),
            compensator: Biquad::new(BiquadType::HighShelf, sample_rate, 3000.0, 0.707, 0.0),
            pre_ratio_smooth: 0.0,
            post_ratio_smooth: 0.0,
            target_boost: 0.0,
            current_boost: 0.0,
            enabled: true,
            max_boost_db: 6.0,
            smoothing: 0.95,
        }
    }

    /// 分析原始信号的高低频比例
    pub fn analyze_pre(&mut self, samples: &[f32]) {
        if !self.enabled || samples.is_empty() {
            return;
        }
        
        let mut hp_energy = 0.0f32;
        let mut lp_energy = 0.0f32;
        
        // 复制滤波器状态用于分析（不影响主滤波器）
        let mut hp = self.analysis_hp.clone();
        let mut lp = self.analysis_lp.clone();
        
        for &sample in samples {
            let h = hp.process(sample);
            let l = lp.process(sample);
            hp_energy += h * h;
            lp_energy += l * l;
        }
        
        let ratio = if lp_energy > 1e-10 {
            (hp_energy / lp_energy).sqrt()
        } else {
            0.0
        };
        
        self.pre_ratio_smooth = self.smoothing * self.pre_ratio_smooth 
                               + (1.0 - self.smoothing) * ratio;
    }

    /// 分析降噪后信号并应用补偿
    pub fn analyze_and_compensate(&mut self, samples: &mut [f32]) {
        if !self.enabled || samples.is_empty() {
            return;
        }
        
        // 分析降噪后的高低频比例
        let mut hp_energy = 0.0f32;
        let mut lp_energy = 0.0f32;
        
        let mut hp = self.analysis_hp.clone();
        let mut lp = self.analysis_lp.clone();
        
        for &sample in samples.iter() {
            let h = hp.process(sample);
            let l = lp.process(sample);
            hp_energy += h * h;
            lp_energy += l * l;
        }
        
        let post_ratio = if lp_energy > 1e-10 {
            (hp_energy / lp_energy).sqrt()
        } else {
            0.0
        };
        
        self.post_ratio_smooth = self.smoothing * self.post_ratio_smooth 
                                + (1.0 - self.smoothing) * post_ratio;
        
        // 计算需要的补偿量
        if self.pre_ratio_smooth > 1e-6 && self.post_ratio_smooth > 1e-6 {
            let ratio_drop = self.pre_ratio_smooth / self.post_ratio_smooth;
            
            // 只有当高频比例下降时才补偿
            if ratio_drop > 1.1 {
                // 转换为 dB
                self.target_boost = (20.0 * ratio_drop.log10())
                    .clamp(0.0, self.max_boost_db);
            } else {
                self.target_boost = 0.0;
            }
        }
        
        // 平滑过渡
        self.current_boost = self.smoothing * self.current_boost 
                           + (1.0 - self.smoothing) * self.target_boost;
        
        // 应用补偿
        if self.current_boost > 0.1 {
            self.compensator.set_gain_db(self.current_boost);
            for sample in samples.iter_mut() {
                *sample = self.compensator.process(*sample);
            }
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_max_boost(&mut self, db: f32) {
        self.max_boost_db = db.clamp(0.0, 12.0);
    }

    pub fn current_boost_db(&self) -> f32 {
        self.current_boost
    }
}
```

---

## 三、改进的饱和器

修改 `src/audio/saturation.rs`：

```rust
use super::eq::biquad::{Biquad, BiquadType};
use log::warn;

/// 中高频饱和器
/// 只对中高频进行饱和处理，保持低频干净
pub struct MidHighSaturation {
    sample_rate: f32,
    
    // 分频滤波器 (~600Hz)
    crossover_lp: Biquad,
    crossover_hp: Biquad,
    
    // 参数
    drive: f32,
    makeup_db: f32,
    mix: f32,
    crossover_freq: f32,
}

impl MidHighSaturation {
    pub fn new(sample_rate: f32) -> Self {
        let crossover_freq = 600.0;
        Self {
            sample_rate,
            crossover_lp: Biquad::new(BiquadType::LowPass, sample_rate, crossover_freq, 0.707, 0.0),
            crossover_hp: Biquad::new(BiquadType::HighPass, sample_rate, crossover_freq, 0.707, 0.0),
            drive: 1.3,
            makeup_db: 0.0,
            mix: 0.8,
            crossover_freq,
        }
    }

    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.clamp(1.0, 3.0);
    }

    pub fn set_makeup(&mut self, makeup_db: f32) {
        self.makeup_db = makeup_db.clamp(-6.0, 6.0);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn set_crossover(&mut self, freq: f32) {
        self.crossover_freq = freq.clamp(200.0, 2000.0);
        self.crossover_lp = Biquad::new(
            BiquadType::LowPass, 
            self.sample_rate, 
            self.crossover_freq, 
            0.707, 
            0.0
        );
        self.crossover_hp = Biquad::new(
            BiquadType::HighPass, 
            self.sample_rate, 
            self.crossover_freq, 
            0.707, 
            0.0
        );
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() || self.mix <= 0.0 {
            return;
        }
        
        if sanitize_samples("MidHighSaturation", samples) {
            return;
        }
        
        let makeup = db_to_linear(self.makeup_db);
        let drive = self.drive;
        let wet = self.mix;
        let dry = 1.0 - wet;
        
        for sample in samples.iter_mut() {
            // 分频
            let low = self.crossover_lp.process(*sample);
            let high = self.crossover_hp.process(*sample);
            
            // 只对高频部分进行饱和
            let driven = high * drive;
            let saturated = driven.tanh() / drive;  // 归一化补偿
            
            // 混合：低频保持不变，高频混合饱和信号
            let processed_high = high * dry + saturated * wet;
            
            // 合并并应用 makeup
            *sample = (low + processed_high) * makeup;
        }
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn sanitize_samples(tag: &str, samples: &mut [f32]) -> bool {
    let mut found = false;
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            found = true;
        }
    }
    if found {
        warn!("{tag} 检测到非法音频数据 (NaN/Inf)，跳过本帧处理");
    }
    found
}
```

---

## 四、新增音色还原 EQ 预设

在 `src/audio/eq/presets.rs` 中添加：

```rust
/// 音色还原预设
/// 专门针对降噪后的高频损失进行补偿
const RESTORATION: EqPreset = EqPreset {
    name: "音色还原",
    default_mix: 0.7,
    bands: [
        // Band 1: 低频收紧
        BandSettings {
            label: "低频控制",
            frequency_hz: 100.0,
            q: 0.7,
            detector_q: 0.6,
            threshold_db: -34.0,
            ratio: 3.5,
            max_gain_db: 10.0,
            attack_ms: 30.0,
            release_ms: 250.0,
            mode: BandMode::Downward,
            filter: FilterKind::LowShelf,
            makeup_db: 0.0,
            static_gain_db: -1.5,
        },
        // Band 2: 温暖度保持
        BandSettings {
            label: "温暖度",
            frequency_hz: 200.0,
            q: 1.0,
            detector_q: 1.0,
            threshold_db: -42.0,
            ratio: 1.5,
            max_gain_db: 5.0,
            attack_ms: 25.0,
            release_ms: 200.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 1.5,
        },
        // Band 3: 存在感补偿（关键频段）
        BandSettings {
            label: "存在感",
            frequency_hz: 2800.0,
            q: 0.9,
            detector_q: 1.0,
            threshold_db: -36.0,
            ratio: 2.2,
            max_gain_db: 7.0,
            attack_ms: 15.0,
            release_ms: 100.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 3.0,  // 较强提升
        },
        // Band 4: 清晰度恢复
        BandSettings {
            label: "清晰度",
            frequency_hz: 5500.0,
            q: 1.1,
            detector_q: 1.2,
            threshold_db: -32.0,
            ratio: 2.0,
            max_gain_db: 6.0,
            attack_ms: 12.0,
            release_ms: 80.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 2.5,
        },
        // Band 5: 空气感恢复
        BandSettings {
            label: "空气感",
            frequency_hz: 10000.0,
            q: 0.65,
            detector_q: 0.7,
            threshold_db: -40.0,
            ratio: 1.8,
            max_gain_db: 9.0,
            attack_ms: 35.0,
            release_ms: 280.0,
            mode: BandMode::Upward,
            filter: FilterKind::HighShelf,
            makeup_db: 0.0,
            static_gain_db: 4.0,  // 明显提升
        },
    ],
};

/// 自然还原预设
/// 更保守的补偿，适合高质量录音
const NATURAL_RESTORATION: EqPreset = EqPreset {
    name: "自然还原",
    default_mix: 0.55,
    bands: [
        BandSettings {
            label: "低频平衡",
            frequency_hz: 80.0,
            q: 0.8,
            detector_q: 0.7,
            threshold_db: -38.0,
            ratio: 2.5,
            max_gain_db: 8.0,
            attack_ms: 35.0,
            release_ms: 280.0,
            mode: BandMode::Downward,
            filter: FilterKind::LowShelf,
            makeup_db: 0.0,
            static_gain_db: 0.0,
        },
        BandSettings {
            label: "中低频",
            frequency_hz: 300.0,
            q: 1.2,
            detector_q: 1.0,
            threshold_db: -30.0,
            ratio: 2.0,
            max_gain_db: 5.0,
            attack_ms: 28.0,
            release_ms: 180.0,
            mode: BandMode::Downward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: -0.5,
        },
        BandSettings {
            label: "存在感",
            frequency_hz: 3200.0,
            q: 1.0,
            detector_q: 1.0,
            threshold_db: -38.0,
            ratio: 1.8,
            max_gain_db: 5.0,
            attack_ms: 18.0,
            release_ms: 100.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 2.0,
        },
        BandSettings {
            label: "齿音控制",
            frequency_hz: 6500.0,
            q: 2.0,
            detector_q: 1.8,
            threshold_db: -26.0,
            ratio: 4.5,
            max_gain_db: 10.0,
            attack_ms: 8.0,
            release_ms: 140.0,
            mode: BandMode::Downward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 0.0,
        },
        BandSettings {
            label: "空气感",
            frequency_hz: 12000.0,
            q: 0.7,
            detector_q: 0.7,
            threshold_db: -42.0,
            ratio: 1.5,
            max_gain_db: 6.0,
            attack_ms: 40.0,
            release_ms: 300.0,
            mode: BandMode::Upward,
            filter: FilterKind::HighShelf,
            makeup_db: 0.0,
            static_gain_db: 2.5,
        },
    ],
};
```

---

## 五、处理链集成

在 `capture.rs` 中修改处理流程：

```rust
// 初始化新组件
let mut multiband_exciter = MultibandExciter::new(df.sr as f32);
let mut spectral_compensator = SpectralCompensator::new(df.sr as f32);
let mut mid_high_saturation = MidHighSaturation::new(df.sr as f32);

// ... 在处理循环中 ...

// 1. 高通滤波（已有）
if highpass_enabled {
    if let Some(buffer) = inframe.as_slice_mut() {
        highpass.process(buffer);
    }
}

// 2. 分析原始信号的频谱特征（新增）
if let Some(buffer) = inframe.as_slice() {
    spectral_compensator.analyze_pre(buffer);
}

// 3. DeepFilterNet 降噪（已有）
let lsnr = df.process(inframe.view(), outframe.view_mut())
    .expect("Failed to run DeepFilterNet");

// 4. 频谱补偿（新增）
if let Some(buffer) = outframe.as_slice_mut() {
    spectral_compensator.analyze_and_compensate(buffer);
}

// 5. 根据降噪强度调整激励器（新增）
let current_atten = df.atten_lim.unwrap_or(30.0);
multiband_exciter.adapt_to_denoising(current_atten);

// 6. 瞬态整形（已有）
if transient_enabled {
    if let Some(buffer) = outframe.as_slice_mut() {
        transient_shaper.process(buffer);
    }
}

// 7. 动态EQ（已有，但使用新预设）
let metrics = {
    let buffer = outframe.as_slice_mut().expect("...");
    dynamic_eq.process_block(buffer)
};

// 8. 多频段激励（替换原来的单频段激励）
if exciter_enabled {
    if let Some(buffer) = outframe.as_slice_mut() {
        multiband_exciter.process(buffer);
    }
}

// 9. 中高频饱和（替换原来的全频段饱和）
if saturation_enabled {
    if let Some(buffer) = outframe.as_slice_mut() {
        mid_high_saturation.process(buffer);
    }
}

// 10. AGC + Limiter（已有）
// ...
```

---

## 六、配置参数建议

```rust
// 针对不同场景的推荐配置

/// 会议室场景（混响较多）
pub fn config_meeting_room() -> ProcessingConfig {
    ProcessingConfig {
        // 降噪较强
        df_attenuation: 40.0,
        df_mix: 0.95,
        
        // 激励器：补偿高频损失
        presence_mix: 0.20,
        air_mix: 0.25,
        brilliance_mix: 0.12,
        
        // EQ：使用会议室预设
        eq_preset: EqPresetKind::ConferenceHall,
        eq_mix: 0.80,
        
        // 其他
        highpass_cutoff: 80.0,
        agc_enabled: true,
    }
}

/// 安静办公室场景
pub fn config_quiet_office() -> ProcessingConfig {
    ProcessingConfig {
        df_attenuation: 28.0,
        df_mix: 0.85,
        
        presence_mix: 0.15,
        air_mix: 0.18,
        brilliance_mix: 0.08,
        
        eq_preset: EqPresetKind::Restoration,
        eq_mix: 0.65,
        
        highpass_cutoff: 60.0,
        agc_enabled: true,
    }
}

/// 嘈杂环境场景
pub fn config_noisy() -> ProcessingConfig {
    ProcessingConfig {
        df_attenuation: 55.0,
        df_mix: 1.0,
        
        // 强降噪需要更强的激励补偿
        presence_mix: 0.25,
        air_mix: 0.30,
        brilliance_mix: 0.15,
        
        eq_preset: EqPresetKind::Restoration,
        eq_mix: 0.85,
        
        highpass_cutoff: 100.0,
        agc_enabled: true,
    }
}
```

---

## 七、测试建议

1. **A/B 测试**：保存降噪前后的音频，对比频谱
2. **关键频段检查**：
   - 2-4 kHz 的存在感
   - 6-8 kHz 的齿音
   - 10+ kHz 的空气感
3. **避免过度处理**：激励过强会产生"金属感"
4. **不同说话人测试**：男声、女声、不同口音