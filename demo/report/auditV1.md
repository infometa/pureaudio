# 多场景自适应语音增强方案

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v2.0 |
| 目标 | 自动适配开放办公区与会议室场景，无需手动切换 |
| 状态 | 待开发评估 |

---

## 一、场景分析

### 1.1 两种场景特征对比

| 特征 | 开放办公区 | 会议室 |
|------|-----------|--------|
| 空间大小 | 大/开阔 | 小/封闭 |
| 混响时间 (RT60) | 短 (0.3-0.5s) | 中-长 (0.4-0.8s) |
| 背景噪声类型 | 多人交谈、键盘、空调 | 空调、投影仪风扇 |
| 背景噪声电平 | 高 (-35 ~ -25 dB) | 低-中 (-50 ~ -35 dB) |
| 干扰人声距离 | 近 (1-3米) | 远或无 |
| 干扰人声特点 | 持续、多源 | 偶发、单源 |
| 直达声/反射声比 | 高 | 低 |
| 频谱特征 | 宽带、平坦 | 低频突出（驻波） |

### 1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| 自动识别 | 无需用户干预，程序自动判断当前场景 |
| 平滑切换 | 场景变化时参数渐变，避免音质突变 |
| 快速适应 | 进入新场景后 3-5 秒内完成适配 |
| 误判容错 | 短暂异常不触发场景切换 |

---

## 二、整体架构

### 2.1 自适应系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         自适应语音增强系统                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    场景感知层 (SceneAnalyzer)                        │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐          │   │
│  │  │ 能量分析   │ │ 混响估计   │ │ 人声检测   │ │ 频谱分析   │          │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘          │   │
│  │        └─────────────┴─────────────┴─────────────┘                 │   │
│  │                              ↓                                      │   │
│  │                    ┌─────────────────┐                             │   │
│  │                    │ 场景分类器       │ → 场景类型 + 置信度          │   │
│  │                    └─────────────────┘                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    参数调度层 (ParameterScheduler)                   │   │
│  │                                                                     │   │
│  │  场景 A 参数集 ←──→ 平滑插值器 ←──→ 场景 B 参数集                    │   │
│  │                         ↓                                           │   │
│  │                   当前生效参数                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    处理执行层 (ProcessingChain)                      │   │
│  │                                                                     │   │
│  │  NoiseGate → SpatialFilter → Highpass → DeepFilter →               │   │
│  │  VoiceIsolator → SpectralGate → EQ → AGC → Limiter                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 场景状态机

```
                    ┌─────────────────┐
                    │   Unknown       │ ← 启动/重置
                    │   (校准中)      │
                    └────────┬────────┘
                             │ 校准完成 (3-5秒)
                             ↓
           ┌─────────────────┴─────────────────┐
           ↓                                   ↓
    ┌──────────────┐                   ┌──────────────┐
    │ OpenOffice   │ ←───────────────→ │ MeetingRoom  │
    │ (开放办公区)  │   渐变切换 (2秒)   │ (会议室)     │
    └──────────────┘                   └──────────────┘
           ↑                                   ↑
           │         ┌──────────────┐         │
           └────────→│   Transition │←────────┘
                     │   (过渡中)    │
                     └──────────────┘
```

---

## 三、场景感知模块

### 3.1 SceneAnalyzer（场景分析器）

```rust
// ====================
// 文件: audio/adaptive/scene_analyzer.rs
// ====================

use std::collections::VecDeque;

/// 场景类型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneType {
    Unknown,        // 未知/校准中
    OpenOffice,     // 开放办公区
    MeetingRoom,    // 会议室
    Quiet,          // 安静环境（家、录音室）
    Noisy,          // 极端嘈杂（咖啡厅、街道）
}

impl SceneType {
    pub fn display_name(&self) -> &'static str {
        match self {
            SceneType::Unknown => "检测中...",
            SceneType::OpenOffice => "开放办公区",
            SceneType::MeetingRoom => "会议室",
            SceneType::Quiet => "安静环境",
            SceneType::Noisy => "嘈杂环境",
        }
    }
}

/// 场景特征
#[derive(Clone, Debug, Default)]
pub struct SceneFeatures {
    pub noise_floor_db: f32,          // 噪底电平
    pub noise_variance_db: f32,       // 噪声波动
    pub reverb_amount: f32,           // 混响量 (0-1)
    pub spectral_centroid: f32,       // 频谱质心 (Hz)
    pub spectral_flatness: f32,       // 频谱平坦度 (0-1)
    pub voice_activity_ratio: f32,    // 背景人声活动比例
    pub transient_density: f32,       // 瞬态密度
    pub low_freq_energy_ratio: f32,   // 低频能量占比
}

/// 场景分析结果
#[derive(Clone, Debug)]
pub struct SceneAnalysis {
    pub scene_type: SceneType,
    pub confidence: f32,              // 0-1
    pub features: SceneFeatures,
    pub transition_progress: f32,     // 场景切换进度 0-1
}

/// 场景分析器
pub struct SceneAnalyzer {
    sample_rate: f32,
    
    // === 状态 ===
    current_scene: SceneType,
    target_scene: SceneType,
    scene_confidence: f32,
    transition_progress: f32,
    
    // === 特征提取 ===
    energy_history: VecDeque<f32>,
    reverb_estimator: ReverbEstimator,
    voice_detector: BackgroundVoiceDetector,
    spectrum_analyzer: SpectrumAnalyzer,
    
    // === 决策控制 ===
    scene_hold_frames: usize,
    scene_hold_counter: usize,
    min_confidence_for_switch: f32,
    transition_frames: usize,
    transition_counter: usize,
    
    // === 校准 ===
    is_calibrating: bool,
    calibration_frames: usize,
    calibration_counter: usize,
    calibration_features: Vec<SceneFeatures>,
    baseline_features: Option<SceneFeatures>,
}

impl SceneAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        let frame_duration_ms = 10.0;
        let history_duration_sec = 5.0;
        let history_length = (history_duration_sec * 1000.0 / frame_duration_ms) as usize;
        
        let calibration_duration_sec = 3.0;
        let transition_duration_sec = 2.0;
        let hold_duration_sec = 3.0;
        
        Self {
            sample_rate,
            
            current_scene: SceneType::Unknown,
            target_scene: SceneType::Unknown,
            scene_confidence: 0.0,
            transition_progress: 1.0,
            
            energy_history: VecDeque::with_capacity(history_length),
            reverb_estimator: ReverbEstimator::new(sample_rate),
            voice_detector: BackgroundVoiceDetector::new(sample_rate),
            spectrum_analyzer: SpectrumAnalyzer::new(sample_rate),
            
            scene_hold_frames: (hold_duration_sec * 1000.0 / frame_duration_ms) as usize,
            scene_hold_counter: 0,
            min_confidence_for_switch: 0.7,
            transition_frames: (transition_duration_sec * 1000.0 / frame_duration_ms) as usize,
            transition_counter: 0,
            
            is_calibrating: true,
            calibration_frames: (calibration_duration_sec * 1000.0 / frame_duration_ms) as usize,
            calibration_counter: 0,
            calibration_features: Vec::new(),
            baseline_features: None,
        }
    }
    
    /// 分析音频帧，返回场景分析结果
    pub fn analyze(&mut self, samples: &[f32], is_user_speaking: bool) -> SceneAnalysis {
        // 仅在用户不说话时分析环境
        if !is_user_speaking {
            self.update_features(samples);
        }
        
        // 校准阶段
        if self.is_calibrating {
            self.run_calibration(samples, is_user_speaking);
            return SceneAnalysis {
                scene_type: SceneType::Unknown,
                confidence: 0.0,
                features: self.extract_features(),
                transition_progress: 0.0,
            };
        }
        
        // 提取当前特征
        let features = self.extract_features();
        
        // 分类
        let (detected_scene, confidence) = self.classify_scene(&features);
        
        // 场景切换决策
        self.update_scene_decision(detected_scene, confidence);
        
        // 更新过渡进度
        self.update_transition();
        
        SceneAnalysis {
            scene_type: self.current_scene,
            confidence: self.scene_confidence,
            features,
            transition_progress: self.transition_progress,
        }
    }
    
    /// 获取当前场景
    pub fn current_scene(&self) -> SceneType {
        self.current_scene
    }
    
    /// 获取目标场景（正在切换到的场景）
    pub fn target_scene(&self) -> SceneType {
        self.target_scene
    }
    
    /// 获取过渡进度 (0=当前场景, 1=目标场景)
    pub fn transition_progress(&self) -> f32 {
        self.transition_progress
    }
    
    /// 是否正在校准
    pub fn is_calibrating(&self) -> bool {
        self.is_calibrating
    }
    
    /// 强制重新校准
    pub fn recalibrate(&mut self) {
        self.is_calibrating = true;
        self.calibration_counter = 0;
        self.calibration_features.clear();
        self.current_scene = SceneType::Unknown;
        self.target_scene = SceneType::Unknown;
    }
    
    fn update_features(&mut self, samples: &[f32]) {
        // 能量
        let energy = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let energy_db = 10.0 * energy.max(1e-10).log10();
        self.energy_history.push_back(energy_db);
        if self.energy_history.len() > 500 {
            self.energy_history.pop_front();
        }
        
        // 混响
        self.reverb_estimator.process(samples);
        
        // 背景人声
        self.voice_detector.process(samples);
        
        // 频谱
        self.spectrum_analyzer.process(samples);
    }
    
    fn extract_features(&self) -> SceneFeatures {
        // 噪底：取能量历史的 10% 分位数
        let noise_floor_db = self.compute_percentile(0.1);
        
        // 噪声波动：能量标准差
        let noise_variance_db = self.compute_energy_variance();
        
        SceneFeatures {
            noise_floor_db,
            noise_variance_db,
            reverb_amount: self.reverb_estimator.reverb_amount(),
            spectral_centroid: self.spectrum_analyzer.centroid(),
            spectral_flatness: self.spectrum_analyzer.flatness(),
            voice_activity_ratio: self.voice_detector.activity_ratio(),
            transient_density: self.spectrum_analyzer.transient_density(),
            low_freq_energy_ratio: self.spectrum_analyzer.low_freq_ratio(),
        }
    }
    
    fn classify_scene(&self, features: &SceneFeatures) -> (SceneType, f32) {
        // 多维度评分
        let mut scores = [
            (SceneType::Quiet, 0.0f32),
            (SceneType::MeetingRoom, 0.0f32),
            (SceneType::OpenOffice, 0.0f32),
            (SceneType::Noisy, 0.0f32),
        ];
        
        // === 安静环境特征 ===
        // 噪底低、无背景人声、波动小
        if features.noise_floor_db < -50.0 {
            scores[0].1 += 0.4;
        }
        if features.voice_activity_ratio < 0.05 {
            scores[0].1 += 0.3;
        }
        if features.noise_variance_db < 3.0 {
            scores[0].1 += 0.3;
        }
        
        // === 会议室特征 ===
        // 中等噪底、有混响、低频突出、少量背景人声
        if features.noise_floor_db >= -50.0 && features.noise_floor_db < -35.0 {
            scores[1].1 += 0.25;
        }
        if features.reverb_amount > 0.3 {
            scores[1].1 += 0.25;
        }
        if features.low_freq_energy_ratio > 0.4 {
            scores[1].1 += 0.25;
        }
        if features.voice_activity_ratio < 0.2 {
            scores[1].1 += 0.25;
        }
        
        // === 开放办公区特征 ===
        // 较高噪底、频繁背景人声、噪声波动大、混响小
        if features.noise_floor_db >= -40.0 && features.noise_floor_db < -25.0 {
            scores[2].1 += 0.2;
        }
        if features.voice_activity_ratio > 0.3 {
            scores[2].1 += 0.3;
        }
        if features.noise_variance_db > 5.0 {
            scores[2].1 += 0.25;
        }
        if features.reverb_amount < 0.3 {
            scores[2].1 += 0.15;
        }
        if features.spectral_flatness > 0.4 {
            scores[2].1 += 0.1;
        }
        
        // === 极端嘈杂特征 ===
        // 非常高的噪底、持续高活动
        if features.noise_floor_db >= -25.0 {
            scores[3].1 += 0.5;
        }
        if features.voice_activity_ratio > 0.6 {
            scores[3].1 += 0.3;
        }
        if features.noise_variance_db > 8.0 {
            scores[3].1 += 0.2;
        }
        
        // 归一化并选择最高分
        let total: f32 = scores.iter().map(|(_, s)| s).sum();
        if total > 0.0 {
            for (_, score) in &mut scores {
                *score /= total;
            }
        }
        
        // 选择最高分
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        (scores[0].0, scores[0].1)
    }
    
    fn update_scene_decision(&mut self, detected_scene: SceneType, confidence: f32) {
        // 置信度平滑
        self.scene_confidence = 0.9 * self.scene_confidence + 0.1 * confidence;
        
        // 检测到不同场景
        if detected_scene != self.current_scene && detected_scene != self.target_scene {
            if self.scene_confidence > self.min_confidence_for_switch {
                self.scene_hold_counter += 1;
                
                // 持续检测到新场景才切换
                if self.scene_hold_counter >= self.scene_hold_frames {
                    self.target_scene = detected_scene;
                    self.transition_counter = 0;
                    self.scene_hold_counter = 0;
                    log::info!(
                        "场景切换: {:?} → {:?} (置信度: {:.0}%)",
                        self.current_scene, self.target_scene, self.scene_confidence * 100.0
                    );
                }
            } else {
                self.scene_hold_counter = 0;
            }
        } else {
            self.scene_hold_counter = 0;
        }
    }
    
    fn update_transition(&mut self) {
        if self.current_scene != self.target_scene {
            self.transition_counter += 1;
            self.transition_progress = 
                (self.transition_counter as f32 / self.transition_frames as f32).min(1.0);
            
            // 过渡完成
            if self.transition_counter >= self.transition_frames {
                self.current_scene = self.target_scene;
                self.transition_progress = 1.0;
                log::info!("场景切换完成: {:?}", self.current_scene);
            }
        }
    }
    
    fn run_calibration(&mut self, samples: &[f32], is_user_speaking: bool) {
        if is_user_speaking {
            return;  // 用户说话时不校准
        }
        
        self.calibration_counter += 1;
        
        // 收集特征
        if self.calibration_counter % 10 == 0 {  // 每 100ms 采样一次
            self.calibration_features.push(self.extract_features());
        }
        
        // 校准完成
        if self.calibration_counter >= self.calibration_frames {
            self.finish_calibration();
        }
    }
    
    fn finish_calibration(&mut self) {
        if self.calibration_features.is_empty() {
            log::warn!("校准数据不足，使用默认基线");
            self.baseline_features = Some(SceneFeatures::default());
        } else {
            // 计算平均特征作为基线
            let n = self.calibration_features.len() as f32;
            let mut baseline = SceneFeatures::default();
            
            for f in &self.calibration_features {
                baseline.noise_floor_db += f.noise_floor_db;
                baseline.noise_variance_db += f.noise_variance_db;
                baseline.reverb_amount += f.reverb_amount;
                baseline.spectral_centroid += f.spectral_centroid;
                baseline.spectral_flatness += f.spectral_flatness;
                baseline.voice_activity_ratio += f.voice_activity_ratio;
                baseline.low_freq_energy_ratio += f.low_freq_energy_ratio;
            }
            
            baseline.noise_floor_db /= n;
            baseline.noise_variance_db /= n;
            baseline.reverb_amount /= n;
            baseline.spectral_centroid /= n;
            baseline.spectral_flatness /= n;
            baseline.voice_activity_ratio /= n;
            baseline.low_freq_energy_ratio /= n;
            
            self.baseline_features = Some(baseline.clone());
            
            // 初始场景分类
            let (initial_scene, confidence) = self.classify_scene(&baseline);
            self.current_scene = initial_scene;
            self.target_scene = initial_scene;
            self.scene_confidence = confidence;
            
            log::info!(
                "校准完成: 场景={:?}, 噪底={:.1}dB, 混响={:.2}, 背景人声={:.0}%",
                initial_scene,
                baseline.noise_floor_db,
                baseline.reverb_amount,
                baseline.voice_activity_ratio * 100.0
            );
        }
        
        self.is_calibrating = false;
        self.calibration_features.clear();
    }
    
    fn compute_percentile(&self, p: f32) -> f32 {
        if self.energy_history.is_empty() {
            return -60.0;
        }
        let mut sorted: Vec<f32> = self.energy_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f32 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
    
    fn compute_energy_variance(&self) -> f32 {
        if self.energy_history.len() < 2 {
            return 0.0;
        }
        let mean: f32 = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
        let variance: f32 = self.energy_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.energy_history.len() as f32;
        variance.sqrt()
    }
}
```

### 3.2 混响估计器

```rust
// ====================
// 文件: audio/adaptive/reverb_estimator.rs
// ====================

/// 混响估计器 - 基于能量衰减特性
pub struct ReverbEstimator {
    sample_rate: f32,
    
    // 能量包络
    envelope: f32,
    attack_coef: f32,
    release_coef: f32,
    
    // 衰减分析
    decay_history: VecDeque<f32>,
    decay_rate: f32,           // 衰减速率 (dB/s)
    
    // 混响量估计
    reverb_amount: f32,
}

impl ReverbEstimator {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            envelope: 0.0,
            attack_coef: Self::calc_coef(1.0, sample_rate),
            release_coef: Self::calc_coef(100.0, sample_rate),
            decay_history: VecDeque::with_capacity(100),
            decay_rate: 0.0,
            reverb_amount: 0.0,
        }
    }
    
    fn calc_coef(time_ms: f32, sample_rate: f32) -> f32 {
        (-1000.0 / (time_ms * sample_rate)).exp()
    }
    
    pub fn process(&mut self, samples: &[f32]) {
        for &sample in samples {
            let abs_sample = sample.abs();
            
            let coef = if abs_sample > self.envelope {
                self.attack_coef
            } else {
                self.release_coef
            };
            
            self.envelope = coef * self.envelope + (1.0 - coef) * abs_sample;
        }
        
        // 记录包络用于衰减分析
        let env_db = 20.0 * self.envelope.max(1e-10).log10();
        self.decay_history.push_back(env_db);
        if self.decay_history.len() > 100 {
            self.decay_history.pop_front();
        }
        
        // 估计衰减速率
        self.estimate_decay_rate();
        
        // 混响量映射
        // 衰减慢 = 混响多，衰减快 = 混响少
        // 典型值：无混响 > 60 dB/s，强混响 < 20 dB/s
        self.reverb_amount = 1.0 - ((self.decay_rate - 20.0) / 40.0).clamp(0.0, 1.0);
    }
    
    pub fn reverb_amount(&self) -> f32 {
        self.reverb_amount
    }
    
    fn estimate_decay_rate(&mut self) {
        if self.decay_history.len() < 20 {
            return;
        }
        
        // 寻找下降段
        let history: Vec<f32> = self.decay_history.iter().copied().collect();
        let mut max_idx = 0;
        let mut max_val = history[0];
        
        for (i, &val) in history.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        // 从峰值开始计算衰减
        if max_idx < history.len() - 10 {
            let start_db = history[max_idx];
            let end_db = history[history.len() - 1];
            let duration_sec = (history.len() - max_idx) as f32 * 0.01;  // 假设 10ms/帧
            
            if duration_sec > 0.0 {
                let rate = (start_db - end_db) / duration_sec;
                self.decay_rate = 0.9 * self.decay_rate + 0.1 * rate.max(0.0);
            }
        }
    }
}
```

### 3.3 背景人声检测器

```rust
// ====================
// 文件: audio/adaptive/voice_detector.rs
// ====================

use std::collections::VecDeque;

/// 背景人声检测器
pub struct BackgroundVoiceDetector {
    sample_rate: f32,
    
    // 频带能量
    band_energies: [f32; 4],  // 低频、中低、中高、高频
    
    // 语音特征
    speech_likelihood: f32,
    activity_history: VecDeque<bool>,
    activity_ratio: f32,
    
    // 简单滤波器状态
    filter_states: [[f32; 2]; 4],
}

impl BackgroundVoiceDetector {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            band_energies: [0.0; 4],
            speech_likelihood: 0.0,
            activity_history: VecDeque::with_capacity(500),
            activity_ratio: 0.0,
            filter_states: [[0.0; 2]; 4],
        }
    }
    
    pub fn process(&mut self, samples: &[f32]) {
        // 简化的频带能量计算
        // 实际应用中可以使用更精确的滤波器组
        self.compute_band_energies(samples);
        
        // 语音特征检测
        // 语音主要能量集中在 300Hz - 3kHz
        let mid_energy = self.band_energies[1] + self.band_energies[2];
        let total_energy: f32 = self.band_energies.iter().sum();
        
        let speech_band_ratio = if total_energy > 1e-10 {
            mid_energy / total_energy
        } else {
            0.0
        };
        
        // 语音的频谱形状特征
        // 语音：中频突出，高低频相对弱
        // 噪声：相对平坦
        let is_speech_like = speech_band_ratio > 0.5 && total_energy > 1e-8;
        
        // 更新活动历史
        self.activity_history.push_back(is_speech_like);
        if self.activity_history.len() > 500 {
            self.activity_history.pop_front();
        }
        
        // 计算活动比例
        let active_count = self.activity_history.iter().filter(|&&x| x).count();
        self.activity_ratio = active_count as f32 / self.activity_history.len().max(1) as f32;
        
        // 平滑
        let target = if is_speech_like { 1.0 } else { 0.0 };
        self.speech_likelihood = 0.95 * self.speech_likelihood + 0.05 * target;
    }
    
    pub fn activity_ratio(&self) -> f32 {
        self.activity_ratio
    }
    
    pub fn speech_likelihood(&self) -> f32 {
        self.speech_likelihood
    }
    
    fn compute_band_energies(&mut self, samples: &[f32]) {
        // 简化实现：使用一阶滤波器近似频带分离
        // 频带划分：0-300Hz, 300-1kHz, 1k-3kHz, 3k-8kHz
        
        let mut band_sums = [0.0f32; 4];
        
        // 简化：使用样本的不同特征近似频带
        for &sample in samples {
            // 全带能量
            let energy = sample * sample;
            
            // 简化的频带分配（实际应用需要真正的滤波器）
            band_sums[0] += energy * 0.2;  // 低频假设
            band_sums[1] += energy * 0.3;  // 中低频
            band_sums[2] += energy * 0.35; // 中高频
            band_sums[3] += energy * 0.15; // 高频
        }
        
        let n = samples.len() as f32;
        for i in 0..4 {
            let new_energy = band_sums[i] / n.max(1.0);
            self.band_energies[i] = 0.9 * self.band_energies[i] + 0.1 * new_energy;
        }
    }
}
```

### 3.4 频谱分析器

```rust
// ====================
// 文件: audio/adaptive/spectrum_analyzer.rs
// ====================

use std::collections::VecDeque;

/// 频谱分析器 - 提取频谱特征
pub struct SpectrumAnalyzer {
    sample_rate: f32,
    
    // 特征
    centroid: f32,           // 频谱质心
    flatness: f32,           // 频谱平坦度
    low_freq_ratio: f32,     // 低频能量占比
    transient_density: f32,  // 瞬态密度
    
    // 瞬态检测
    prev_energy: f32,
    transient_history: VecDeque<bool>,
}

impl SpectrumAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            centroid: 1000.0,
            flatness: 0.5,
            low_freq_ratio: 0.3,
            transient_density: 0.0,
            prev_energy: 0.0,
            transient_history: VecDeque::with_capacity(100),
        }
    }
    
    pub fn process(&mut self, samples: &[f32]) {
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        
        // 瞬态检测：能量突变
        let energy_ratio = energy / self.prev_energy.max(1e-10);
        let is_transient = energy_ratio > 3.0 || energy_ratio < 0.33;
        
        self.transient_history.push_back(is_transient);
        if self.transient_history.len() > 100 {
            self.transient_history.pop_front();
        }
        
        let transient_count = self.transient_history.iter().filter(|&&x| x).count();
        self.transient_density = transient_count as f32 / self.transient_history.len().max(1) as f32;
        
        self.prev_energy = energy;
        
        // 简化的频谱特征（实际应用需要 FFT）
        // 这里使用零交叉率作为频谱质心的近似
        let mut zcr_count = 0usize;
        let mut prev_sign = samples.first().map(|&s| s >= 0.0).unwrap_or(false);
        
        for &sample in samples.iter().skip(1) {
            let current_sign = sample >= 0.0;
            if current_sign != prev_sign {
                zcr_count += 1;
            }
            prev_sign = current_sign;
        }
        
        let zcr = zcr_count as f32 / samples.len().max(1) as f32;
        
        // ZCR 映射到频率估计
        let estimated_freq = zcr * self.sample_rate / 2.0;
        self.centroid = 0.9 * self.centroid + 0.1 * estimated_freq;
        
        // 平坦度和低频比例使用默认值（需要 FFT 才能准确计算）
        // 在完整实现中应该使用 rustfft
    }
    
    pub fn centroid(&self) -> f32 {
        self.centroid
    }
    
    pub fn flatness(&self) -> f32 {
        self.flatness
    }
    
    pub fn low_freq_ratio(&self) -> f32 {
        self.low_freq_ratio
    }
    
    pub fn transient_density(&self) -> f32 {
        self.transient_density
    }
}
```

---

## 四、参数调度模块

### 4.1 场景参数预设

```rust
// ====================
// 文件: audio/adaptive/scene_presets.rs
// ====================

use super::SceneType;

/// 场景参数集
#[derive(Clone, Debug)]
pub struct SceneParameters {
    // === DeepFilter ===
    pub df_atten_lim: f32,
    pub df_min_thresh: f32,
    pub df_mix: f32,
    
    // === 高通 ===
    pub highpass_cutoff: f32,
    
    // === 噪声门 ===
    pub noise_gate_threshold: f32,
    pub noise_gate_floor: f32,
    pub noise_gate_hold_ms: f32,
    
    // === 空间滤波 ===
    pub spatial_enabled: bool,
    pub spatial_sensitivity: f32,
    pub spatial_attenuation: f32,
    
    // === 语音隔离 ===
    pub voice_isolator_enabled: bool,
    pub voice_isolator_threshold: f32,
    pub voice_isolator_suppression: f32,
    
    // === 频谱门限 ===
    pub spectral_gate_enabled: bool,
    pub spectral_gate_threshold: f32,
    pub spectral_gate_floor: f32,
    
    // === EQ ===
    pub eq_preset: EqPresetKind,
    pub eq_mix: f32,
    
    // === 瞬态 ===
    pub transient_enabled: bool,
    pub transient_gain: f32,
    pub transient_sustain: f32,
    
    // === AGC ===
    pub agc_target: f32,
    pub agc_max_gain: f32,
    pub agc_max_atten: f32,
}

impl SceneParameters {
    /// 获取场景预设参数
    pub fn for_scene(scene: SceneType) -> Self {
        match scene {
            SceneType::Unknown => Self::default(),
            SceneType::Quiet => Self::quiet_preset(),
            SceneType::MeetingRoom => Self::meeting_room_preset(),
            SceneType::OpenOffice => Self::open_office_preset(),
            SceneType::Noisy => Self::noisy_preset(),
        }
    }
    
    /// 安静环境预设
    fn quiet_preset() -> Self {
        Self {
            df_atten_lim: 25.0,
            df_min_thresh: -60.0,
            df_mix: 0.8,
            
            highpass_cutoff: 50.0,
            
            noise_gate_threshold: -50.0,
            noise_gate_floor: -30.0,
            noise_gate_hold_ms: 200.0,
            
            spatial_enabled: false,
            spatial_sensitivity: 0.5,
            spatial_attenuation: -12.0,
            
            voice_isolator_enabled: false,
            voice_isolator_threshold: 0.5,
            voice_isolator_suppression: -15.0,
            
            spectral_gate_enabled: false,
            spectral_gate_threshold: 6.0,
            spectral_gate_floor: -15.0,
            
            eq_preset: EqPresetKind::Broadcast,
            eq_mix: 0.6,
            
            transient_enabled: true,
            transient_gain: 3.0,
            transient_sustain: 0.0,
            
            agc_target: -14.0,
            agc_max_gain: 12.0,
            agc_max_atten: 10.0,
        }
    }
    
    /// 会议室预设
    fn meeting_room_preset() -> Self {
        Self {
            df_atten_lim: 35.0,
            df_min_thresh: -55.0,
            df_mix: 0.9,
            
            highpass_cutoff: 70.0,  // 切除驻波
            
            noise_gate_threshold: -45.0,
            noise_gate_floor: -24.0,
            noise_gate_hold_ms: 180.0,
            
            spatial_enabled: true,
            spatial_sensitivity: 0.6,
            spatial_attenuation: -15.0,
            
            voice_isolator_enabled: true,
            voice_isolator_threshold: 0.5,
            voice_isolator_suppression: -15.0,
            
            spectral_gate_enabled: true,
            spectral_gate_threshold: 8.0,
            spectral_gate_floor: -15.0,
            
            eq_preset: EqPresetKind::ConferenceHall,
            eq_mix: 0.7,
            
            transient_enabled: true,
            transient_gain: 3.5,
            transient_sustain: -2.0,  // 减少混响尾音
            
            agc_target: -16.0,
            agc_max_gain: 10.0,
            agc_max_atten: 12.0,
        }
    }
    
    /// 开放办公区预设
    fn open_office_preset() -> Self {
        Self {
            df_atten_lim: 50.0,
            df_min_thresh: -50.0,
            df_mix: 1.0,
            
            highpass_cutoff: 85.0,
            
            noise_gate_threshold: -38.0,
            noise_gate_floor: -22.0,
            noise_gate_hold_ms: 150.0,
            
            spatial_enabled: true,
            spatial_sensitivity: 0.35,  // 更严格
            spatial_attenuation: -20.0,
            
            voice_isolator_enabled: true,
            voice_isolator_threshold: 0.45,
            voice_isolator_suppression: -22.0,
            
            spectral_gate_enabled: true,
            spectral_gate_threshold: 10.0,
            spectral_gate_floor: -18.0,
            
            eq_preset: EqPresetKind::OpenOffice,
            eq_mix: 0.8,
            
            transient_enabled: true,
            transient_gain: 5.0,
            transient_sustain: 0.0,
            
            agc_target: -14.0,
            agc_max_gain: 8.0,
            agc_max_atten: 15.0,
        }
    }
    
    /// 极端嘈杂预设
    fn noisy_preset() -> Self {
        Self {
            df_atten_lim: 65.0,
            df_min_thresh: -45.0,
            df_mix: 1.0,
            
            highpass_cutoff: 100.0,
            
            noise_gate_threshold: -32.0,
            noise_gate_floor: -20.0,
            noise_gate_hold_ms: 120.0,
            
            spatial_enabled: true,
            spatial_sensitivity: 0.25,
            spatial_attenuation: -24.0,
            
            voice_isolator_enabled: true,
            voice_isolator_threshold: 0.4,
            voice_isolator_suppression: -25.0,
            
            spectral_gate_enabled: true,
            spectral_gate_threshold: 12.0,
            spectral_gate_floor: -20.0,
            
            eq_preset: EqPresetKind::Meeting,
            eq_mix: 0.85,
            
            transient_enabled: true,
            transient_gain: 6.0,
            transient_sustain: -3.0,
            
            agc_target: -12.0,
            agc_max_gain: 6.0,
            agc_max_atten: 18.0,
        }
    }
}

impl Default for SceneParameters {
    fn default() -> Self {
        Self::meeting_room_preset()  // 默认使用会议室预设
    }
}

// EqPresetKind 引用
use crate::audio::eq::EqPresetKind;
```

### 4.2 参数调度器

```rust
// ====================
// 文件: audio/adaptive/parameter_scheduler.rs
// ====================

use super::{SceneParameters, SceneType, SceneAnalysis};

/// 参数调度器 - 平滑参数过渡
pub struct ParameterScheduler {
    current_params: SceneParameters,
    source_params: SceneParameters,
    target_params: SceneParameters,
    
    transition_progress: f32,
    
    // 用户覆盖
    user_overrides: UserOverrides,
}

/// 用户手动覆盖的参数
#[derive(Clone, Debug, Default)]
pub struct UserOverrides {
    pub df_atten_lim: Option<f32>,
    pub highpass_cutoff: Option<f32>,
    pub spatial_sensitivity: Option<f32>,
    pub eq_mix: Option<f32>,
    pub agc_target: Option<f32>,
    // ... 其他可覆盖参数
}

impl ParameterScheduler {
    pub fn new() -> Self {
        let default_params = SceneParameters::default();
        Self {
            current_params: default_params.clone(),
            source_params: default_params.clone(),
            target_params: default_params,
            transition_progress: 1.0,
            user_overrides: UserOverrides::default(),
        }
    }
    
    /// 更新场景，触发参数过渡
    pub fn update_scene(&mut self, analysis: &SceneAnalysis) {
        let new_target = SceneParameters::for_scene(analysis.scene_type);
        
        // 检测目标变化
        if analysis.transition_progress < 1.0 {
            if self.transition_progress >= 1.0 {
                // 开始新的过渡
                self.source_params = self.current_params.clone();
                self.target_params = new_target;
            }
            self.transition_progress = analysis.transition_progress;
        } else {
            self.transition_progress = 1.0;
        }
        
        // 插值计算当前参数
        self.interpolate_params();
        
        // 应用用户覆盖
        self.apply_user_overrides();
    }
    
    /// 获取当前生效参数
    pub fn current_params(&self) -> &SceneParameters {
        &self.current_params
    }
    
    /// 设置用户覆盖
    pub fn set_override<F>(&mut self, setter: F)
    where
        F: FnOnce(&mut UserOverrides),
    {
        setter(&mut self.user_overrides);
    }
    
    /// 清除所有用户覆盖
    pub fn clear_overrides(&mut self) {
        self.user_overrides = UserOverrides::default();
    }
    
    fn interpolate_params(&mut self) {
        let t = self.ease_in_out(self.transition_progress);
        
        // 线性插值各参数
        self.current_params.df_atten_lim = Self::lerp(
            self.source_params.df_atten_lim,
            self.target_params.df_atten_lim,
            t,
        );
        
        self.current_params.df_min_thresh = Self::lerp(
            self.source_params.df_min_thresh,
            self.target_params.df_min_thresh,
            t,
        );
        
        self.current_params.df_mix = Self::lerp(
            self.source_params.df_mix,
            self.target_params.df_mix,
            t,
        );
        
        self.current_params.highpass_cutoff = Self::lerp(
            self.source_params.highpass_cutoff,
            self.target_params.highpass_cutoff,
            t,
        );
        
        self.current_params.noise_gate_threshold = Self::lerp(
            self.source_params.noise_gate_threshold,
            self.target_params.noise_gate_threshold,
            t,
        );
        
        self.current_params.noise_gate_floor = Self::lerp(
            self.source_params.noise_gate_floor,
            self.target_params.noise_gate_floor,
            t,
        );
        
        self.current_params.spatial_sensitivity = Self::lerp(
            self.source_params.spatial_sensitivity,
            self.target_params.spatial_sensitivity,
            t,
        );
        
        self.current_params.spatial_attenuation = Self::lerp(
            self.source_params.spatial_attenuation,
            self.target_params.spatial_attenuation,
            t,
        );
        
        self.current_params.voice_isolator_threshold = Self::lerp(
            self.source_params.voice_isolator_threshold,
            self.target_params.voice_isolator_threshold,
            t,
        );
        
        self.current_params.voice_isolator_suppression = Self::lerp(
            self.source_params.voice_isolator_suppression,
            self.target_params.voice_isolator_suppression,
            t,
        );
        
        self.current_params.spectral_gate_threshold = Self::lerp(
            self.source_params.spectral_gate_threshold,
            self.target_params.spectral_gate_threshold,
            t,
        );
        
        self.current_params.spectral_gate_floor = Self::lerp(
            self.source_params.spectral_gate_floor,
            self.target_params.spectral_gate_floor,
            t,
        );
        
        self.current_params.eq_mix = Self::lerp(
            self.source_params.eq_mix,
            self.target_params.eq_mix,
            t,
        );
        
        self.current_params.transient_gain = Self::lerp(
            self.source_params.transient_gain,
            self.target_params.transient_gain,
            t,
        );
        
        self.current_params.transient_sustain = Self::lerp(
            self.source_params.transient_sustain,
            self.target_params.transient_sustain,
            t,
        );
        
        self.current_params.agc_target = Self::lerp(
            self.source_params.agc_target,
            self.target_params.agc_target,
            t,
        );
        
        self.current_params.agc_max_gain = Self::lerp(
            self.source_params.agc_max_gain,
            self.target_params.agc_max_gain,
            t,
        );
        
        self.current_params.agc_max_atten = Self::lerp(
            self.source_params.agc_max_atten,
            self.target_params.agc_max_atten,
            t,
        );
        
        // 布尔值在过渡中点切换
        let switch_point = 0.5;
        self.current_params.spatial_enabled = if t < switch_point {
            self.source_params.spatial_enabled
        } else {
            self.target_params.spatial_enabled
        };
        
        self.current_params.voice_isolator_enabled = if t < switch_point {
            self.source_params.voice_isolator_enabled
        } else {
            self.target_params.voice_isolator_enabled
        };
        
        self.current_params.spectral_gate_enabled = if t < switch_point {
            self.source_params.spectral_gate_enabled
        } else {
            self.target_params.spectral_gate_enabled
        };
        
        // EQ 预设在过渡中点切换
        self.current_params.eq_preset = if t < switch_point {
            self.source_params.eq_preset
        } else {
            self.target_params.eq_preset
        };
    }
    
    fn apply_user_overrides(&mut self) {
        if let Some(v) = self.user_overrides.df_atten_lim {
            self.current_params.df_atten_lim = v;
        }
        if let Some(v) = self.user_overrides.highpass_cutoff {
            self.current_params.highpass_cutoff = v;
        }
        if let Some(v) = self.user_overrides.spatial_sensitivity {
            self.current_params.spatial_sensitivity = v;
        }
        if let Some(v) = self.user_overrides.eq_mix {
            self.current_params.eq_mix = v;
        }
        if let Some(v) = self.user_overrides.agc_target {
            self.current_params.agc_target = v;
        }
    }
    
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    
    fn ease_in_out(&self, t: f32) -> f32 {
        // 平滑的 ease-in-out 曲线
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        }
    }
}
```

---

## 五、集成到处理链

### 5.1 自适应处理管理器

```rust
// ====================
// 文件: audio/adaptive/mod.rs
// ====================

mod scene_analyzer;
mod scene_presets;
mod parameter_scheduler;
mod reverb_estimator;
mod voice_detector;
mod spectrum_analyzer;

pub use scene_analyzer::{SceneAnalyzer, SceneType, SceneFeatures, SceneAnalysis};
pub use scene_presets::SceneParameters;
pub use parameter_scheduler::{ParameterScheduler, UserOverrides};

use crate::audio::processors::*;
use crate::audio::bus::ProcessContext;

/// 自适应处理管理器
pub struct AdaptiveProcessor {
    // 场景分析
    scene_analyzer: SceneAnalyzer,
    parameter_scheduler: ParameterScheduler,
    
    // 处理模块
    noise_gate: NoiseGate,
    spatial_filter: SpatialFilter,
    voice_isolator: VoiceIsolator,
    spectral_gate: SpectralGate,
    
    // 状态
    enabled: bool,
    last_analysis: Option<SceneAnalysis>,
}

impl AdaptiveProcessor {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            scene_analyzer: SceneAnalyzer::new(sample_rate),
            parameter_scheduler: ParameterScheduler::new(),
            
            noise_gate: NoiseGate::new(sample_rate),
            spatial_filter: SpatialFilter::new(sample_rate),
            voice_isolator: VoiceIsolator::new(sample_rate),
            spectral_gate: SpectralGate::new(sample_rate),
            
            enabled: true,
            last_analysis: None,
        }
    }
    
    /// 启用/禁用自适应处理
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// 强制重新校准
    pub fn recalibrate(&mut self) {
        self.scene_analyzer.recalibrate();
    }
    
    /// 获取当前场景
    pub fn current_scene(&self) -> SceneType {
        self.scene_analyzer.current_scene()
    }
    
    /// 获取最新分析结果
    pub fn last_analysis(&self) -> Option<&SceneAnalysis> {
        self.last_analysis.as_ref()
    }
    
    /// 设置用户覆盖参数
    pub fn set_user_override<F>(&mut self, setter: F)
    where
        F: FnOnce(&mut UserOverrides),
    {
        self.parameter_scheduler.set_override(setter);
    }
    
    /// 处理音频（DeepFilter 之前调用）
    pub fn process_pre_df(
        &mut self,
        samples: &mut [f32],
        ctx: &mut ProcessContext,
        is_user_speaking: bool,
    ) {
        if !self.enabled {
            return;
        }
        
        // 场景分析
        let analysis = self.scene_analyzer.analyze(samples, is_user_speaking);
        self.last_analysis = Some(analysis.clone());
        
        // 更新参数
        self.parameter_scheduler.update_scene(&analysis);
        let params = self.parameter_scheduler.current_params();
        
        // 应用参数到模块
        self.apply_params_to_modules(params);
        
        // 执行预处理
        self.noise_gate.process(samples, ctx);
        
        if params.spatial_enabled {
            self.spatial_filter.process(samples, ctx);
            ctx.is_near_field = self.spatial_filter.is_near_field();
        }
    }
    
    /// 处理音频（DeepFilter 之后调用）
    pub fn process_post_df(
        &mut self,
        samples: &mut [f32],
        ctx: &mut ProcessContext,
    ) {
        if !self.enabled {
            return;
        }
        
        let params = self.parameter_scheduler.current_params();
        
        if params.voice_isolator_enabled {
            self.voice_isolator.process(samples, ctx);
            ctx.voice_activity = if self.voice_isolator.is_speaking() { 1.0 } else { 0.3 };
        }
        
        if params.spectral_gate_enabled {
            self.spectral_gate.process(samples, ctx);
        }
    }
    
    /// 获取当前参数供其他模块使用
    pub fn current_params(&self) -> &SceneParameters {
        self.parameter_scheduler.current_params()
    }
    
    fn apply_params_to_modules(&mut self, params: &SceneParameters) {
        // NoiseGate
        self.noise_gate.set_threshold(params.noise_gate_threshold);
        self.noise_gate.set_floor_db(params.noise_gate_floor);
        self.noise_gate.set_hold_ms(params.noise_gate_hold_ms);
        
        // SpatialFilter
        self.spatial_filter.set_sensitivity(params.spatial_sensitivity);
        self.spatial_filter.set_far_field_attenuation_db(params.spatial_attenuation);
        
        // VoiceIsolator
        self.voice_isolator.set_continuity_threshold(params.voice_isolator_threshold);
        self.voice_isolator.set_suppression_db(params.voice_isolator_suppression);
        
        // SpectralGate
        self.spectral_gate.set_threshold_db(params.spectral_gate_threshold);
        self.spectral_gate.set_floor_db(params.spectral_gate_floor);
    }
}
```

### 5.2 集成到 capture.rs

```rust
// ====================
// 修改: capture.rs worker 函数
// ====================

use crate::audio::adaptive::{AdaptiveProcessor, SceneType};

// 在 worker 初始化部分:
let mut adaptive_processor = AdaptiveProcessor::new(df.sr as f32);
let mut adaptive_enabled = true;

// 处理控制消息:
ControlMessage::AdaptiveEnabled(enabled) => {
    adaptive_enabled = enabled;
    adaptive_processor.set_enabled(enabled);
    log::info!("自适应模式: {}", if enabled { "开启" } else { "关闭" });
}

ControlMessage::AdaptiveRecalibrate => {
    adaptive_processor.recalibrate();
    log::info!("触发重新校准");
}

ControlMessage::AdaptiveUserOverride(override_type, value) => {
    adaptive_processor.set_user_override(|overrides| {
        match override_type {
            OverrideType::DfAttenLim => overrides.df_atten_lim = Some(value),
            OverrideType::HighpassCutoff => overrides.highpass_cutoff = Some(value),
            OverrideType::SpatialSensitivity => overrides.spatial_sensitivity = Some(value),
            OverrideType::EqMix => overrides.eq_mix = Some(value),
            OverrideType::AgcTarget => overrides.agc_target = Some(value),
        }
    });
}

// 在处理链中:

// === 预处理阶段 ===
if let Some(buffer) = inframe.as_slice_mut() {
    // 构建上下文
    let mut ctx = ProcessContext {
        sample_rate: df.sr as f32,
        block_size: buffer.len(),
        voice_activity: 0.0,
        is_near_field: false,
        continuity_score: 0.5,
        input_level_db: calculate_level_db(buffer),
    };
    
    // 判断用户是否在说话（基于近场检测 + 能量）
    let is_user_speaking = spatial_filter.is_near_field() && ctx.input_level_db > -35.0;
    
    // 自适应预处理
    if adaptive_enabled {
        adaptive_processor.process_pre_df(buffer, &mut ctx, is_user_speaking);
        
        // 获取自适应参数，应用到其他模块
        let params = adaptive_processor.current_params();
        
        // 更新 DeepFilter 参数
        df.set_atten_lim(params.df_atten_lim);
        df.min_db_thresh = params.df_min_thresh;
        df_mix = params.df_mix;
        
        // 更新高通
        highpass.set_cutoff(params.highpass_cutoff);
        
        // 更新 EQ
        if dynamic_eq.preset() != params.eq_preset {
            dynamic_eq.apply_preset(params.eq_preset);
        }
        dynamic_eq.set_dry_wet(params.eq_mix);
        
        // 更新瞬态
        transient_shaper.set_attack_gain(params.transient_gain);
        transient_shaper.set_sustain_gain(params.transient_sustain);
        
        // 更新 AGC
        agc.set_target_level(params.agc_target);
        agc.set_max_gain(params.agc_max_gain);
        agc.set_max_attenuation(params.agc_max_atten);
    }
    
    // 高通滤波
    if highpass_enabled {
        highpass.process(buffer);
    }
}

// === DeepFilter 降噪 ===
let lsnr = df.process(inframe.view(), outframe.view_mut())?;

// === 后处理阶段 ===
if let Some(buffer) = outframe.as_slice_mut() {
    // 自适应后处理
    if adaptive_enabled {
        adaptive_processor.process_post_df(buffer, &mut ctx);
    }
    
    // 后续处理（瞬态、饱和、EQ、AGC）...
}

// === 发送状态到 UI ===
if let Some(ref sender) = s_adaptive_status {
    if let Some(analysis) = adaptive_processor.last_analysis() {
        let status = AdaptiveStatus {
            scene_type: adaptive_processor.current_scene(),
            scene_name: adaptive_processor.current_scene().display_name().to_string(),
            confidence: analysis.confidence,
            is_calibrating: adaptive_processor.scene_analyzer.is_calibrating(),
            features: analysis.features.clone(),
            transition_progress: analysis.transition_progress,
        };
        let _ = sender.try_send(status);
    }
}
```

---

## 六、UI 状态显示

### 6.1 状态结构

```rust
// ====================
// 修改: capture.rs 新增状态结构
// ====================

#[derive(Debug, Clone)]
pub struct AdaptiveStatus {
    pub scene_type: SceneType,
    pub scene_name: String,
    pub confidence: f32,
    pub is_calibrating: bool,
    pub features: SceneFeatures,
    pub transition_progress: f32,
}

pub type SendAdaptiveStatus = Sender<AdaptiveStatus>;
pub type RecvAdaptiveStatus = Receiver<AdaptiveStatus>;
```

### 6.2 UI 面板

```rust
// ====================
// 修改: main.rs 新增自适应状态面板
// ====================

impl SpecView {
    fn create_adaptive_panel(&self) -> Element<'_, Message> {
        // 场景指示器
        let scene_indicator = {
            let (icon, color) = match self.adaptive_status.scene_type {
                SceneType::Unknown => ("🔄", Color::from_rgb(0.5, 0.5, 0.5)),
                SceneType::Quiet => ("🤫", Color::from_rgb(0.3, 0.7, 0.3)),
                SceneType::MeetingRoom => ("🚪", Color::from_rgb(0.3, 0.5, 0.8)),
                SceneType::OpenOffice => ("🏢", Color::from_rgb(0.8, 0.6, 0.2)),
                SceneType::Noisy => ("📢", Color::from_rgb(0.8, 0.3, 0.3)),
            };
            
            row![
                text(icon).size(24),
                text(&self.adaptive_status.scene_name).size(16).style(color),
                text(format!("({:.0}%)", self.adaptive_status.confidence * 100.0))
                    .size(12)
                    .style(Color::from_rgb(0.5, 0.5, 0.5)),
            ]
            .spacing(8)
            .align_items(Alignment::Center)
        };
        
        // 校准状态
        let calibration_status = if self.adaptive_status.is_calibrating {
            row![
                text("🔄").size(14),
                text("正在校准环境...").size(14),
                // 进度条可以加在这里
            ]
            .spacing(8)
        } else {
            row![
                text("✓").size(14).style(Color::from_rgb(0.3, 0.7, 0.3)),
                text("环境已识别").size(14),
            ]
            .spacing(8)
        };
        
        // 过渡进度（仅在切换时显示）
        let transition_indicator = if self.adaptive_status.transition_progress < 1.0 
            && self.adaptive_status.transition_progress > 0.0 
        {
            row![
                text("切换中").size(12),
                text(format!("{:.0}%", self.adaptive_status.transition_progress * 100.0))
                    .size(12),
            ]
            .spacing(4)
        } else {
            row![]
        };
        
        // 特征显示（可折叠）
        let features_panel = if self.show_adaptive_features {
            let f = &self.adaptive_status.features;
            column![
                text("环境特征").size(14),
                row![
                    text("噪底:").size(12).width(80),
                    text(format!("{:.1} dB", f.noise_floor_db)).size(12),
                ],
                row![
                    text("混响:").size(12).width(80),
                    text(format!("{:.0}%", f.reverb_amount * 100.0)).size(12),
                ],
                row![
                    text("背景人声:").size(12).width(80),
                    text(format!("{:.0}%", f.voice_activity_ratio * 100.0)).size(12),
                ],
            ]
            .spacing(4)
        } else {
            column![]
        };
        
        // 控制按钮
        let controls = row![
            toggler(
                Some("自适应模式".to_string()),
                self.adaptive_enabled,
                Message::AdaptiveEnabledChanged,
            ),
            button("重新校准").on_press(Message::AdaptiveRecalibrate),
            button(if self.show_adaptive_features { "隐藏详情" } else { "显示详情" })
                .on_press(Message::ToggleAdaptiveFeatures),
        ]
        .spacing(12)
        .align_items(Alignment::Center);
        
        container(
            column![
                text("场景自适应").size(16),
                scene_indicator,
                calibration_status,
                transition_indicator,
                features_panel,
                controls,
            ]
            .spacing(10)
        )
        .padding(12)
        .style(iced::theme::Container::Box)
        .into()
    }
}
```

---

## 七、配置持久化

### 7.1 扩展配置结构

```json
{
  "version": 3,
  
  "_comment": "=== 自适应模式配置 ===",
  "adaptive": {
    "enabled": true,
    "auto_calibrate_on_start": true,
    "transition_duration_sec": 2.0,
    "min_confidence_for_switch": 0.7,
    "hold_duration_sec": 3.0,
    
    "user_overrides": {
      "df_atten_lim": null,
      "highpass_cutoff": null,
      "spatial_sensitivity": null,
      "eq_mix": null,
      "agc_target": null
    },
    
    "scene_customization": {
      "OpenOffice": {
        "df_atten_lim_offset": 0,
        "spatial_sensitivity_offset": 0
      },
      "MeetingRoom": {
        "df_atten_lim_offset": 0,
        "spatial_sensitivity_offset": 0
      }
    }
  }
}
```

---

## 八、测试场景

### 8.1 场景识别测试

| 测试 | 步骤 | 预期 |
|------|------|------|
| 启动校准 | 在安静环境启动 | 3秒内完成校准，识别为 Quiet/MeetingRoom |
| 进入办公区 | 从会议室走到工位 | 5-10秒内识别切换，参数渐变 |
| 返回会议室 | 从工位进入会议室 | 5-10秒内识别切换 |
| 突发噪声 | 短暂喧哗（<3秒） | 不触发场景切换 |
| 持续噪声 | 持续嘈杂（>5秒） | 切换到 Noisy 预设 |

### 8.2 音质测试

| 场景 | 测试 | 通过标准 |
|------|------|---------|
| 开放办公 | 自己清晰说话 | 对方听到清晰语音，背景人声抑制 |
| 开放办公 | 周围人大声说话 | 干扰明显减弱，不影响理解 |
| 会议室 | 正常说话 | 无明显处理痕迹，自然 |
| 会议室 | 有混响 | 混响减少，声音干净 |
| 切换过程 | 走动中说话 | 无明显音质跳变 |

---

## 九、文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `audio/adaptive/mod.rs` | 新增 | 自适应模块入口 |
| `audio/adaptive/scene_analyzer.rs` | 新增 | 场景分析器 |
| `audio/adaptive/scene_presets.rs` | 新增 | 场景参数预设 |
| `audio/adaptive/parameter_scheduler.rs` | 新增 | 参数调度器 |
| `audio/adaptive/reverb_estimator.rs` | 新增 | 混响估计 |
| `audio/adaptive/voice_detector.rs` | 新增 | 背景人声检测 |
| `audio/adaptive/spectrum_analyzer.rs` | 新增 | 频谱分析 |
| `audio/processors/noise_gate.rs` | 新增 | 噪声门 |
| `audio/processors/spatial_filter.rs` | 新增 | 空间滤波 |
| `audio/processors/voice_isolator.rs` | 新增 | 语音隔离 |
| `audio/processors/spectral_gate.rs` | 新增 | 频谱门限 |
| `capture.rs` | 修改 | 集成自适应处理 |
| `main.rs` | 修改 | UI 面板 |

---

## 十、实施建议

### 10.1 分阶段实施

| 阶段 | 内容 | 优先级 | 说明 |
|------|------|--------|------|
| **Phase 1** | SceneAnalyzer + 基础分类 | P0 | 实现场景识别核心 |
| **Phase 2** | NoiseGate + SpatialFilter | P0 | 基础增强模块 |
| **Phase 3** | ParameterScheduler + 平滑过渡 | P0 | 参数自动调整 |
| **Phase 4** | VoiceIsolator | P1 | 竞争人声抑制 |
| **Phase 5** | SpectralGate | P2 | 精细频域处理 |
| **Phase 6** | UI + 配置持久化 | P1 | 用户交互 |

### 10.2 关键指标

| 指标 | 目标 |
|------|------|
| 场景识别准确率 | > 90% |
| 场景切换延迟 | < 5 秒 |
| 参数过渡时间 | 2 秒 |
| 误切换率 | < 5% |
| CPU 新增开销 | < 15% |

---

**文档结束**

此方案确保用户在开放办公区与会议室之间切换时，系统自动识别环境并调整参数，无需手动干预。如有问题请联系音频架构组讨论。