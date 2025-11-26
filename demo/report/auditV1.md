# DeepFilterNet 实时音频处理系统优化方案

## 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 日期 | 2024年 |
| 状态 | 待评估 |
| 优先级 | P0-P2 |

---

## 一、现状问题总结

### 1.1 架构层面

| 问题 | 现状 | 影响 |
|------|------|------|
| 处理链硬编码 | 串行调用分散在 `capture.rs` 的 800+ 行 worker 函数中 | 难以维护、无法动态调整 |
| 多级限幅 | AGC、BusLimiter、FinalLimiter、EQ 内部共 4 处限幅 | 音质压缩、动态损失 |
| 延迟不可控 | 无延迟预算管理，BusLimiter 额外增加 6ms | 实时性无保障 |
| 线程同步 | 大量使用 `Ordering::Relaxed` | 潜在跨线程可见性问题 |

### 1.2 自适应性层面

| 问题 | 现状 | 影响 |
|------|------|------|
| 环境检测 | 3 级硬编码阈值分类 | 不同环境表现差异大 |
| 设备适配 | 无启动校准流程 | 不同麦克风灵敏度差异未处理 |
| 采样率 | 各模块独立假设，重采样配置固定 | 非标准采样率兼容性差 |

### 1.3 资源管理层面

| 问题 | 现状 | 影响 |
|------|------|------|
| 录音缓冲 | 固定 15 分钟预分配 | 内存占用过大 |
| 频谱渲染 | 每帧重新分配 `Vec<u8>` | GC 压力、卡顿风险 |
| 错误恢复 | 锁失败仅打日志 | 无降级策略 |

---

## 二、统一音频总线架构

### 2.1 目标

- 模块化处理链，支持动态插拔
- 统一增益管理，单点限幅
- 延迟预算可控
- 便于测试和调试

### 2.2 架构设计

```
┌────────────────────────────────────────────────────────────────────────┐
│                            AudioBus                                     │
│                                                                         │
│  ┌─────────┐   ┌─────────────┐   ┌─────────┐   ┌─────────────────────┐ │
│  │ InputStage │ → │ PreProcessors │ → │ CoreDF │ → │ PostProcessors      │ │
│  │ - 输入增益  │   │ - Highpass    │   │        │   │ - Transient        │ │
│  │ - 校准补偿  │   │ - NoiseGate   │   │        │   │ - Saturation       │ │
│  └─────────┘   └─────────────┘   └─────────┘   │ - DynamicEQ         │ │
│                                                  │ - AGC               │ │
│                                                  └─────────────────────┘ │
│                                                             ↓            │
│                                              ┌─────────────────────────┐ │
│                                              │ OutputStage (唯一限幅)  │ │
│                                              │ - True Peak Limiter     │ │
│                                              │ - Dither (可选)         │ │
│                                              └─────────────────────────┘ │
│                                                                         │
│  全局监控: peak_hold, gain_reduction, latency_ms, cpu_load             │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.3 核心接口定义

```rust
// ====================
// 文件: audio/bus/mod.rs
// ====================

/// 音频处理器统一接口
pub trait AudioProcessor: Send {
    /// 处理器名称，用于调试
    fn name(&self) -> &'static str;
    
    /// 处理延迟（采样点数）
    fn latency_samples(&self) -> usize { 0 }
    
    /// 处理音频块（原地修改）
    fn process(&mut self, samples: &mut [f32], ctx: &ProcessContext);
    
    /// 重置内部状态
    fn reset(&mut self);
    
    /// 是否启用
    fn is_enabled(&self) -> bool { true }
}

/// 处理上下文，传递全局状态
pub struct ProcessContext {
    pub sample_rate: f32,
    pub block_size: usize,
    pub timestamp_ms: f64,
    pub input_level_db: f32,      // 输入电平，供各模块参考
    pub voice_activity: f32,      // VAD 概率 0-1
    pub env_class: EnvironmentClass,
}

/// 环境分类（扩展为 5 级）
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnvironmentClass {
    Silent,      // < -60 dB，几乎无信号
    Quiet,       // -60 ~ -45 dB，安静房间
    Moderate,    // -45 ~ -35 dB，正常办公
    Noisy,       // -35 ~ -25 dB，嘈杂环境
    Extreme,     // > -25 dB，极端噪声
}
```

### 2.4 AudioBus 实现

```rust
// ====================
// 文件: audio/bus/audio_bus.rs
// ====================

pub struct AudioBus {
    sample_rate: f32,
    
    // 处理模块
    input_stage: InputStage,
    pre_processors: Vec<Box<dyn AudioProcessor>>,
    core_processor: Option<Box<dyn AudioProcessor>>,
    post_processors: Vec<Box<dyn AudioProcessor>>,
    output_stage: OutputStage,
    
    // 状态监控
    metrics: BusMetrics,
    
    // 延迟预算
    latency_budget_samples: usize,
}

pub struct BusMetrics {
    pub total_latency_ms: f32,
    pub peak_input_db: f32,
    pub peak_output_db: f32,
    pub gain_reduction_db: f32,
    pub cpu_load_percent: f32,
    pub voice_activity: f32,
}

impl AudioBus {
    pub fn new(sample_rate: f32, latency_budget_ms: f32) -> Self {
        let latency_budget_samples = (sample_rate * latency_budget_ms / 1000.0) as usize;
        
        Self {
            sample_rate,
            input_stage: InputStage::new(sample_rate),
            pre_processors: Vec::new(),
            core_processor: None,
            post_processors: Vec::new(),
            output_stage: OutputStage::new(sample_rate),
            metrics: BusMetrics::default(),
            latency_budget_samples,
        }
    }
    
    /// 添加前处理器
    pub fn add_pre_processor(&mut self, processor: Box<dyn AudioProcessor>) {
        self.pre_processors.push(processor);
        self.validate_latency_budget();
    }
    
    /// 添加后处理器
    pub fn add_post_processor(&mut self, processor: Box<dyn AudioProcessor>) {
        self.post_processors.push(processor);
        self.validate_latency_budget();
    }
    
    /// 设置核心处理器（DeepFilter）
    pub fn set_core_processor(&mut self, processor: Box<dyn AudioProcessor>) {
        self.core_processor = Some(processor);
        self.validate_latency_budget();
    }
    
    /// 主处理函数
    pub fn process(&mut self, samples: &mut [f32]) -> &BusMetrics {
        let start_time = std::time::Instant::now();
        
        // 1. 输入电平检测
        self.metrics.peak_input_db = calculate_peak_db(samples);
        
        // 2. 构建处理上下文
        let ctx = ProcessContext {
            sample_rate: self.sample_rate,
            block_size: samples.len(),
            timestamp_ms: 0.0, // TODO: 实际时间戳
            input_level_db: self.metrics.peak_input_db,
            voice_activity: self.input_stage.voice_activity(),
            env_class: self.input_stage.environment_class(),
        };
        
        // 3. 输入阶段
        self.input_stage.process(samples, &ctx);
        
        // 4. 前处理链
        for processor in &mut self.pre_processors {
            if processor.is_enabled() {
                processor.process(samples, &ctx);
            }
        }
        
        // 5. 核心处理（DeepFilter）
        if let Some(ref mut core) = self.core_processor {
            if core.is_enabled() {
                core.process(samples, &ctx);
            }
        }
        
        // 6. 后处理链
        for processor in &mut self.post_processors {
            if processor.is_enabled() {
                processor.process(samples, &ctx);
            }
        }
        
        // 7. 输出阶段（唯一限幅点）
        self.output_stage.process(samples, &ctx);
        
        // 8. 更新指标
        self.metrics.peak_output_db = calculate_peak_db(samples);
        self.metrics.gain_reduction_db = self.output_stage.gain_reduction_db();
        self.metrics.cpu_load_percent = calculate_cpu_load(start_time, samples.len(), self.sample_rate);
        self.metrics.voice_activity = ctx.voice_activity;
        
        &self.metrics
    }
    
    /// 验证延迟预算
    fn validate_latency_budget(&self) {
        let total = self.calculate_total_latency();
        if total > self.latency_budget_samples {
            log::warn!(
                "延迟预算超出: {} samples > {} samples ({:.1}ms > {:.1}ms)",
                total, self.latency_budget_samples,
                total as f32 / self.sample_rate * 1000.0,
                self.latency_budget_samples as f32 / self.sample_rate * 1000.0
            );
        }
    }
    
    fn calculate_total_latency(&self) -> usize {
        let mut total = self.input_stage.latency_samples();
        for p in &self.pre_processors {
            total += p.latency_samples();
        }
        if let Some(ref core) = self.core_processor {
            total += core.latency_samples();
        }
        for p in &self.post_processors {
            total += p.latency_samples();
        }
        total += self.output_stage.latency_samples();
        total
    }
}

fn calculate_peak_db(samples: &[f32]) -> f32 {
    let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    20.0 * peak.max(1e-10).log10()
}

fn calculate_cpu_load(start: std::time::Instant, block_size: usize, sample_rate: f32) -> f32 {
    let elapsed = start.elapsed().as_secs_f32();
    let block_duration = block_size as f32 / sample_rate;
    (elapsed / block_duration * 100.0).min(100.0)
}
```

### 2.5 统一输出阶段（单点限幅）

```rust
// ====================
// 文件: audio/bus/output_stage.rs
// ====================

/// 输出阶段 - 系统唯一的限幅点
pub struct OutputStage {
    sample_rate: f32,
    
    // True Peak 限幅器
    ceiling_db: f32,
    lookahead_samples: usize,
    lookahead_buffer: Vec<f32>,
    lookahead_pos: usize,
    
    // 增益平滑
    current_gain: f32,
    attack_coef: f32,
    release_coef: f32,
    
    // 监控
    gain_reduction_db: f32,
}

impl OutputStage {
    pub fn new(sample_rate: f32) -> Self {
        let lookahead_ms = 1.5;  // 减少到 1.5ms，平衡延迟和质量
        let lookahead_samples = (sample_rate * lookahead_ms / 1000.0) as usize;
        
        // 快攻慢放
        let attack_ms = 0.1;
        let release_ms = 100.0;
        
        Self {
            sample_rate,
            ceiling_db: -0.5,  // -0.5 dBFS，留余量
            lookahead_samples,
            lookahead_buffer: vec![0.0; lookahead_samples],
            lookahead_pos: 0,
            current_gain: 1.0,
            attack_coef: (-1.0 / (attack_ms * sample_rate / 1000.0)).exp(),
            release_coef: (-1.0 / (release_ms * sample_rate / 1000.0)).exp(),
            gain_reduction_db: 0.0,
        }
    }
    
    pub fn gain_reduction_db(&self) -> f32 {
        self.gain_reduction_db
    }
    
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }
}

impl AudioProcessor for OutputStage {
    fn name(&self) -> &'static str { "OutputStage" }
    
    fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }
    
    fn process(&mut self, samples: &mut [f32], _ctx: &ProcessContext) {
        let ceiling_linear = db_to_linear(self.ceiling_db);
        
        for sample in samples.iter_mut() {
            // 写入 lookahead 缓冲
            let delayed = self.lookahead_buffer[self.lookahead_pos];
            self.lookahead_buffer[self.lookahead_pos] = *sample;
            self.lookahead_pos = (self.lookahead_pos + 1) % self.lookahead_samples;
            
            // 计算 lookahead 窗口内的峰值
            let peak = self.lookahead_buffer.iter()
                .fold(0.0f32, |acc, &s| acc.max(s.abs()));
            
            // 计算目标增益
            let target_gain = if peak > ceiling_linear {
                ceiling_linear / peak
            } else {
                1.0
            };
            
            // 平滑增益变化
            let coef = if target_gain < self.current_gain {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.current_gain = coef * self.current_gain + (1.0 - coef) * target_gain;
            
            // 应用增益到延迟后的信号
            *sample = delayed * self.current_gain;
        }
        
        // 更新监控指标
        self.gain_reduction_db = -20.0 * self.current_gain.log10();
    }
    
    fn reset(&mut self) {
        self.lookahead_buffer.fill(0.0);
        self.lookahead_pos = 0;
        self.current_gain = 1.0;
        self.gain_reduction_db = 0.0;
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}
```

---

## 三、自适应校准系统

### 3.1 目标

- 启动时自动检测环境基线
- 适配不同麦克风灵敏度
- 动态调整处理参数

### 3.2 校准流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      启动校准流程 (3秒)                          │
│                                                                  │
│  T=0s        T=1s           T=2s           T=3s                 │
│  ├───────────┼──────────────┼──────────────┤                    │
│  │ 静默检测   │ 噪声特征分析  │ 参数计算     │ → 正常处理        │
│  │           │              │              │                    │
│  │ - 底噪电平 │ - 频谱形状   │ - DF阈值     │                    │
│  │ - 峰值统计 │ - 平坦度     │ - 高通截止   │                    │
│  │           │ - 频谱质心   │ - AGC目标    │                    │
│  │           │ - 调制特征   │ - EQ预设     │                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 实现代码

```rust
// ====================
// 文件: audio/calibration.rs
// ====================

use std::collections::VecDeque;

/// 校准状态机
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationState {
    NotStarted,
    CollectingSilence,    // 收集静默数据
    AnalyzingNoise,       // 分析噪声特征
    ComputingParams,      // 计算参数
    Completed,
    Failed(CalibrationError),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationError {
    TooMuchNoise,         // 环境太吵，无法校准
    SignalDetected,       // 检测到语音信号
    Timeout,
}

/// 校准结果
#[derive(Clone, Debug)]
pub struct CalibrationResult {
    pub noise_floor_db: f32,
    pub peak_noise_db: f32,
    pub spectral_flatness: f32,
    pub spectral_centroid_hz: f32,
    pub recommended_params: RecommendedParams,
    pub confidence: f32,  // 0-1，校准置信度
}

#[derive(Clone, Debug)]
pub struct RecommendedParams {
    pub df_atten_lim: f32,
    pub df_min_thresh: f32,
    pub df_mix: f32,
    pub highpass_cutoff: f32,
    pub agc_target_db: f32,
    pub agc_max_gain: f32,
    pub eq_preset: EqPresetKind,
    pub eq_mix: f32,
}

/// 自适应校准器
pub struct AdaptiveCalibrator {
    sample_rate: f32,
    state: CalibrationState,
    
    // 数据收集
    samples_collected: usize,
    target_samples: usize,
    
    // 统计数据
    energy_history: VecDeque<f32>,
    spectrum_accumulator: Vec<f32>,
    spectrum_count: usize,
    peak_history: VecDeque<f32>,
    
    // FFT
    fft_size: usize,
    fft_buffer: Vec<f32>,
    
    // 结果
    result: Option<CalibrationResult>,
}

impl AdaptiveCalibrator {
    pub fn new(sample_rate: f32) -> Self {
        let calibration_duration_sec = 3.0;
        let target_samples = (sample_rate * calibration_duration_sec) as usize;
        let fft_size = 2048;
        
        Self {
            sample_rate,
            state: CalibrationState::NotStarted,
            samples_collected: 0,
            target_samples,
            energy_history: VecDeque::with_capacity(1000),
            spectrum_accumulator: vec![0.0; fft_size / 2],
            spectrum_count: 0,
            peak_history: VecDeque::with_capacity(100),
            fft_size,
            fft_buffer: vec![0.0; fft_size],
            result: None,
        }
    }
    
    /// 开始校准
    pub fn start(&mut self) {
        self.state = CalibrationState::CollectingSilence;
        self.samples_collected = 0;
        self.energy_history.clear();
        self.spectrum_accumulator.fill(0.0);
        self.spectrum_count = 0;
        self.peak_history.clear();
        self.result = None;
    }
    
    /// 处理音频块，返回是否完成
    pub fn process(&mut self, samples: &[f32]) -> bool {
        match self.state {
            CalibrationState::NotStarted => false,
            CalibrationState::Completed | CalibrationState::Failed(_) => true,
            _ => {
                self.collect_data(samples);
                self.samples_collected += samples.len();
                
                // 检查是否检测到语音
                if self.detect_voice_activity(samples) {
                    self.state = CalibrationState::Failed(CalibrationError::SignalDetected);
                    return true;
                }
                
                // 检查进度
                let progress = self.samples_collected as f32 / self.target_samples as f32;
                
                if progress >= 0.33 && self.state == CalibrationState::CollectingSilence {
                    self.state = CalibrationState::AnalyzingNoise;
                } else if progress >= 0.66 && self.state == CalibrationState::AnalyzingNoise {
                    self.state = CalibrationState::ComputingParams;
                } else if progress >= 1.0 {
                    self.compute_result();
                    self.state = CalibrationState::Completed;
                    return true;
                }
                
                false
            }
        }
    }
    
    /// 获取校准结果
    pub fn result(&self) -> Option<&CalibrationResult> {
        self.result.as_ref()
    }
    
    /// 获取当前状态
    pub fn state(&self) -> CalibrationState {
        self.state
    }
    
    /// 获取进度 (0-1)
    pub fn progress(&self) -> f32 {
        (self.samples_collected as f32 / self.target_samples as f32).min(1.0)
    }
    
    fn collect_data(&mut self, samples: &[f32]) {
        // 计算块能量
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let energy_db = 10.0 * energy.max(1e-10).log10();
        self.energy_history.push_back(energy_db);
        
        // 记录峰值
        let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        self.peak_history.push_back(peak);
        
        // 累积频谱（简化版，实际应用 FFT）
        // TODO: 使用 rustfft 进行真实频谱分析
        self.spectrum_count += 1;
    }
    
    fn detect_voice_activity(&self, samples: &[f32]) -> bool {
        // 简化的 VAD：检测能量突变
        let current_energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let current_db = 10.0 * current_energy.max(1e-10).log10();
        
        if let Some(&baseline) = self.energy_history.front() {
            // 如果当前能量比基线高 15dB，可能有语音
            if current_db > baseline + 15.0 {
                return true;
            }
        }
        
        false
    }
    
    fn compute_result(&mut self) {
        // 计算统计数据
        let noise_floor_db = self.compute_percentile(&self.energy_history, 0.1);
        let peak_noise_db = self.compute_percentile(&self.energy_history, 0.95);
        let noise_variance = self.compute_variance(&self.energy_history);
        
        // 频谱平坦度（简化计算）
        let spectral_flatness = self.compute_spectral_flatness();
        let spectral_centroid_hz = self.compute_spectral_centroid();
        
        // 根据分析结果计算推荐参数
        let recommended_params = self.compute_recommended_params(
            noise_floor_db,
            peak_noise_db,
            spectral_flatness,
            spectral_centroid_hz,
        );
        
        // 计算置信度
        let confidence = self.compute_confidence(noise_variance);
        
        self.result = Some(CalibrationResult {
            noise_floor_db,
            peak_noise_db,
            spectral_flatness,
            spectral_centroid_hz,
            recommended_params,
            confidence,
        });
    }
    
    fn compute_recommended_params(
        &self,
        noise_floor_db: f32,
        peak_noise_db: f32,
        spectral_flatness: f32,
        spectral_centroid_hz: f32,
    ) -> RecommendedParams {
        // 根据噪声电平选择降噪强度
        let (df_atten_lim, df_mix) = match noise_floor_db {
            x if x < -60.0 => (25.0, 0.85),   // 非常安静
            x if x < -50.0 => (35.0, 0.90),   // 安静
            x if x < -40.0 => (45.0, 0.95),   // 中等噪声
            x if x < -30.0 => (55.0, 1.00),   // 嘈杂
            _ => (65.0, 1.00),                 // 极端噪声
        };
        
        // 根据噪声频谱特征选择高通截止
        let highpass_cutoff = if spectral_centroid_hz < 500.0 {
            // 低频噪声为主（空调、风扇）
            80.0
        } else if spectral_flatness > 0.5 {
            // 宽带噪声
            60.0
        } else {
            // 正常情况
            50.0
        };
        
        // AGC 参数
        let agc_target_db = if noise_floor_db < -50.0 { -14.0 } else { -18.0 };
        let agc_max_gain = if noise_floor_db < -55.0 { 15.0 } else { 10.0 };
        
        // EQ 预设选择
        let eq_preset = if noise_floor_db > -35.0 {
            EqPresetKind::Meeting  // 嘈杂环境用会议预设
        } else if spectral_flatness > 0.4 {
            EqPresetKind::OpenOffice
        } else {
            EqPresetKind::Broadcast
        };
        
        let eq_mix = (0.5 + (noise_floor_db + 50.0) / 100.0).clamp(0.4, 0.8);
        
        RecommendedParams {
            df_atten_lim,
            df_min_thresh: noise_floor_db - 10.0,
            df_mix,
            highpass_cutoff,
            agc_target_db,
            agc_max_gain,
            eq_preset,
            eq_mix,
        }
    }
    
    fn compute_percentile(&self, data: &VecDeque<f32>, p: f32) -> f32 {
        if data.is_empty() {
            return -60.0;
        }
        let mut sorted: Vec<f32> = data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f32 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
    
    fn compute_variance(&self, data: &VecDeque<f32>) -> f32 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }
    
    fn compute_spectral_flatness(&self) -> f32 {
        // 简化实现，实际应基于 FFT 结果
        // 返回 0-1，越接近 1 表示越接近白噪声
        0.3  // 占位值
    }
    
    fn compute_spectral_centroid(&self) -> f32 {
        // 简化实现
        1000.0  // 占位值，单位 Hz
    }
    
    fn compute_confidence(&self, noise_variance: f32) -> f32 {
        // 噪声越稳定，置信度越高
        let stability_score = 1.0 / (1.0 + noise_variance / 5.0);
        
        // 数据量足够
        let data_score = (self.energy_history.len() as f32 / 500.0).min(1.0);
        
        (stability_score * data_score).clamp(0.0, 1.0)
    }
}

// EqPresetKind 引用
use super::eq::EqPresetKind;
```

### 3.4 运行时自适应

```rust
// ====================
// 文件: audio/adaptive.rs
// ====================

/// 运行时环境监测器
pub struct EnvironmentMonitor {
    sample_rate: f32,
    
    // 滑动窗口统计
    energy_window: SlidingWindow,
    flatness_window: SlidingWindow,
    centroid_window: SlidingWindow,
    
    // 当前分类
    current_class: EnvironmentClass,
    class_hold_frames: usize,
    class_change_threshold: usize,
    
    // VAD
    vad_state: VadState,
}

struct SlidingWindow {
    buffer: VecDeque<f32>,
    capacity: usize,
    sum: f32,
}

impl SlidingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }
    
    fn push(&mut self, value: f32) {
        if self.buffer.len() >= self.capacity {
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
            }
        }
        self.buffer.push_back(value);
        self.sum += value;
    }
    
    fn mean(&self) -> f32 {
        if self.buffer.is_empty() {
            0.0
        } else {
            self.sum / self.buffer.len() as f32
        }
    }
}

struct VadState {
    speech_probability: f32,
    hangover_frames: usize,
    energy_threshold: f32,
}

impl EnvironmentMonitor {
    pub fn new(sample_rate: f32) -> Self {
        let window_sec = 2.0;
        let window_size = (sample_rate * window_sec / 480.0) as usize; // 假设 480 采样/帧
        
        Self {
            sample_rate,
            energy_window: SlidingWindow::new(window_size),
            flatness_window: SlidingWindow::new(window_size),
            centroid_window: SlidingWindow::new(window_size),
            current_class: EnvironmentClass::Quiet,
            class_hold_frames: 0,
            class_change_threshold: 30,  // 约 0.3 秒
            vad_state: VadState {
                speech_probability: 0.0,
                hangover_frames: 0,
                energy_threshold: -45.0,
            },
        }
    }
    
    /// 更新环境监测，返回当前分类
    pub fn update(&mut self, samples: &[f32]) -> EnvironmentClass {
        // 计算特征
        let energy_db = self.compute_energy_db(samples);
        let flatness = self.compute_flatness(samples);
        let centroid = self.compute_centroid(samples);
        
        // 仅在非语音期间更新环境统计
        if !self.is_speech_active(energy_db) {
            self.energy_window.push(energy_db);
            self.flatness_window.push(flatness);
            self.centroid_window.push(centroid);
        }
        
        // 分类决策
        let target_class = self.classify(
            self.energy_window.mean(),
            self.flatness_window.mean(),
            self.centroid_window.mean(),
        );
        
        // 滞后切换，避免频繁变化
        if target_class != self.current_class {
            self.class_hold_frames += 1;
            if self.class_hold_frames >= self.class_change_threshold {
                self.current_class = target_class;
                self.class_hold_frames = 0;
                log::info!("环境分类切换: {:?}", self.current_class);
            }
        } else {
            self.class_hold_frames = 0;
        }
        
        self.current_class
    }
    
    pub fn voice_activity(&self) -> f32 {
        self.vad_state.speech_probability
    }
    
    pub fn environment_class(&self) -> EnvironmentClass {
        self.current_class
    }
    
    fn compute_energy_db(&self, samples: &[f32]) -> f32 {
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32;
        10.0 * energy.max(1e-10).log10()
    }
    
    fn compute_flatness(&self, _samples: &[f32]) -> f32 {
        // 简化实现，实际应计算频谱平坦度
        0.3
    }
    
    fn compute_centroid(&self, _samples: &[f32]) -> f32 {
        // 简化实现
        0.4
    }
    
    fn is_speech_active(&mut self, energy_db: f32) -> bool {
        // 简化 VAD
        if energy_db > self.vad_state.energy_threshold + 10.0 {
            self.vad_state.speech_probability = 0.9;
            self.vad_state.hangover_frames = 20;
        } else if self.vad_state.hangover_frames > 0 {
            self.vad_state.hangover_frames -= 1;
            self.vad_state.speech_probability = 0.5;
        } else {
            self.vad_state.speech_probability = 0.1;
        }
        
        self.vad_state.speech_probability > 0.5
    }
    
    fn classify(&self, energy_db: f32, flatness: f32, centroid: f32) -> EnvironmentClass {
        // 五级分类，带模糊边界
        match energy_db {
            x if x < -55.0 => EnvironmentClass::Silent,
            x if x < -45.0 => {
                // 考虑频谱特征进行细分
                if flatness > 0.5 {
                    EnvironmentClass::Moderate  // 宽带噪声
                } else {
                    EnvironmentClass::Quiet
                }
            }
            x if x < -35.0 => EnvironmentClass::Moderate,
            x if x < -25.0 => EnvironmentClass::Noisy,
            _ => EnvironmentClass::Extreme,
        }
    }
}
```

---

## 四、延迟预算管理

### 4.1 目标延迟

| 场景 | 目标延迟 | 说明 |
|------|---------|------|
| 实时通话 | ≤ 20ms | 交互感知阈值 |
| 直播推流 | ≤ 40ms | 可接受范围 |
| 后期处理 | 无限制 | 质量优先 |

### 4.2 各模块延迟分配

```
┌──────────────────────────────────────────────────────────┐
│              20ms 延迟预算分配 (48kHz)                    │
│                                                          │
│  模块              延迟(ms)    采样点    占比            │
│  ─────────────────────────────────────────────────       │
│  InputStage        0.0         0        0%               │
│  Highpass          0.0         0        0%               │
│  DeepFilter        10.0        480      50%    ← 固定    │
│  Transient         0.0         0        0%               │
│  Saturation        0.0         0        0%               │
│  DynamicEQ         0.0         0        0%               │
│  AGC               0.0         0        0%               │
│  OutputLimiter     1.5         72       7.5%             │
│  ─────────────────────────────────────────────────       │
│  缓冲/调度余量      8.5         408      42.5%           │
│  ─────────────────────────────────────────────────       │
│  总计              20.0        960      100%             │
└──────────────────────────────────────────────────────────┘
```

### 4.3 实现代码

```rust
// ====================
// 文件: audio/latency.rs
// ====================

/// 延迟预算管理器
pub struct LatencyBudget {
    sample_rate: f32,
    total_budget_ms: f32,
    allocations: Vec<LatencyAllocation>,
}

struct LatencyAllocation {
    name: String,
    latency_ms: f32,
    is_fixed: bool,
}

impl LatencyBudget {
    pub fn new(sample_rate: f32, total_budget_ms: f32) -> Self {
        Self {
            sample_rate,
            total_budget_ms,
            allocations: Vec::new(),
        }
    }
    
    /// 注册模块延迟
    pub fn register(&mut self, name: &str, latency_ms: f32, is_fixed: bool) -> Result<(), String> {
        let current_total: f32 = self.allocations.iter().map(|a| a.latency_ms).sum();
        
        if current_total + latency_ms > self.total_budget_ms {
            return Err(format!(
                "延迟预算超出: 当前 {:.1}ms + 新增 {:.1}ms > 预算 {:.1}ms",
                current_total, latency_ms, self.total_budget_ms
            ));
        }
        
        self.allocations.push(LatencyAllocation {
            name: name.to_string(),
            latency_ms,
            is_fixed,
        });
        
        Ok(())
    }
    
    /// 获取剩余预算
    pub fn remaining_ms(&self) -> f32 {
        let used: f32 = self.allocations.iter().map(|a| a.latency_ms).sum();
        self.total_budget_ms - used
    }
    
    /// 获取总使用延迟
    pub fn total_used_ms(&self) -> f32 {
        self.allocations.iter().map(|a| a.latency_ms).sum()
    }
    
    /// 打印报告
    pub fn report(&self) -> String {
        let mut lines = vec![
            format!("延迟预算报告 (采样率: {}Hz)", self.sample_rate),
            "─".repeat(50),
        ];
        
        for alloc in &self.allocations {
            let samples = (alloc.latency_ms * self.sample_rate / 1000.0) as usize;
            let fixed_mark = if alloc.is_fixed { "[固定]" } else { "" };
            lines.push(format!(
                "  {:20} {:>6.2}ms {:>6} samples {}",
                alloc.name, alloc.latency_ms, samples, fixed_mark
            ));
        }
        
        lines.push("─".repeat(50));
        lines.push(format!(
            "  总计: {:.2}ms / {:.2}ms (剩余: {:.2}ms)",
            self.total_used_ms(),
            self.total_budget_ms,
            self.remaining_ms()
        ));
        
        lines.join("\n")
    }
}
```

---

## 五、线程同步优化

### 5.1 问题分析

当前代码大量使用 `Ordering::Relaxed`，可能导致：
- 跨线程写入可见性延迟
- 在某些 CPU 架构上出现数据竞争

### 5.2 修复方案

```rust
// ====================
// 修改: capture.rs
// ====================

// 修改前
should_stop.load(Ordering::Relaxed)
should_stop.store(true, Ordering::Relaxed)

// 修改后 - 关键同步点使用正确的内存序
// 停止标志：写入用 Release，读取用 Acquire
should_stop.store(true, Ordering::Release)    // 写入端
should_stop.load(Ordering::Acquire)           // 读取端

// 初始化标志
has_init.store(true, Ordering::Release)       // worker 设置
has_init.load(Ordering::Acquire)              // main 等待

// 非关键路径可以保持 Relaxed
// 例如：监控计数器、统计数据
```

### 5.3 通道优化

```rust
// 当前：使用 unbounded 通道可能导致内存增长
let (s_lsnr, r_lsnr) = unbounded();

// 改进：使用有界通道 + 背压策略
use crossbeam_channel::bounded;

const CHANNEL_CAPACITY: usize = 64;

let (s_lsnr, r_lsnr) = bounded(CHANNEL_CAPACITY);

// 发送端：非阻塞尝试，满则丢弃
if s_lsnr.try_send(value).is_err() {
    log::trace!("LSNR 通道已满，丢弃数据");
}
```

---

## 六、资源管理优化

### 6.1 录音缓冲优化

```rust
// ====================
// 文件: audio/recording.rs
// ====================

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

/// 流式录音器 - 边录边写，避免内存爆炸
pub struct StreamingRecorder {
    sample_rate: u32,
    temp_dir: PathBuf,
    
    // 各轨道写入器
    noisy_writer: Option<WavWriter>,
    denoised_writer: Option<WavWriter>,
    processed_writer: Option<WavWriter>,
    
    // 统计
    samples_written: usize,
}

struct WavWriter {
    writer: hound::WavWriter<BufWriter<File>>,
    path: PathBuf,
}

impl StreamingRecorder {
    pub fn new(sample_rate: u32, output_dir: &Path) -> Result<Self, String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let temp_dir = output_dir.join(&timestamp);
        std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
        
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        
        Ok(Self {
            sample_rate,
            temp_dir: temp_dir.clone(),
            noisy_writer: Some(Self::create_writer(&temp_dir.join("raw.wav"), spec)?),
            denoised_writer: Some(Self::create_writer(&temp_dir.join("nc.wav"), spec)?),
            processed_writer: Some(Self::create_writer(&temp_dir.join("eq.wav"), spec)?),
            samples_written: 0,
        })
    }
    
    fn create_writer(path: &Path, spec: hound::WavSpec) -> Result<WavWriter, String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let buf_writer = BufWriter::with_capacity(64 * 1024, file);  // 64KB 缓冲
        let writer = hound::WavWriter::new(buf_writer, spec).map_err(|e| e.to_string())?;
        Ok(WavWriter {
            writer,
            path: path.to_path_buf(),
        })
    }
    
    /// 写入原始音频
    pub fn write_noisy(&mut self, samples: &[f32]) {
        if let Some(ref mut w) = self.noisy_writer {
            for &sample in samples {
                let _ = w.writer.write_sample(sample.clamp(-1.0, 1.0));
            }
        }
    }
    
    /// 写入降噪后音频
    pub fn write_denoised(&mut self, samples: &[f32]) {
        if let Some(ref mut w) = self.denoised_writer {
            for &sample in samples {
                let _ = w.writer.write_sample(sample.clamp(-1.0, 1.0));
            }
        }
    }
    
    /// 写入最终处理音频
    pub fn write_processed(&mut self, samples: &[f32]) {
        if let Some(ref mut w) = self.processed_writer {
            for &sample in samples {
                let _ = w.writer.write_sample(sample.clamp(-1.0, 1.0));
            }
        }
        self.samples_written += samples.len();
    }
    
    /// 完成录音，返回文件路径
    pub fn finalize(mut self) -> Result<(PathBuf, PathBuf, PathBuf), String> {
        let noisy_path = self.noisy_writer.take()
            .map(|w| { let _ = w.writer.finalize(); w.path })
            .ok_or("noisy writer missing")?;
        
        let denoised_path = self.denoised_writer.take()
            .map(|w| { let _ = w.writer.finalize(); w.path })
            .ok_or("denoised writer missing")?;
        
        let processed_path = self.processed_writer.take()
            .map(|w| { let _ = w.writer.finalize(); w.path })
            .ok_or("processed writer missing")?;
        
        Ok((noisy_path, denoised_path, processed_path))
    }
    
    /// 录音时长（秒）
    pub fn duration_sec(&self) -> f32 {
        self.samples_written as f32 / self.sample_rate as f32
    }
}
```

### 6.2 频谱图双缓冲

```rust
// ====================
// 修改: main.rs 中的 SpecImage
// ====================

pub struct SpecImage {
    // 双缓冲
    front_buffer: Vec<u8>,
    back_buffer: Vec<u8>,
    buffer_ready: AtomicBool,
    
    // 原有字段
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    // ...
}

impl SpecImage {
    pub fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        let buffer_size = (n_frames * n_freqs * 4) as usize;
        Self {
            front_buffer: vec![0; buffer_size],
            back_buffer: vec![0; buffer_size],
            buffer_ready: AtomicBool::new(false),
            im: RgbaImage::new(n_frames, n_freqs),
            n_frames,
            n_freqs,
            vmin,
            vmax,
            write_pos: 0,
            frames_written: 0,
        }
    }
    
    /// 更新后台缓冲
    pub fn update_back_buffer(&mut self) {
        // 复用 back_buffer，避免重新分配
        self.render_to_buffer(&mut self.back_buffer);
        self.buffer_ready.store(true, Ordering::Release);
    }
    
    /// 交换缓冲
    pub fn swap_buffers(&mut self) {
        if self.buffer_ready.load(Ordering::Acquire) {
            std::mem::swap(&mut self.front_buffer, &mut self.back_buffer);
            self.buffer_ready.store(false, Ordering::Release);
        }
    }
    
    /// 获取显示用缓冲（无分配）
    pub fn display_buffer(&self) -> &[u8] {
        &self.front_buffer
    }
    
    fn render_to_buffer(&self, buffer: &mut [u8]) {
        // ... 渲染逻辑，复用 buffer
    }
}
```

---

## 七、模块迁移计划

### 7.1 文件结构重组

```
demo/src/
├── main.rs                 # UI 入口
├── capture.rs              # 简化为设备管理
│
├── audio/
│   ├── mod.rs
│   ├── bus/               # 新增：统一音频总线
│   │   ├── mod.rs
│   │   ├── audio_bus.rs
│   │   ├── input_stage.rs
│   │   └── output_stage.rs
│   │
│   ├── processors/        # 重构：处理器模块
│   │   ├── mod.rs
│   │   ├── highpass.rs
│   │   ├── transient.rs
│   │   ├── saturation.rs
│   │   ├── agc.rs
│   │   └── deep_filter.rs  # DF 封装
│   │
│   ├── eq/                # 保持
│   │
│   ├── calibration.rs     # 新增：校准系统
│   ├── adaptive.rs        # 新增：自适应监测
│   ├── latency.rs         # 新增：延迟管理
│   └── recording.rs       # 新增：流式录音
│
└── ui/
    ├── mod.rs
    └── tooltips.rs
```

### 7.2 迁移步骤

| 阶段 | 内容 | 预计工时 | 风险 |
|------|------|---------|------|
| Phase 1 | 定义 AudioProcessor trait，包装现有模块 | 2天 | 低 |
| Phase 2 | 实现 AudioBus，替换硬编码处理链 | 3天 | 中 |
| Phase 3 | 统一 OutputStage，移除多余限幅 | 2天 | 中 |
| Phase 4 | 实现校准系统 | 3天 | 低 |
| Phase 5 | 实现运行时自适应 | 2天 | 低 |
| Phase 6 | 流式录音改造 | 1天 | 低 |
| Phase 7 | 线程同步修复 | 1天 | 低 |
| Phase 8 | 测试与调优 | 3天 | - |

**总计预估：17 个工作日**

---

## 八、测试验证计划

### 8.1 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_output_limiter_ceiling() {
        let mut limiter = OutputStage::new(48000.0);
        let mut samples = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        
        let ctx = ProcessContext::default();
        limiter.process(&mut samples, &ctx);
        
        // 验证不超过 ceiling
        for &s in &samples {
            assert!(s.abs() <= 0.95, "Sample {} exceeds ceiling", s);
        }
    }
    
    #[test]
    fn test_calibration_quiet_room() {
        let mut calibrator = AdaptiveCalibrator::new(48000.0);
        calibrator.start();
        
        // 模拟安静环境噪声
        let quiet_noise: Vec<f32> = (0..48000)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.001)
            .collect();
        
        for chunk in quiet_noise.chunks(480) {
            if calibrator.process(chunk) {
                break;
            }
        }
        
        let result = calibrator.result().unwrap();
        assert!(result.noise_floor_db < -50.0);
        assert!(result.recommended_params.df_atten_lim < 40.0);
    }
    
    #[test]
    fn test_latency_budget() {
        let mut budget = LatencyBudget::new(48000.0, 20.0);
        
        budget.register("DeepFilter", 10.0, true).unwrap();
        budget.register("OutputLimiter", 1.5, true).unwrap();
        
        assert_eq!(budget.remaining_ms(), 8.5);
        
        // 超出预算应失败
        let result = budget.register("Extra", 10.0, false);
        assert!(result.is_err());
    }
}
```

### 8.2 集成测试场景

| 场景 | 测试内容 | 通过标准 |
|------|---------|---------|
| 安静房间 | 白噪声 -60dB | 降噪后底噪 < -70dB |
| 办公环境 | 键盘 + 空调 | 可懂度 > 95% |
| 嘈杂环境 | 多人交谈背景 | 主说话人清晰 |
| 极端测试 | 1kHz 正弦波 0dBFS | 无削波失真 |
| 长时运行 | 持续 1 小时 | 内存稳定，无泄漏 |
| 设备切换 | 热插拔麦克风 | 自动恢复 |

---

## 九、风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| DeepFilter 延迟不可控 | 高 | 高 | 保持 DF 帧大小不变，仅优化周边 |
| 校准期间用户说话 | 中 | 中 | 增加 VAD 检测，提示用户保持安静 |
| 架构重构引入 bug | 中 | 高 | 渐进式重构，保留旧代码作为回退 |
| 不同平台表现差异 | 中 | 中 | 增加平台特定测试，尤其是 Windows |

---

## 十、后续扩展建议

1. **多通道支持**：当前仅支持单声道，可扩展为立体声处理
2. **GPU 加速**：频谱分析和 EQ 计算可迁移到 GPU
3. **模型热更新**：支持运行时切换 DeepFilter 模型
4. **远程监控**：增加 WebSocket 接口，支持远程查看处理状态
5. **A/B 测试框架**：便于对比不同参数配置效果

---

## 附录：关键代码修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `capture.rs` | 重构 | 拆分 worker 函数，集成 AudioBus |
| `audio/mod.rs` | 修改 | 增加新模块导出 |
| `audio/agc.rs` | 修改 | 实现 AudioProcessor trait，移除内部限幅 |
| `audio/highpass.rs` | 修改 | 实现 AudioProcessor trait |
| `audio/transient_shaper.rs` | 修改 | 实现 AudioProcessor trait |
| `audio/saturation.rs` | 修改 | 实现 AudioProcessor trait |
| `audio/eq/dynamic_eq.rs` | 修改 | 实现 AudioProcessor trait，移除内部限幅 |
| `audio/bus/*` | 新增 | 音频总线架构 |
| `audio/calibration.rs` | 新增 | 校准系统 |
| `audio/adaptive.rs` | 新增 | 运行时自适应 |
| `audio/latency.rs` | 新增 | 延迟管理 |
| `audio/recording.rs` | 新增 | 流式录音 |

---

**文档结束**

如有问题请联系音频架构组讨论。