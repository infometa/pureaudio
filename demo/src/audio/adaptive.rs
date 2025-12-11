/// 自适应配置模块
/// 
/// 功能：
/// 1. 设备特征识别（内置麦克风、USB设备、会议音箱等）
/// 2. 音量自动监控和调整建议
/// 3. AEC延迟自动估计
/// 4. 环境噪声自适应参数调整
/// 
/// 作者：音频处理专家
/// 日期：2025-12-11

use log::{info, warn, debug};
use std::collections::VecDeque;

/// 设备类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// 内置麦克风（MacBook、台式机等）
    BuiltinMicrophone,
    /// USB外置设备
    UsbDevice,
    /// 会议音箱（如Mic 100）
    ConferenceSpeaker,
    /// 专业音频接口
    AudioInterface,
    /// 未知设备
    Unknown,
}

/// 推荐配置
#[derive(Debug, Clone)]
pub struct RecommendedConfig {
    /// 是否启用AEC
    pub enable_aec: bool,
    /// 推荐的AEC延迟（ms）
    pub aec_delay_ms: i32,
    /// 推荐的AGC最大增益（dB）
    pub agc_max_gain_db: f32,
    /// 推荐的输出音量（0.0-1.0）
    pub output_volume: f32,
    /// 推荐的高通频率（Hz）
    pub highpass_freq: f32,
    /// 配置原因说明
    pub reason: String,
}

/// 设备特征识别器
pub struct DeviceDetector;

impl DeviceDetector {
    /// 根据设备名称识别设备类型
    pub fn detect_device_type(device_name: &str) -> DeviceType {
        let name_lower = device_name.to_lowercase();
        
        // 内置麦克风特征
        if name_lower.contains("built-in") 
            || name_lower.contains("内置") 
            || name_lower.contains("macbook")
            || name_lower.contains("imac")
            || name_lower.contains("internal") {
            return DeviceType::BuiltinMicrophone;
        }
        
        // 会议音箱特征
        if name_lower.contains("conference") 
            || name_lower.contains("会议")
            || name_lower.contains("speakerphone")
            || name_lower.contains("mic 100")
            || name_lower.contains("jabra")
            || name_lower.contains("poly")
            || name_lower.contains("yealink") {
            return DeviceType::ConferenceSpeaker;
        }
        
        // 专业音频接口特征
        if name_lower.contains("focusrite")
            || name_lower.contains("scarlett")
            || name_lower.contains("motu")
            || name_lower.contains("rme")
            || name_lower.contains("universal audio")
            || name_lower.contains("apollo") {
            return DeviceType::AudioInterface;
        }
        
        // USB设备特征
        if name_lower.contains("usb") {
            return DeviceType::UsbDevice;
        }
        
        DeviceType::Unknown
    }
    
    /// 为设备类型生成推荐配置
    pub fn recommend_config(
        input_device_type: DeviceType,
        output_device_type: DeviceType,
        need_aec: bool,
    ) -> RecommendedConfig {
        match (input_device_type, output_device_type, need_aec) {
            // 内置麦克风 + 会议音箱 + 需要AEC（典型的双讲场景）
            (DeviceType::BuiltinMicrophone, DeviceType::ConferenceSpeaker, true) => {
                RecommendedConfig {
                    enable_aec: true,
                    aec_delay_ms: 60,
                    agc_max_gain_db: 3.0,  // ⚠️ 紧急降低到3dB，防止啸叫
                    output_volume: 0.3, // 30% 音量，避免回声过强
                    highpass_freq: 80.0,
                    reason: "内置麦克风+会议音箱：⚠️ AGC降低到3dB防止啸叫，请将扬声器音量降到10-12dB！".to_string(),
                }
            }
            
            // 会议音箱自带麦克风（使用硬件AEC）
            (DeviceType::ConferenceSpeaker, DeviceType::ConferenceSpeaker, _) => {
                RecommendedConfig {
                    enable_aec: false, // 硬件已处理
                    aec_delay_ms: 0,
                    agc_max_gain_db: 12.0,
                    output_volume: 0.5,
                    highpass_freq: 80.0,
                    reason: "会议音箱自带AEC，无需软件AEC".to_string(),
                }
            }
            
            // 专业音频接口（高质量，可以用更高增益）
            (DeviceType::AudioInterface, _, _) => {
                RecommendedConfig {
                    enable_aec: need_aec,
                    aec_delay_ms: 80, // 音频接口延迟通常更高
                    agc_max_gain_db: 12.0,
                    output_volume: 0.5,
                    highpass_freq: 60.0, // 更低的高通（保留更多低频）
                    reason: "专业音频接口：高质量，可用更高增益".to_string(),
                }
            }
            
            // 只有输入，不需要AEC（单讲录音）
            (_, _, false) => {
                RecommendedConfig {
                    enable_aec: false,
                    aec_delay_ms: 0,
                    agc_max_gain_db: 12.0,
                    output_volume: 0.7,
                    highpass_freq: 80.0,
                    reason: "单讲模式：无需AEC".to_string(),
                }
            }
            
            // 默认配置
            _ => {
                RecommendedConfig {
                    enable_aec: need_aec,
                    aec_delay_ms: 60,
                    agc_max_gain_db: 3.0,  // ⚠️ 默认降低到3dB，防止啸叫
                    output_volume: 0.4,
                    highpass_freq: 80.0,
                    reason: "默认配置：⚠️ AGC降低到3dB防止啸叫".to_string(),
                }
            }
        }
    }
}

/// 音量监控器
pub struct VolumeMonitor {
    /// 输入能量历史（dB）
    input_energy_history: VecDeque<f32>,
    /// 输出能量历史（dB）
    output_energy_history: VecDeque<f32>,
    /// 历史长度（帧数）
    history_len: usize,
    /// 上次警告时间（帧数）
    last_warning_frame: usize,
    /// 当前帧计数
    frame_count: usize,
}

impl VolumeMonitor {
    pub fn new(history_len: usize) -> Self {
        Self {
            input_energy_history: VecDeque::with_capacity(history_len),
            output_energy_history: VecDeque::with_capacity(history_len),
            history_len,
            last_warning_frame: 0,
            frame_count: 0,
        }
    }
    
    /// 更新输入能量
    pub fn update_input(&mut self, energy_db: f32) {
        if self.input_energy_history.len() >= self.history_len {
            self.input_energy_history.pop_front();
        }
        self.input_energy_history.push_back(energy_db);
        self.frame_count += 1;
    }
    
    /// 更新输出能量
    pub fn update_output(&mut self, energy_db: f32) {
        if self.output_energy_history.len() >= self.history_len {
            self.output_energy_history.pop_front();
        }
        self.output_energy_history.push_back(energy_db);
    }
    
    /// 获取输入平均能量
    pub fn get_avg_input_db(&self) -> f32 {
        if self.input_energy_history.is_empty() {
            return -80.0;
        }
        let sum: f32 = self.input_energy_history.iter().sum();
        sum / self.input_energy_history.len() as f32
    }
    
    /// 获取输出平均能量
    pub fn get_avg_output_db(&self) -> f32 {
        if self.output_energy_history.is_empty() {
            return -80.0;
        }
        let sum: f32 = self.output_energy_history.iter().sum();
        sum / self.output_energy_history.len() as f32
    }
    
    /// 检查是否需要调整音量（每5秒检查一次）
    pub fn check_volume_adjustment(&mut self) -> Option<VolumeAdjustment> {
        // 限流：每5秒（500帧@100fps）检查一次
        if self.frame_count - self.last_warning_frame < 500 {
            return None;
        }
        
        let avg_input = self.get_avg_input_db();
        let avg_output = self.get_avg_output_db();
        
        // 输出音量过高（可能导致啸叫）
        if avg_output > -15.0 {
            self.last_warning_frame = self.frame_count;
            return Some(VolumeAdjustment {
                adjustment_type: AdjustmentType::OutputTooHigh,
                current_db: avg_output,
                recommended_db: -25.0,
                reason: "输出音量过高，可能导致啸叫".to_string(),
            });
        }
        
        // 输入音量过低（信噪比差）
        if avg_input < -50.0 && avg_input > -70.0 {
            self.last_warning_frame = self.frame_count;
            return Some(VolumeAdjustment {
                adjustment_type: AdjustmentType::InputTooLow,
                current_db: avg_input,
                recommended_db: -30.0,
                reason: "输入音量过低，建议增加麦克风增益或靠近麦克风".to_string(),
            });
        }
        
        // 输入音量过高（可能削波）
        if avg_input > -5.0 {
            self.last_warning_frame = self.frame_count;
            return Some(VolumeAdjustment {
                adjustment_type: AdjustmentType::InputTooHigh,
                current_db: avg_input,
                recommended_db: -15.0,
                reason: "输入音量过高，可能削波失真".to_string(),
            });
        }
        
        None
    }
}

/// 音量调整建议
#[derive(Debug, Clone)]
pub struct VolumeAdjustment {
    pub adjustment_type: AdjustmentType,
    pub current_db: f32,
    pub recommended_db: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustmentType {
    OutputTooHigh,
    OutputTooLow,
    InputTooHigh,
    InputTooLow,
}

/// AEC延迟估计器（使用互相关分析）
pub struct DelayEstimator {
    /// 近端信号缓冲（用于相关性分析）
    near_buffer: VecDeque<f32>,
    /// 远端信号缓冲
    far_buffer: VecDeque<f32>,
    /// 缓冲区大小（样本数）
    buffer_size: usize,
    /// 采样率
    sample_rate: usize,
    /// 上次估计时间（帧数）
    last_estimate_frame: usize,
    /// 当前帧计数
    frame_count: usize,
    /// 当前估计的延迟（ms）
    estimated_delay_ms: i32,
}

impl DelayEstimator {
    pub fn new(sample_rate: usize, buffer_duration_ms: usize) -> Self {
        let buffer_size = sample_rate * buffer_duration_ms / 1000;
        Self {
            near_buffer: VecDeque::with_capacity(buffer_size),
            far_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            sample_rate,
            last_estimate_frame: 0,
            frame_count: 0,
            estimated_delay_ms: 60, // 默认60ms
        }
    }
    
    /// 添加近端和远端信号样本
    pub fn add_samples(&mut self, near: &[f32], far: &[f32]) {
        for (&n, &f) in near.iter().zip(far.iter()) {
            if self.near_buffer.len() >= self.buffer_size {
                self.near_buffer.pop_front();
            }
            if self.far_buffer.len() >= self.buffer_size {
                self.far_buffer.pop_front();
            }
            self.near_buffer.push_back(n);
            self.far_buffer.push_back(f);
        }
        self.frame_count += 1;
    }
    
    /// 估计延迟（每1秒估计一次）
    pub fn estimate_delay(&mut self) -> Option<i32> {
        // 限流：每1秒（100 frames @ 100fps）估计一次
        if self.frame_count - self.last_estimate_frame < 100 {
            return None;
        }
        
        // 检查缓冲区是否已满
        if self.near_buffer.len() < self.buffer_size || self.far_buffer.len() < self.buffer_size {
            // [DEBUG] 缓冲区未满
             // log::warn!("AEC Buffer Not Full: {}/{}", self.near_buffer.len(), self.buffer_size); // Too noisy?
            return None;
        }
        
        self.last_estimate_frame = self.frame_count;
        
        // 归一化互相关分析
        // 搜索范围：0-500ms
        let max_delay_samples = (self.sample_rate * 500) / 1000;
        let mut max_correlation = 0.0f32;
        let mut best_delay_samples = 0;
        
        // 提前检查远端信号能量
        let total_far_energy: f32 = self.far_buffer.iter().map(|&x| x * x).sum();
        if total_far_energy < 1e-6 {
            debug!("AEC延迟估计：远端信号能量太低，跳过");
            log::warn!("🔍 AEC延迟估计跳过: 远端参考信号能量过低 ({:.8})", total_far_energy);
            return None;
        }
        
        // 搜索最佳延迟（归一化互相关）
        // 注意：near[i] 是麦克风采集，far[i+delay] 是远端参考
        // 我们要找的是：near 信号比 far 信号晚了多少样本（回声路径延迟）
        let buffer_len = self.near_buffer.len().min(self.far_buffer.len());
        for delay in 0..max_delay_samples.min(buffer_len.saturating_sub(64)) {
            let valid_len = buffer_len.saturating_sub(delay);
            if valid_len < 64 { continue; }
            
            let mut correlation = 0.0f32;
            let mut near_energy = 0.0f32;
            let mut far_energy = 0.0f32;
            
            // 完整计算（不跳过），保证精度
            for i in 0..valid_len {
                let near_val = self.near_buffer[i];
                let far_idx = i + delay;
                if far_idx < self.far_buffer.len() {
                    let far_val = self.far_buffer[far_idx];
                    correlation += near_val * far_val;
                    near_energy += near_val * near_val;
                    far_energy += far_val * far_val;
                }
            }
            
            // 归一化
            let norm = (near_energy * far_energy).sqrt();
            if norm > 1e-10 {
                correlation /= norm;
            }
            
            if correlation.abs() > max_correlation.abs() {
                max_correlation = correlation;
                best_delay_samples = delay;
            }
        }
        
        // 转换为毫秒
        let delay_ms = (best_delay_samples * 1000) / self.sample_rate;
        
        // 相关性阈值判断
        if max_correlation.abs() > 0.05 {
            self.estimated_delay_ms = delay_ms as i32;
            log::warn!("🔍 AEC延迟自动估计: {}ms (归一化相关性: {:.3})", delay_ms, max_correlation);
            Some(delay_ms as i32)
        } else {
            log::warn!("🔍 AEC延迟估计失败: 相关性太低 ({:.3})，保持当前值", max_correlation);
            None
        }
    }
    
    /// 获取当前估计的延迟
    pub fn get_estimated_delay(&self) -> i32 {
        self.estimated_delay_ms
    }
}

/// 环境噪声分析器
pub struct NoiseAnalyzer {
    /// 噪声底噪历史（dB）
    noise_floor_history: VecDeque<f32>,
    /// 历史长度
    history_len: usize,
    /// 当前噪声底噪（dB）
    current_noise_floor: f32,
}

impl NoiseAnalyzer {
    pub fn new(history_len: usize) -> Self {
        Self {
            noise_floor_history: VecDeque::with_capacity(history_len),
            history_len,
            current_noise_floor: -60.0,
        }
    }
    
    /// 更新噪声底噪（在VAD检测到静音时更新）
    pub fn update_noise_floor(&mut self, energy_db: f32, is_silence: bool) {
        if is_silence {
            if self.noise_floor_history.len() >= self.history_len {
                self.noise_floor_history.pop_front();
            }
            self.noise_floor_history.push_back(energy_db);
            
            // 计算平均噪声底噪
            if !self.noise_floor_history.is_empty() {
                let sum: f32 = self.noise_floor_history.iter().sum();
                self.current_noise_floor = sum / self.noise_floor_history.len() as f32;
            }
        }
    }
    
    /// 获取当前噪声底噪
    pub fn get_noise_floor_db(&self) -> f32 {
        self.current_noise_floor
    }
    
    /// 根据噪声底噪推荐降噪强度
    pub fn recommend_noise_suppression(&self) -> f32 {
        // 噪声越高，降噪越强
        if self.current_noise_floor > -40.0 {
            35.0 // 高噪声环境：强降噪
        } else if self.current_noise_floor > -50.0 {
            30.0 // 中等噪声：标准降噪
        } else {
            25.0 // 低噪声环境：轻度降噪（保留更多音质）
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_detection() {
        assert_eq!(
            DeviceDetector::detect_device_type("MacBook Pro麦克风"),
            DeviceType::BuiltinMicrophone
        );
        
        assert_eq!(
            DeviceDetector::detect_device_type("Mic 100会议音箱1"),
            DeviceType::ConferenceSpeaker
        );
        
        assert_eq!(
            DeviceDetector::detect_device_type("Focusrite Scarlett 2i2"),
            DeviceType::AudioInterface
        );
    }
    
    #[test]
    fn test_volume_monitor() {
        let mut monitor = VolumeMonitor::new(100);
        
        // 模拟输出音量过高
        for _ in 0..100 {
            monitor.update_output(-10.0); // 很高的输出
        }
        monitor.frame_count = 600; // 跳过限流
        
        let adjustment = monitor.check_volume_adjustment();
        assert!(adjustment.is_some());
        
        if let Some(adj) = adjustment {
            assert_eq!(adj.adjustment_type, AdjustmentType::OutputTooHigh);
        }
    }
}
