use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Silero VAD 配置
/// 
/// 注意：Silero VAD模型仅支持8kHz和16kHz采样率
/// 其他采样率需要预先重采样
#[derive(Clone, Copy)]
pub struct SileroVadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub redemption_frames: usize,
    pub sample_rate: usize,  // 必须为8000或16000
}

impl Default for SileroVadConfig {
    fn default() -> Self {
        Self {
            positive_speech_threshold: 0.5,
            negative_speech_threshold: 0.35,
            redemption_frames: 20,
            sample_rate: 16000,
        }
    }
}

pub struct SileroVad {
    session: Session,
    h: Vec<f32>,
    c: Vec<f32>,
    cfg: SileroVadConfig,
    hidden_size: usize,
    num_layers: usize,
    redemption: usize,
    speaking: bool,
    // 性能统计
    total_frames: std::sync::atomic::AtomicU64,
}

impl SileroVad {
    pub fn new(model_path: impl AsRef<std::path::Path>, cfg: SileroVadConfig) -> Result<Self> {
        // 验证采样率
        if cfg.sample_rate != 8000 && cfg.sample_rate != 16000 {
            return Err(anyhow::anyhow!(
                "Silero VAD仅支持8kHz或16kHz采样率，当前设置: {}Hz",
                cfg.sample_rate
            ));
        }
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        let hidden_size = 64;
        let num_layers = 2;
        Ok(Self {
            session,
            h: vec![0.0; hidden_size * num_layers],
            c: vec![0.0; hidden_size * num_layers],
            cfg,
            hidden_size,
            num_layers,
            redemption: 0,
            speaking: false,
            total_frames: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// 输入 30ms 帧（需 16k/8k采样率），返回当前语音状态
    /// 
    /// 帧长要求：
    /// - 16kHz: 480 samples (30ms)
    /// - 8kHz: 240 samples (30ms)
    pub fn process(&mut self, frame: &[f32]) -> Result<bool> {
        self.total_frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // 验证帧长
        let expected_len = (self.cfg.sample_rate * 30) / 1000;  // 30ms
        if frame.len() != expected_len {
            return Err(anyhow::anyhow!(
                "VAD帧长不匹配：期望{}样本(30ms @ {}Hz)，实际{}样本",
                expected_len, self.cfg.sample_rate, frame.len()
            ));
        }
        
        // 注意：这里的to_vec()和clone()是性能瓶颈
        // ort 2.0 API限制，必须提供数据所有权
        // 每次调用产生约 (480 + 4 + 128*2) = 740 floats = 2960 bytes的堆分配
        let audio = Tensor::from_array((vec![1usize, frame.len()], frame.to_vec()))?;
        let sr = Tensor::from_array((vec![1usize], vec![self.cfg.sample_rate as i64]))?;
        let h_in = Tensor::from_array((vec![self.num_layers, 1usize, self.hidden_size], self.h.clone()))?;
        let c_in = Tensor::from_array((vec![self.num_layers, 1usize, self.hidden_size], self.c.clone()))?;

        let outputs = self.session.run(ort::inputs![audio, sr, h_in, c_in])?;
        let prob_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let prob = *prob_tensor.1.first().unwrap_or(&0.0);
        let h_out = outputs[1].try_extract_tensor::<f32>()?;
        let c_out = outputs[2].try_extract_tensor::<f32>()?;
        
        // 优化：仅在大小匹配时复制，避免不必要的重分配
        if self.h.len() == h_out.1.len() {
            self.h.copy_from_slice(h_out.1);
        } else {
            self.h = h_out.1.to_vec();
        }
        if self.c.len() == c_out.1.len() {
            self.c.copy_from_slice(c_out.1);
        } else {
            self.c = c_out.1.to_vec();
        }

        if prob > self.cfg.positive_speech_threshold {
            self.speaking = true;
            self.redemption = 0;
        } else if prob < self.cfg.negative_speech_threshold {
            self.redemption = self.redemption.saturating_add(1);
            if self.redemption > self.cfg.redemption_frames {
                self.speaking = false;
            }
        } else {
            self.redemption = 0;
        }
        Ok(self.speaking)
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.redemption = 0;
        self.speaking = false;
    }
    
    /// 获取处理的总帧数
    pub fn total_frames(&self) -> u64 {
        self.total_frames.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 获取redemption配置（毫秒）
    pub fn redemption_time_ms(&self) -> f32 {
        let frame_ms = 30.0;  // 固定30ms帧长
        self.cfg.redemption_frames as f32 * frame_ms
    }
    
    /// 优化1.4：动态调整VAD阈值
    /// 
    /// 根据环境噪音水平自适应调整阈值：
    /// - 噪音大 → 提高阈值（减少误触发）
    /// - 噪音小 → 降低阈值（提高灵敏度）
    /// 
    /// # 参数
    /// - `adjustment`: 阈值调整量（dB），范围建议 ±5dB
    pub fn adjust_thresholds(&mut self, adjustment: f32) {
        // 将dB转换为概率调整（简化映射）
        // adjustment > 0 → 提高阈值（降低灵敏度）
        // adjustment < 0 → 降低阈值（提高灵敏度）
        let delta = adjustment * 0.02;  // 每dB调整2%
        
        // 调整正向阈值（激活阈值）
        let new_positive = (self.cfg.positive_speech_threshold + delta).clamp(0.3, 0.8);
        
        // 调整负向阈值（去激活阈值），保持与正向阈值的差距
        let threshold_gap = 0.15;  // 保持15%的gap
        let new_negative = (new_positive - threshold_gap).max(0.15);
        
        self.cfg.positive_speech_threshold = new_positive;
        self.cfg.negative_speech_threshold = new_negative;
    }
    
    /// 获取当前阈值配置
    pub fn get_thresholds(&self) -> (f32, f32) {
        (self.cfg.positive_speech_threshold, self.cfg.negative_speech_threshold)
    }
}
