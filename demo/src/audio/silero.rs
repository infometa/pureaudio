use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Silero VAD 配置
#[derive(Clone, Copy)]
pub struct SileroVadConfig {
    pub positive_speech_threshold: f32,
    pub negative_speech_threshold: f32,
    pub redemption_frames: usize,
    pub sample_rate: usize,
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
}

impl SileroVad {
    pub fn new(model_path: impl AsRef<std::path::Path>, cfg: SileroVadConfig) -> Result<Self> {
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
        })
    }

    /// 输入 30ms 帧（需 16k/8k），返回当前语音状态
    pub fn process(&mut self, frame: &[f32]) -> Result<bool> {
        let audio = Tensor::from_array((vec![1usize, frame.len()], frame.to_vec()))?;
        let sr = Tensor::from_array((vec![1usize], vec![self.cfg.sample_rate as i64]))?;
        let h_in = Tensor::from_array((vec![self.num_layers, 1usize, self.hidden_size], self.h.clone()))?;
        let c_in = Tensor::from_array((vec![self.num_layers, 1usize, self.hidden_size], self.c.clone()))?;

        let outputs = self.session.run(ort::inputs![audio, sr, h_in, c_in])?;
        let prob_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let prob = *prob_tensor.1.first().unwrap_or(&0.0);
        let h_out = outputs[1].try_extract_tensor::<f32>()?;
        let c_out = outputs[2].try_extract_tensor::<f32>()?;
        self.h = h_out.1.to_vec();
        self.c = c_out.1.to_vec();

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
}
