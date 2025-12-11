use log::{error, warn};
use std::sync::atomic::{AtomicUsize, Ordering};
use webrtc_audio_processing::{Config, GainControl, GainControlMode, InitializationConfig, Processor};

/// WebRTC 数字 AGC 封装，替换自研实现，减少开口/起音吞噪。
pub struct AutoGainControl {
    processor: Option<Processor>,
    cfg: GainControl,
    frame_size: usize,
    scratch: Vec<f32>,
    enabled: bool,
}

impl AutoGainControl {
    pub fn new(sample_rate: f32, hop_size: usize) -> Self {
        // WebRTC APM 期望 10ms 帧长；按 sr/100 计算期望采样数
        let frame_size = (sample_rate / 100.0).round() as usize;
        let init = InitializationConfig { num_capture_channels: 1, num_render_channels: 0, ..Default::default() };
        let cfg = GainControl {
            mode: GainControlMode::AdaptiveDigital,
            target_level_dbfs: 16,     // -16 dBFS 目标（更保守，避免啸叫）
            compression_gain_db: 9,    // ✅ 降低到9dB（原15dB太大，会放大残留回声导致啸叫）
            enable_limiter: false,     // 内限幅关闭，交给最终限幅器
        };
        let frame_ok = frame_size > 0 && hop_size % frame_size == 0;
        let processor = if frame_ok {
            match Processor::new(&init) {
                Ok(mut p) => {
                    p.set_config(Config { gain_control: Some(cfg.clone()), ..Config::default() });
                    Some(p)
                }
                Err(err) => {
                    error!("初始化 WebRTC AGC 失败，回退旁路: {err}");
                    None
                }
            }
        } else {
            warn!(
                "WebRTC AGC 旁路：DF hop_size={} 与 10ms 帧长({} samples)不匹配",
                hop_size, frame_size
            );
            None
        };
        Self {
            processor,
            cfg,
            frame_size,
            scratch: vec![0.0; frame_size],
            enabled: frame_ok,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if !self.enabled {
            return;
        }
        if sanitize_samples("WebRTC AGC", samples) {
            return;
        }
        let Some(proc) = self.processor.as_mut() else { return; };
        let frame = self.frame_size;
        for chunk in samples.chunks_mut(frame) {
            if chunk.len() == frame {
                if let Err(e) = proc.process_capture_frame(chunk) {
                    warn!("WebRTC AGC 处理失败: {e}");
                }
            } else {
                self.scratch[..chunk.len()].copy_from_slice(chunk);
                self.scratch[chunk.len()..frame].fill(0.0);
                if let Err(e) = proc.process_capture_frame(&mut self.scratch[..]) {
                    warn!("WebRTC AGC 处理失败: {e}");
                }
                chunk.copy_from_slice(&self.scratch[..chunk.len()]);
            }
        }
    }

    /// 获取当前增益估算值（WebRTC APM 不直接暴露增益，返回None）
    /// 调用者应通过监控输入/输出RMS来计算实际增益
    pub fn current_gain_db(&self) -> Option<f32> {
        None
    }

    pub fn reset(&mut self) {
        if let Some(proc) = self.processor.as_mut() {
            proc.set_config(Config { gain_control: Some(self.cfg.clone()), ..Config::default() });
        }
    }

    pub fn set_target_level(&mut self, db: f32) {
        // WebRTC 参数是“距满刻度的正值（0~31 dBFS）”，UI 负值取绝对值映射
        self.cfg.target_level_dbfs = db.abs().round().clamp(0.0, 31.0) as i32;
        self.apply_cfg();
    }

    pub fn set_max_gain(&mut self, db: f32) {
        self.cfg.compression_gain_db = db.round().clamp(0.0, 90.0) as i32;
        self.apply_cfg();
    }

    /// WebRTC AGC 不支持最大衰减参数配置
    pub fn set_max_attenuation(&mut self, _db: f32) {
        log::warn!("WebRTC AGC 不支持 max_attenuation 参数，调用被忽略");
    }

    /// WebRTC AGC 内部窗口不可配置
    pub fn set_window_seconds(&mut self, _seconds: f32) {
        log::warn!("WebRTC AGC 不支持 window_seconds 参数，调用被忽略");
    }

    /// WebRTC AGC 使用内部自适应时间常数，不可配置
    pub fn set_attack_release(&mut self, _attack_ms: f32, _release_ms: f32) {
        log::warn!("WebRTC AGC 不支持 attack/release 参数，调用被忽略");
    }

    fn apply_cfg(&mut self) {
        if let Some(proc) = self.processor.as_mut() {
            proc.set_config(Config { gain_control: Some(self.cfg.clone()), ..Config::default() });
        }
    }
}

fn sanitize_samples(_tag: &str, samples: &mut [f32]) -> bool {
    let mut found = false;
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            found = true;
        }
    }
    if found {
        AGC_SANITIZE_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    found
}

static AGC_SANITIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
