use log::{error, warn};
use webrtc_audio_processing::{
    Config, EchoCancellation, EchoCancellationSuppressionLevel, InitializationConfig, Processor,
};

/// 简易 AEC3 包装（基于 WebRTC Audio Processing），默认关闭，可按需启用。
pub struct EchoCanceller {
    processor: Option<Processor>,
    frame_size: usize,
    scratch: Vec<f32>,
    enabled: bool,
    active: bool,
    delay_ms: i32,
}

impl EchoCanceller {
    pub fn new(sample_rate: f32, hop_size: usize, delay_ms: i32) -> Self {
        let frame_size = (sample_rate / 100.0).round() as usize; // 10ms 帧
        let frame_ok = frame_size > 0 && hop_size % frame_size == 0;
        let init = InitializationConfig {
            num_capture_channels: 1,
            num_render_channels: 1,
            ..InitializationConfig::default()
        };
        let processor = if frame_ok {
            match Processor::new(&init) {
                Ok(mut p) => {
                    let ec = EchoCancellation {
                        suppression_level: EchoCancellationSuppressionLevel::High,
                        stream_delay_ms: Some(delay_ms.max(0)),
                        enable_delay_agnostic: true,
                        enable_extended_filter: true,
                    };
                    p.set_config(Config {
                        echo_cancellation: Some(ec),
                        enable_high_pass_filter: true,
                        ..Config::default()
                    });
                    Some(p)
                }
                Err(e) => {
                    error!("初始化 AEC3 失败，旁路: {e}");
                    None
                }
            }
        } else {
            warn!(
                "AEC3 旁路：DF hop_size={} 与 10ms 帧长({} samples)不匹配",
                hop_size, frame_size
            );
            None
        };
        let active = frame_ok && processor.is_some();
        Self { processor, frame_size, scratch: vec![0.0; frame_size], enabled: frame_ok, active, delay_ms }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.processor.is_some();
        self.active = self.enabled;
        if enabled && self.processor.is_none() {
            warn!("AEC3 启用失败：处理器不可用（帧长不匹配或初始化失败）");
        }
    }

    pub fn set_delay_ms(&mut self, delay_ms: i32) {
        self.delay_ms = delay_ms.max(0);
        if let Some(proc) = self.processor.as_mut() {
            let ec = EchoCancellation {
                suppression_level: EchoCancellationSuppressionLevel::High,
                stream_delay_ms: Some(self.delay_ms),
                enable_delay_agnostic: true,
                enable_extended_filter: true,
            };
            let cfg = Config { echo_cancellation: Some(ec), enable_high_pass_filter: true, ..Config::default() };
            proc.set_config(cfg);
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn process_capture(&mut self, buf: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let Some(proc) = self.processor.as_mut() else { return; };
        let frame = self.frame_size;
        for chunk in buf.chunks_mut(frame) {
            if chunk.len() == frame {
                if let Err(e) = proc.process_capture_frame(chunk) {
                    warn!("AEC3 捕获处理失败: {e}");
                }
            } else {
                // pad 短帧
                self.scratch[..chunk.len()].copy_from_slice(chunk);
                self.scratch[chunk.len()..frame].fill(0.0);
                if let Err(e) = proc.process_capture_frame(&mut self.scratch[..]) {
                    warn!("AEC3 捕获处理失败: {e}");
                }
                chunk.copy_from_slice(&self.scratch[..chunk.len()]);
            }
        }
    }

    pub fn process_render(&mut self, buf: &[f32]) {
        if !self.enabled {
            return;
        }
        let Some(proc) = self.processor.as_mut() else { return; };
        let frame = self.frame_size;
        for chunk in buf.chunks(frame) {
            if chunk.len() == frame {
                let mut temp = chunk.to_vec();
                if let Err(e) = proc.process_render_frame(&mut temp) {
                    warn!("AEC3 渲染参考处理失败: {e}");
                }
            } else {
                self.scratch[..chunk.len()].copy_from_slice(chunk);
                self.scratch[chunk.len()..frame].fill(0.0);
                if let Err(e) = proc.process_render_frame(&mut self.scratch[..]) {
                    warn!("AEC3 渲染参考处理失败: {e}");
                }
            }
        }
    }
}
