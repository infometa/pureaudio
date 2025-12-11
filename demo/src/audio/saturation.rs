use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Saturation {
    drive: f32,
    makeup_db: f32,
    mix: f32,
    compensate: bool,
    // 平滑系数，避免mix参数突变产生zipper噪声
    mix_smooth: f32,
    smoothing_coef: f32,
}

impl Saturation {
    pub fn new() -> Self {
        Self {
            drive: 1.2,
            makeup_db: -0.5,
            mix: 1.0,
            compensate: true,
            mix_smooth: 1.0,
            // 平滑时间常数约10ms @ 48kHz
            smoothing_coef: 0.95,
        }
    }

    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.clamp(0.5, 2.0);
    }

    pub fn set_makeup(&mut self, makeup_db: f32) {
        self.makeup_db = makeup_db.clamp(-12.0, 6.0);
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
        // 不立即更新mix_smooth，让process()中的平滑器处理
    }
    
    /// 重置内部状态
    pub fn reset(&mut self) {
        self.mix_smooth = self.mix;
    }

    #[allow(dead_code)]
    pub fn set_compensate(&mut self, enable: bool) {
        self.compensate = enable;
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if sanitize_samples("Saturation", samples) {
            return;
        }
        
        let drive = self.drive;
        let makeup = db_to_linear(self.makeup_db);
        
        for sample in samples.iter_mut() {
            // 平滑mix参数，避免zipper噪声
            self.mix_smooth += (self.mix - self.mix_smooth) * (1.0 - self.smoothing_coef);
            let wet_ratio = self.mix_smooth;
            let dry_ratio = 1.0 - wet_ratio;
            
            let dry = *sample;
            
            // 改进的补偿算法：使用自适应gain而非简单除以drive
            // tanh(x*d)的RMS约为原信号的factor倍，factor ≈ 1/(1+0.3*d)
            let driven = if self.compensate {
                let saturation_factor = 1.0 / (1.0 + 0.3 * (drive - 1.0));
                (dry * drive).tanh() / saturation_factor
            } else {
                (dry * drive).tanh()
            };
            
            let driven = driven * makeup;
            *sample = driven * wet_ratio + dry * dry_ratio;
        }
    }
    
    /// 获取当前输出峰值和RMS（用于监控）
    pub fn get_metrics(&self, samples: &[f32]) -> (f32, f32) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        (peak, rms)
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
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
        SAT_SANITIZE_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    found
}

static SAT_SANITIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
