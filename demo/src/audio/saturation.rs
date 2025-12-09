use log::warn;

pub struct Saturation {
    drive: f32,
    makeup_db: f32,
    mix: f32,
    compensate: bool,
}

impl Saturation {
    pub fn new() -> Self {
        Self {
            drive: 1.2,
            makeup_db: -0.5,
            mix: 1.0,
            compensate: true,
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
    }

    #[allow(dead_code)]
    pub fn set_compensate(&mut self, enable: bool) {
        self.compensate = enable;
    }

    pub fn process(&self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if sanitize_samples("Saturation", samples) {
            return;
        }
        let drive = self.drive;
        let makeup = db_to_linear(self.makeup_db);
        let wet_ratio = self.mix;
        let dry_ratio = 1.0 - wet_ratio;
        for sample in samples.iter_mut() {
            let dry = *sample;
            let driven = if self.compensate {
                (dry * drive).tanh() / drive
            } else {
                (dry * drive).tanh()
            };
            let driven = driven * makeup;
            *sample = driven * wet_ratio + dry * dry_ratio;
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
