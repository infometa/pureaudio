use log::warn;

pub struct AutoGainControl {
    sample_rate: f32,
    target_level_db: f32,
    max_gain_db: f32,
    max_attenuation_db: f32,
    current_gain_db: f32,
    attack_coef: f32,
    release_coef: f32,
    rms_buffer: Vec<f32>,
    rms_index: usize,
    rms_sum: f32,
}

impl AutoGainControl {
    pub fn new(sample_rate: f32) -> Self {
        let buffer_size = ((sample_rate * 0.6).round() as usize).max(1);
        Self {
            sample_rate,
            target_level_db: -20.0,
            max_gain_db: 9.0,
            max_attenuation_db: 6.0,
            current_gain_db: 0.0,
            attack_coef: 0.9,    // 快速降低增益（压制响信号）
            release_coef: 0.995, // 慢速提升增益（恢复轻信号）
            rms_buffer: vec![0.0; buffer_size],
            rms_index: 0,
            rms_sum: 0.0,
        }
    }

    pub fn set_window_seconds(&mut self, seconds: f32) {
        let len = ((self.sample_rate * seconds.max(0.05)).round() as usize).max(1);
        self.rms_buffer.resize(len, 0.0);
        self.rms_sum = 0.0;
        self.rms_index = 0;
    }

    pub fn set_attack_release(&mut self, attack_ms: f32, release_ms: f32) {
        let attack_ms = attack_ms.clamp(0.1, 200.0);
        let release_ms = release_ms.clamp(1.0, 1000.0);
        self.attack_coef = (-1000.0 / (attack_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
        self.release_coef = (-1000.0 / (release_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if sanitize_samples("AutoGainControl", samples) {
            return;
        }
        for &sample in samples.iter() {
            let old_val = self.rms_buffer[self.rms_index];
            let new_val = sample.powi(2);
            self.rms_sum = self.rms_sum - old_val + new_val;
            self.rms_buffer[self.rms_index] = new_val;
            self.rms_index = (self.rms_index + 1) % self.rms_buffer.len();
        }

        let win_len = self.rms_buffer.len().max(1) as f32;
        let rms = (self.rms_sum / win_len).max(1e-10).sqrt();
        let rms_db = 20.0 * rms.log10();

        let required_gain_db = self.target_level_db - rms_db;
        let limited_gain_db = required_gain_db.clamp(-self.max_attenuation_db, self.max_gain_db);
        let coef = if limited_gain_db < self.current_gain_db {
            // 需要降低增益 → 使用 attack（快速响应）
            self.attack_coef
        } else {
            // 需要提升增益 → 使用 release（缓慢响应）
            self.release_coef
        };
        self.current_gain_db = coef * self.current_gain_db + (1.0 - coef) * limited_gain_db;

        let gain_linear = db_to_linear(self.current_gain_db);
        let mut peak_after_gain: f32 = 0.0;
        for sample in samples.iter_mut() {
            *sample *= gain_linear;
            peak_after_gain = peak_after_gain.max(sample.abs());
        }

        const LIMIT_THRESHOLD: f32 = 0.9;
        const LIMIT_CEILING: f32 = 0.98;
        if peak_after_gain > LIMIT_THRESHOLD {
            for sample in samples.iter_mut() {
                *sample = soft_clip(*sample, LIMIT_THRESHOLD, LIMIT_CEILING);
            }
            let over_db = 20.0 * (peak_after_gain / LIMIT_CEILING.max(1e-6)).log10();
            if over_db.is_finite() && over_db > 0.0 {
                self.current_gain_db =
                    (self.current_gain_db - over_db).max(-self.max_attenuation_db);
            }
        }
        self.current_gain_db =
            self.current_gain_db.clamp(-self.max_attenuation_db, self.max_gain_db);
    }

    pub fn current_gain_db(&self) -> f32 {
        self.current_gain_db
    }

    pub fn target_level_db(&self) -> f32 {
        self.target_level_db
    }

    pub fn max_gain_db(&self) -> f32 {
        self.max_gain_db
    }

    pub fn max_attenuation_db(&self) -> f32 {
        self.max_attenuation_db
    }

    pub fn reset(&mut self) {
        self.rms_buffer.fill(0.0);
        self.rms_sum = 0.0;
        self.rms_index = 0;
        self.current_gain_db = 0.0;
    }

    pub fn set_target_level(&mut self, db: f32) {
        self.target_level_db = db.clamp(-30.0, -6.0);
    }

    pub fn set_max_gain(&mut self, db: f32) {
        self.max_gain_db = db.clamp(0.0, 20.0);
    }

    pub fn set_max_attenuation(&mut self, db: f32) {
        self.max_attenuation_db = db.clamp(0.0, 15.0);
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn soft_clip(x: f32, threshold: f32, ceiling: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= threshold {
        x
    } else if abs_x < ceiling {
        let t = (abs_x - threshold) / (ceiling - threshold);
        let soft = threshold + (ceiling - threshold) * (3.0 * t.powi(2) - 2.0 * t.powi(3));
        x.signum() * soft
    } else {
        x.signum() * ceiling
    }
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
        warn!("{tag} 检测到非法音频数据 (NaN/Inf)，已重置该帧");
    }
    found
}
