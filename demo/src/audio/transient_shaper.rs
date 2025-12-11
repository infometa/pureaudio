use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TransientShaper {
    sample_rate: f32,
    attack_gain_db: f32,
    sustain_gain_db: f32,
    threshold_db: f32,
    hold_ms: f32,
    envelope: f32,
    attack_coef: f32,
    release_coef: f32,
    hold_samples: usize,
    hold_counter: usize,
    prev_envelope: f32,
    is_transient: bool,
    error_count: AtomicUsize,
    // 瞬态检测参数
    relative_threshold_ratio: f32,  // 相对阈值比例
    absolute_threshold_multiplier: f32,  // 绝对阈值倍数
}

impl TransientShaper {
    pub fn new(sample_rate: f32) -> Self {
        let mut shaper = Self {
            sample_rate,
            attack_gain_db: 4.0,
            sustain_gain_db: 0.0,
            threshold_db: -30.0,
            hold_ms: 8.0,
            envelope: 0.0,
            attack_coef: 0.0,
            release_coef: 0.0,
            hold_samples: 0,
            hold_counter: 0,
            prev_envelope: 0.0,
            is_transient: false,
            error_count: AtomicUsize::new(0),
            // 简化后的检测参数
            relative_threshold_ratio: 0.10,  // 10% 相对变化
            absolute_threshold_multiplier: 1.3,  // 1.3倍绝对阈值
        };
        shaper.update_coefficients();
        shaper
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if self.sanitize_samples(samples) {
            return;
        }
        let threshold_linear = db_to_linear(self.threshold_db);
        let attack_gain = db_to_linear(self.attack_gain_db);
        let sustain_gain = db_to_linear(self.sustain_gain_db);

        for sample in samples.iter_mut() {
            let input = *sample;
            let abs_input = input.abs();

            let coef = if abs_input > self.envelope {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.envelope = coef * self.envelope + (1.0 - coef) * abs_input;

            // 改进的瞬态检测：使用独立的相对和绝对阈值
            let envelope_delta = self.envelope - self.prev_envelope;
            
            // 相对检测：包络增长速度超过当前包络的一定比例
            let relative_threshold = self.envelope * self.relative_threshold_ratio;
            let relative_detected = envelope_delta > relative_threshold;
            
            // 绝对检测：包络超过设定阈值的倍数
            let absolute_threshold = threshold_linear * self.absolute_threshold_multiplier;
            let absolute_detected = self.envelope > absolute_threshold;
            
            // 两个条件都满足才认为是瞬态
            if relative_detected && absolute_detected {
                self.is_transient = true;
                // 固定延长，而非之前的max(x*2, x)永远返回x*2
                self.hold_counter = self.hold_samples + (self.hold_samples / 2);
            }

            if self.hold_counter > 0 {
                self.hold_counter -= 1;
            } else {
                self.is_transient = false;
            }

            self.prev_envelope = self.envelope;

            let gain = if self.is_transient {
                attack_gain
            } else {
                sustain_gain
            };
            
            // 注意：这是全湿处理（无dry/wet混合）
            // 所有信号都经过增益调整，可能改变整体电平
            *sample = input * gain;
        }
    }

    pub fn set_attack_gain(&mut self, db: f32) {
        // 允许负值以在键盘/呼吸等冲击时压制瞬态
        self.attack_gain_db = db.clamp(-12.0, 12.0);
    }

    pub fn set_sustain_gain(&mut self, db: f32) {
        self.sustain_gain_db = db.clamp(-12.0, 6.0);
    }

    /// 设置瞬态检测灵敏度
    /// ratio: 相对阈值比例 (0.05 = 5%变化触发, 0.15 = 15%变化触发)
    pub fn set_sensitivity(&mut self, ratio: f32) {
        self.relative_threshold_ratio = ratio.clamp(0.05, 0.25);
    }
    
    /// 设置绝对阈值倍数
    pub fn set_threshold_multiplier(&mut self, multiplier: f32) {
        self.absolute_threshold_multiplier = multiplier.clamp(1.0, 2.0);
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.prev_envelope = 0.0;
        self.hold_counter = 0;
        self.is_transient = false;
    }

    fn update_coefficients(&mut self) {
        let attack_ms = 0.1;
        let release_ms = 200.0;
        
        // 采样率限制使用合理范围，避免极端值
        let sr = self.sample_rate.clamp(8000.0, 192000.0);
        
        self.attack_coef = (-1000.0 / (attack_ms * sr)).exp();
        self.release_coef = (-1000.0 / (release_ms * sr)).exp();
        self.hold_samples = ((self.hold_ms * sr) / 1000.0).round() as usize;
    }

    #[allow(dead_code)]
    pub fn set_detection(
        &mut self,
        threshold_db: f32,
        attack_ms: f32,
        release_ms: f32,
        hold_ms: f32,
    ) {
        self.threshold_db = threshold_db.clamp(-60.0, -3.0);
        let attack_ms = attack_ms.clamp(0.05, 20.0);
        let release_ms = release_ms.clamp(10.0, 400.0);
        self.hold_ms = hold_ms.clamp(1.0, 40.0);
        self.attack_coef = (-1000.0 / (attack_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
        self.release_coef = (-1000.0 / (release_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
        self.hold_samples = ((self.hold_ms * self.sample_rate) / 1000.0).round() as usize;
    }

    fn sanitize_samples(&self, samples: &mut [f32]) -> bool {
        let mut found = false;
        for sample in samples.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
                found = true;
            }
        }
        if found {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
        found
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}
