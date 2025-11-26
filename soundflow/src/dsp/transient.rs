//! Transient shaper implementation adapted from the original soundflow project.
//!
//! A transient shaper enhances or suppresses the attack portion of a signal and
//! can either bring out percussive transients or smooth them. This simple
//! implementation analyses the envelope of the absolute value of the signal
//! and applies either an attack gain or sustain gain accordingly. The output
//! is then blended with the dry input using a wet/dry mix.

use log::warn;

/// Transient shaper structure.
#[derive(Clone, Debug)]
pub struct TransientShaper {
    sample_rate: f32,
    attack_gain_db: f32,
    sustain_gain_db: f32,
    threshold_db: f32,
    hold_ms: f32,
    dry_wet: f32,
    envelope: f32,
    attack_coef: f32,
    release_coef: f32,
    hold_samples: usize,
    hold_counter: usize,
    prev_envelope: f32,
    is_transient: bool,
}

impl TransientShaper {
    /// Create a new transient shaper with reasonable defaults. The default
    /// settings provide a subtle attack boost and little sustain change.
    pub fn new(sample_rate: f32) -> Self {
        let mut shaper = Self {
            sample_rate,
            attack_gain_db: 4.0,
            sustain_gain_db: 0.0,
            threshold_db: -30.0,
            hold_ms: 8.0,
            dry_wet: 0.7,
            envelope: 0.0,
            attack_coef: 0.0,
            release_coef: 0.0,
            hold_samples: 0,
            hold_counter: 0,
            prev_envelope: 0.0,
            is_transient: false,
        };
        shaper.update_coefficients();
        shaper
    }

    /// Process a frame of samples. The input slice is modified in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if sanitize_samples("TransientShaper", samples) {
            return;
        }
        let threshold_linear = db_to_linear(self.threshold_db);
        let attack_gain = db_to_linear(self.attack_gain_db);
        let sustain_gain = db_to_linear(self.sustain_gain_db);
        let wet = self.dry_wet;
        let dry = 1.0 - wet;

        for sample in samples.iter_mut() {
            let input = *sample;
            let abs_input = input.abs();
            // Envelope follower with separate attack/release coefficients.
            let coef = if abs_input > self.envelope {
                self.attack_coef
            } else {
                self.release_coef
            };
            self.envelope = coef * self.envelope + (1.0 - coef) * abs_input;
            // Detect transient by comparing envelope derivative and absolute level.
            let envelope_delta = self.envelope - self.prev_envelope;
            let relative_threshold = (self.envelope * 0.1).max(1e-5);
            if envelope_delta > relative_threshold && self.envelope > threshold_linear {
                self.is_transient = true;
                self.hold_counter = self.hold_samples;
            }
            if self.hold_counter > 0 {
                self.hold_counter -= 1;
            } else {
                self.is_transient = false;
            }
            self.prev_envelope = self.envelope;
            // Choose gain based on whether the current sample is a transient.
            let gain = if self.is_transient {
                attack_gain
            } else {
                sustain_gain
            };
            let output = input * gain;
            *sample = output * wet + input * dry;
        }
    }

    /// Set the attack gain in decibels (0–12 dB). Positive values boost the attack.
    pub fn set_attack_gain(&mut self, db: f32) {
        self.attack_gain_db = db.clamp(0.0, 12.0);
    }

    /// Set the sustain gain in decibels (-12–6 dB). Negative values reduce sustain.
    pub fn set_sustain_gain(&mut self, db: f32) {
        self.sustain_gain_db = db.clamp(-12.0, 6.0);
    }

    /// Set the wet/dry mix (0–1). 1 means fully processed, 0 means bypass.
    pub fn set_dry_wet(&mut self, ratio: f32) {
        self.dry_wet = ratio.clamp(0.0, 1.0);
    }

    /// Reset internal state. Should be called when starting to process new audio.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.prev_envelope = 0.0;
        self.hold_counter = 0;
        self.is_transient = false;
    }

    /// Update time constants. Called automatically during initialization.
    fn update_coefficients(&mut self) {
        let attack_ms = 0.1;
        let release_ms = 200.0;
        self.attack_coef = (-1000.0 / (attack_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
        self.release_coef = (-1000.0 / (release_ms * self.sample_rate.clamp(1.0, f32::MAX))).exp();
        self.hold_samples = ((self.hold_ms * self.sample_rate) / 1000.0).round() as usize;
    }

    /// Set the detection parameters for transients.
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
