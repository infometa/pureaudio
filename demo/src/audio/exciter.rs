/// A lightweight harmonic exciter that targets upper bands only.
/// It applies a simple first-order highpass to isolate high frequencies, then
/// adds gentle saturation and mixes back with the dry signal.
pub struct HarmonicExciter {
    sample_rate: f32,
    cutoff_hz: f32,
    drive: f32,
    mix: f32,
    prev_in: f32,
    prev_hp: f32,
    alpha: f32,
}

impl HarmonicExciter {
    pub fn new(sample_rate: f32, cutoff_hz: f32, drive: f32, mix: f32) -> Self {
        let mut exciter = Self {
            sample_rate,
            cutoff_hz,
            drive: drive.clamp(1.0, 3.0),
            mix: mix.clamp(0.0, 1.0),
            prev_in: 0.0,
            prev_hp: 0.0,
            alpha: 0.0,
        };
        exciter.update_coeff();
        exciter
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn mix(&self) -> f32 {
        self.mix
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if samples.is_empty() || self.mix <= 0.0 {
            return;
        }
        let wet = self.mix;
        let dry = 1.0 - wet;
        let drive = self.drive;
        let alpha = self.alpha;
        for sample in samples.iter_mut() {
            // Highpass to focus excitation on upper band
            let hp = alpha * (self.prev_hp + *sample - self.prev_in);
            self.prev_in = *sample;
            self.prev_hp = hp;
            // Gentle saturation on high band only
            let excited = (hp * drive).tanh() / drive;
            // Mix: add excited high content back to original
            *sample = *sample * dry + (*sample + excited) * wet;
        }
    }

    fn update_coeff(&mut self) {
        // First-order highpass coefficient
        let rc = 1.0 / (2.0 * std::f32::consts::PI * self.cutoff_hz.max(20.0));
        let dt = 1.0 / self.sample_rate.max(1.0);
        self.alpha = rc / (rc + dt);
    }
}
