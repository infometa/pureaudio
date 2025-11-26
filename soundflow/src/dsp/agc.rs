//! Automatic gain control (AGC) implementation.
//!
//! This AGC tracks the short‑term RMS of a signal and applies a gain
//! to push it towards a target level. It constrains the maximum gain
//! increase and attenuation to avoid pumping artifacts. The smoothing
//! factor controls how quickly the gain changes over time.

#[derive(Debug, Clone)]
pub struct Agc {
    /// Desired RMS level in dBFS (negative values).
    target_dbfs: f32,
    /// Maximum gain increase allowed (dB). Limits amplification of quiet signals.
    max_gain_db: f32,
    /// Maximum attenuation allowed (dB). Limits how far loud signals are reduced.
    max_atten_db: f32,
    /// Current linear gain factor.
    gain_linear: f32,
    /// Smoothing factor for gain updates (0.0 .. 1.0). Higher values respond more slowly.
    smoothing: f32,
}

impl Agc {
    /// Create a new AGC with default parameters.
    ///
    /// The default target is −20 dBFS, with ±15 dB of gain/attenuation and
    /// a smoothing factor of 0.1. These defaults can be tuned by the caller.
    pub fn new() -> Self {
        Self {
            target_dbfs: -20.0,
            max_gain_db: 15.0,
            max_atten_db: 15.0,
            gain_linear: 1.0,
            smoothing: 0.1,
        }
    }

    /// Set the target RMS level in dBFS. Values should typically be negative.
    pub fn set_target_dbfs(&mut self, value: f32) {
        self.target_dbfs = value;
    }

    /// Set the maximum gain increase in dB.
    pub fn set_max_gain_db(&mut self, value: f32) {
        self.max_gain_db = value.max(0.0);
    }

    /// Set the maximum attenuation in dB.
    pub fn set_max_atten_db(&mut self, value: f32) {
        self.max_atten_db = value.max(0.0);
    }

    /// Set the smoothing factor for gain updates. Range [0.0, 1.0]. A
    /// larger factor yields slower gain changes.
    pub fn set_smoothing(&mut self, value: f32) {
        self.smoothing = value.clamp(0.0, 1.0);
    }

    /// Apply AGC to a buffer of samples. This computes the buffer RMS,
    /// determines the difference from the target, limits it with max gain
    /// and attenuation bounds, then applies a smoothed gain to the buffer.
    pub fn process_inplace(&mut self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }
        // Compute RMS level in dBFS. Avoid log of zero.
        let mut sum = 0.0f32;
        for &v in data.iter() {
            sum += v * v;
        }
        let rms = (sum / (data.len() as f32)).sqrt().max(1e-9);
        let rms_db = 20.0 * rms.log10();
        // Desired gain in dB.
        let mut gain_db = self.target_dbfs - rms_db;
        gain_db = gain_db.clamp(-self.max_atten_db, self.max_gain_db);
        let desired_gain = 10.0_f32.powf(gain_db / 20.0);
        // Smooth the gain toward the desired value.
        self.gain_linear =
            self.gain_linear * (1.0 - self.smoothing) + desired_gain * self.smoothing;
        for v in data.iter_mut() {
            *v *= self.gain_linear;
        }
    }
}