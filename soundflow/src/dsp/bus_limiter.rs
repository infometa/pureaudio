//! A simple bus limiter to prevent clipping and tame peaks.
//!
//! This limiter tracks the peak envelope of the signal and computes a gain to
//! keep the envelope below a threshold. When the envelope exceeds the
//! threshold, the gain is reduced proportionally. Otherwise, the gain
//! gradually returns to unity at a rate controlled by the release coefficient.
//! After applying the adaptive gain, a hard clipping stage ensures that no
//! sample exceeds the absolute range ±1.0.

#[derive(Debug, Clone)]
pub struct BusLimiter {
    /// Target peak level. Values above this level will be attenuated.
    threshold: f32,
    /// Release coefficient for smoothing the envelope and gain recovery (0.0–1.0).
    release: f32,
    /// Running envelope of the absolute signal.
    envelope: f32,
    /// Current gain applied to the signal.
    gain: f32,
}

impl BusLimiter {
    /// Create a new bus limiter with a default threshold of 0.95 and release of 0.999.
    pub fn new() -> Self {
        Self {
            threshold: 0.95,
            release: 0.999,
            envelope: 0.0,
            gain: 1.0,
        }
    }

    /// Set the limiting threshold. The value should be in (0.0, 1.0].
    pub fn set_threshold(&mut self, thresh: f32) {
        self.threshold = thresh.clamp(0.0, 1.0);
    }

    /// Set the release coefficient. Higher values result in slower recovery.
    pub fn set_release(&mut self, rel: f32) {
        self.release = rel.clamp(0.0, 1.0);
    }

    /// Process a slice of samples in place. The limiter maintains its state across calls.
    pub fn process_inplace(&mut self, data: &mut [f32]) {
        for x in data.iter_mut() {
            let abs_x = x.abs();
            // Update the envelope using a simple peak detector with release.
            self.envelope = if abs_x > self.envelope {
                abs_x
            } else {
                self.envelope * self.release
            };
            // Compute the desired gain. If the envelope exceeds the threshold,
            // scale down proportionally; otherwise slowly return to unity.
            let desired_gain = if self.envelope > self.threshold && self.envelope > 0.0 {
                self.threshold / self.envelope
            } else {
                1.0
            };
            // Smooth the gain change slightly to avoid sudden jumps.
            // Here we blend 90% of the current gain with 10% of the target.
            self.gain = self.gain * 0.9 + desired_gain * 0.1;
            *x *= self.gain;
            // Hard clip as a last resort to guarantee bounds.
            if *x > 1.0 {
                *x = 1.0;
            } else if *x < -1.0 {
                *x = -1.0;
            }
        }
    }
}