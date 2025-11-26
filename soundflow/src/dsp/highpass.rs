//! Simple first‑order high‑pass filter.
//!
//! This filter implements a single pole high‑pass using the standard
//! RC filter equation. It is not as precise as biquad implementations
//! but suffices for low‑frequency rumble suppression.

#[derive(Debug, Clone)]
pub struct Highpass {
    /// Filter coefficient controlling the corner frequency.
    coeff: f32,
    /// Previous input sample for the differentiator section.
    prev_input: f32,
    /// Previous output sample for the integrator section.
    prev_output: f32,
    /// Sample rate in Hz.
    sr: f32,
}

impl Highpass {
    /// Create a new high‑pass filter with the given cutoff and sample rate.
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let mut hp = Highpass {
            coeff: 0.0,
            prev_input: 0.0,
            prev_output: 0.0,
            sr: sample_rate.max(1.0),
        };
        hp.set_cutoff(cutoff_hz);
        hp
    }

    /// Update the cutoff frequency. The provided frequency is clamped
    /// between 1 Hz and the Nyquist frequency. Changing the cutoff does
    /// not reset the internal state.
    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        let cutoff = cutoff_hz.max(1.0).min(self.sr * 0.45);
        // RC high‑pass filter coefficient: alpha = rc / (rc + dt)
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        let dt = 1.0 / self.sr;
        self.coeff = rc / (rc + dt);
    }

    /// Process a buffer of samples in place. The filter maintains internal
    /// state between calls, so consecutive calls will produce continuous
    /// output.
    pub fn process_inplace(&mut self, data: &mut [f32]) {
        for x in data.iter_mut() {
            let input = *x;
            // High‑pass via discrete differentiator/integrator form
            let output = self.coeff * (self.prev_output + input - self.prev_input);
            self.prev_input = input;
            self.prev_output = output;
            *x = output;
        }
    }
}