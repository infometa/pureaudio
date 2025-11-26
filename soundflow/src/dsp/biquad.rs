//! Simple biquad filter implementation inspired by the original soundflow project.
//!
//! A biquad is a second‑order IIR filter that can implement peaking, low‑shelf,
//! high‑shelf and bandpass responses. We use it to build a lightweight EQ for
//! timbre shaping. The filter coefficients are recalculated whenever the
//! frequency, Q or gain is changed. This module is adapted from the
//! `infometa/soundflow` repository but trimmed down for our needs.

use std::f32::consts::PI;

/// The type of biquad filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadType {
    /// Peaking EQ filter that boosts or cuts around a centre frequency.
    Peaking,
    /// Low shelf filter that boosts or cuts below the cutoff frequency.
    LowShelf,
    /// High shelf filter that boosts or cuts above the cutoff frequency.
    HighShelf,
    /// Bandpass filter used internally for envelope detection; gain is ignored.
    BandPass,
}

/// A direct form I biquad filter.
#[derive(Clone, Debug)]
pub struct Biquad {
    kind: BiquadType,
    sample_rate: f32,
    frequency: f32,
    q: f32,
    gain_db: f32,
    // Filter coefficients
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // Delay elements (state)
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Construct a new biquad filter. The `gain_db` parameter is only used for
    /// peaking and shelf filters; it is ignored for bandpass filters. The
    /// frequency and Q are sanitized to prevent numerical issues.
    pub fn new(kind: BiquadType, sample_rate: f32, frequency: f32, q: f32, gain_db: f32) -> Self {
        let mut biquad = Self {
            kind,
            sample_rate,
            frequency: sanitize_frequency(frequency, sample_rate),
            q: sanitize_q(q),
            gain_db,
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        };
        biquad.update_coeffs();
        biquad
    }

    /// Set the gain in decibels. This has no effect for bandpass filters.
    pub fn set_gain_db(&mut self, gain_db: f32) {
        if self.kind == BiquadType::BandPass {
            return;
        }
        // Avoid unnecessary recalculation if the gain is unchanged.
        if (self.gain_db - gain_db).abs() < f32::EPSILON {
            return;
        }
        self.gain_db = gain_db;
        self.update_coeffs();
    }

    /// Reset the internal filter state. This should be called when starting
    /// processing of a new stream to avoid leaking state between files.
    pub fn reset_state(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample through the biquad and return the filtered value.
    pub fn process(&mut self, input: f32) -> f32 {
        // Direct Form I with two feedback terms.
        let out = self.b0 * input + self.z1;
        self.z1 = self.b1 * input + self.z2 - self.a1 * out;
        self.z2 = self.b2 * input - self.a2 * out;
        out
    }

    /// Update the filter coefficients based on the current parameters.
    fn update_coeffs(&mut self) {
        // Clamp parameters to safe ranges.
        self.frequency = sanitize_frequency(self.frequency, self.sample_rate);
        self.q = sanitize_q(self.q);
        let omega = 2.0 * PI * self.frequency / self.sample_rate.max(1.0);
        let sin_w0 = omega.sin();
        let cos_w0 = omega.cos();
        let alpha = sin_w0 / (2.0 * self.q);
        match self.kind {
            BiquadType::Peaking => {
                // Peaking filter uses a boost/cut parameter a (linear).
                let a = db_to_a(self.gain_db);
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
            BiquadType::LowShelf => {
                // Low shelf filter uses slope encoded in q; see RBJ cookbook.
                let a = db_to_a(self.gain_db);
                let s = self.q;
                let sqrt_a = a.sqrt();
                let alpha = shelf_alpha(s, a, sin_w0);
                let beta = 2.0 * sqrt_a * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + beta);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - beta);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + beta;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - beta;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
            BiquadType::HighShelf => {
                let a = db_to_a(self.gain_db);
                let s = self.q;
                let sqrt_a = a.sqrt();
                let alpha = shelf_alpha(s, a, sin_w0);
                let beta = 2.0 * sqrt_a * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + beta);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - beta);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + beta;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - beta;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
            BiquadType::BandPass => {
                // For bandpass, gain_db is ignored.
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
        }
    }

    /// Normalize the filter coefficients by dividing by a0 and store them.
    fn normalize(&mut self, b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) {
        let a0 = if a0.abs() < 1e-12 { 1.0 } else { a0 };
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }
}

/// Clamp the frequency to a reasonable range between 10 Hz and Nyquist.
fn sanitize_frequency(freq: f32, sample_rate: f32) -> f32 {
    let nyquist = sample_rate * 0.49;
    freq.clamp(10.0, nyquist.max(10.0))
}

/// Clamp the Q (quality factor) to prevent extremes.
fn sanitize_q(q: f32) -> f32 {
    q.clamp(0.1, 50.0)
}

/// Convert gain in decibels to amplitude ratio. Uses 40 dB per decade for
/// magnitude; derived from 20*log10(A).
fn db_to_a(gain_db: f32) -> f32 {
    10f32.powf(gain_db / 40.0)
}

/// Helper for shelf filters based on RBJ cookbook. See notes in dynamic_eq.rs.
fn shelf_alpha(slope: f32, a: f32, sin_w0: f32) -> f32 {
    let s = slope.clamp(0.1, 4.0);
    let term = (a + 1.0 / a) * (1.0 / s - 1.0) + 2.0;
    let term = term.max(0.0);
    sin_w0 / 2.0 * term.sqrt()
}
