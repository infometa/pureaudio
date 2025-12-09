use std::f32::consts::PI;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadType {
    Peaking,
    LowShelf,
    HighShelf,
    BandPass,
    LowPass,
    HighPass,
}

#[derive(Clone, Debug)]
pub struct Biquad {
    kind: BiquadType,
    sample_rate: f32,
    frequency: f32,
    q: f32,
    gain_db: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
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

    pub fn set_gain_db(&mut self, gain_db: f32) {
        if self.kind == BiquadType::BandPass {
            return;
        }
        if (self.gain_db - gain_db).abs() < f32::EPSILON {
            return;
        }
        self.gain_db = gain_db;
        self.update_coeffs();
    }

    #[allow(dead_code)]
    pub fn reset_state(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let out = self.b0 * input + self.z1;
        self.z1 = self.b1 * input + self.z2 - self.a1 * out;
        self.z2 = self.b2 * input - self.a2 * out;
        // 防止状态漂移/次正规数
        if !self.z1.is_finite() || self.z1.abs() < 1e-25 {
            self.z1 = 0.0;
        }
        if !self.z2.is_finite() || self.z2.abs() < 1e-25 {
            self.z2 = 0.0;
        }
        out
    }

    fn update_coeffs(&mut self) {
        self.frequency = sanitize_frequency(self.frequency, self.sample_rate);
        self.q = sanitize_q(self.q);
        let omega = 2.0 * PI * self.frequency / self.sample_rate.max(1.0);
        let sin_w0 = omega.sin();
        let cos_w0 = omega.cos();
        let alpha = sin_w0 / (2.0 * self.q);
        match self.kind {
            BiquadType::Peaking => {
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
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
            BiquadType::LowPass => {
                let b0 = (1.0 - cos_w0) * 0.5;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
            BiquadType::HighPass => {
                let b0 = (1.0 + cos_w0) * 0.5;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                self.normalize(b0, b1, b2, a0, a1, a2);
            }
        }
    }

    fn normalize(&mut self, b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) {
        let a0 = if a0.abs() < 1e-12 { 1.0 } else { a0 };
        let mut na1 = a1 / a0;
        let mut na2 = a2 / a0;
        if na1.abs() >= 1.999 || na2.abs() >= 0.999 {
            log::warn!("Biquad 系数接近不稳定，已钳制 (a1={:.3}, a2={:.3})", na1, na2);
            na1 = na1.clamp(-1.999, 1.999);
            na2 = na2.clamp(-0.999, 0.999);
        }
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = na1;
        self.a2 = na2;
    }
}

fn sanitize_frequency(freq: f32, sample_rate: f32) -> f32 {
    let nyquist = sample_rate * 0.49;
    freq.clamp(10.0, nyquist.max(10.0))
}

fn sanitize_q(q: f32) -> f32 {
    q.clamp(0.1, 50.0)
}

fn db_to_a(gain_db: f32) -> f32 {
    10f32.powf(gain_db / 40.0)
}

fn shelf_alpha(slope: f32, a: f32, sin_w0: f32) -> f32 {
    let s = slope.clamp(0.1, 4.0);
    let term = (a + 1.0 / a) * (1.0 / s - 1.0) + 2.0;
    let term = term.max(0.0);
    sin_w0 / 2.0 * term.sqrt()
}
