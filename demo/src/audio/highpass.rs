use log::warn;

pub struct HighpassFilter {
    cutoff_hz: f32,
    sample_rate: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl HighpassFilter {
    pub fn new(sample_rate: f32) -> Self {
        assert!(
            sample_rate >= 1000.0,
            "HighpassFilter requires sample_rate >= 1 kHz"
        );
        let mut filter = Self {
            cutoff_hz: 60.0,
            sample_rate,
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        };
        filter.update_coefficients();
        filter
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        let nyq = (self.sample_rate / 2.0).max(1.0);
        let safe_max = nyq * 0.5;
        self.cutoff_hz = cutoff_hz.clamp(20.0, safe_max);
        self.update_coefficients();
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if sanitize_samples("HighpassFilter", samples) {
            return;
        }
        for sample in samples.iter_mut() {
            let input = *sample;
            let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;

            self.x2 = self.x1;
            self.x1 = input;
            self.y2 = self.y1;
            self.y1 = output;

            *sample = output;
        }
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    fn update_coefficients(&mut self) {
        let omega = 2.0 * std::f32::consts::PI * self.cutoff_hz / self.sample_rate.max(1.0);
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707); // Q = 0.707 (Butterworth)

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
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
