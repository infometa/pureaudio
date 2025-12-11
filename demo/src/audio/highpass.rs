use log::warn;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct HighpassFilter {
    cutoff_hz: f32,
    sample_rate: f32,
    q: f32,
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
            cutoff_hz: 50.0,
            sample_rate,
            q: 1.0,
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
        let safe_max = nyq * 0.45;
        // 过低的截止频率会导致系数接近 1，数值不稳定；限制一个安全下限。
        let safe_min = 20.0;
        let clamped = cutoff_hz.clamp(safe_min, safe_max);
        if clamped != cutoff_hz && cutoff_hz > 0.0 {
            warn!(
                "HighpassFilter 截止频率过低 ({:.1} Hz)，已提升到 {:.1} Hz 以保证稳定",
                cutoff_hz, clamped
            );
        }
        self.cutoff_hz = clamped;
        self.update_coefficients();
    }

    #[allow(dead_code)]
    pub fn set_q(&mut self, q: f32) {
        // 限制Q值范围以避免高通滤波器产生过强共振峰
        // 专业音频建议：Q=0.707 (Butterworth), 最大不超过1.5
        self.q = q.clamp(0.5, 1.5);
        self.update_coefficients();
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        if self.cutoff_hz <= 0.0 {
            return;
        }
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
            
            // 状态防护：防止次正规数累积导致性能下降
            // 阈值设为1e-10，比原来的1e-25更合理（f32精度约7位有效数字）
            if self.y1.abs() < 1e-10 {
                self.y1 = 0.0;
            }
            if self.y2.abs() < 1e-10 {
                self.y2 = 0.0;
            }

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
        if self.cutoff_hz <= 0.0 {
            // Bypass: unity transfer
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }
        let omega = 2.0 * std::f32::consts::PI * self.cutoff_hz / self.sample_rate.max(1.0);
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * self.q.max(1e-6)); // 可调 Q，默认 1.0

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        let a0 = a0;
        let mut na1 = a1 / a0;
        let mut na2 = a2 / a0;
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        
        // 严格的IIR稳定性检查：除了|a1|<2, |a2|<1外，还需满足极点在单位圆内
        // 标准条件：|a2| < 1 且 |a1| < 1 + a2
        let is_unstable = na2.abs() >= 0.999 || na1.abs() >= (1.0 + na2 - 0.001);
        
        if is_unstable {
            static HP_UNSTABLE_COUNT: AtomicUsize = AtomicUsize::new(0);
            let count = HP_UNSTABLE_COUNT.fetch_add(1, Ordering::Relaxed);
            
            // 使用限流日志：每100次警告一次，避免阻塞音频线程
            if count % 100 == 0 {
                warn!(
                    "HighpassFilter 系数不稳定 (第{}次): cutoff={:.1}Hz, Q={:.2}, a1={:.3}, a2={:.3}",
                    count, self.cutoff_hz, self.q, na1, na2
                );
            }
            
            na1 = na1.clamp(-1.998, 1.998);
            na2 = na2.clamp(-0.998, 0.998);
        }
        self.a1 = na1;
        self.a2 = na2;
    }
}

fn sanitize_samples(_tag: &str, samples: &mut [f32]) -> bool {
    let mut found = false;
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            found = true;
        }
    }
    if found {
        HIGHPASS_SANITIZE_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    found
}

static HIGHPASS_SANITIZE_COUNT: AtomicUsize = AtomicUsize::new(0);
