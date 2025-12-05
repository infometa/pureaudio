use super::biquad::{Biquad, BiquadType};
use super::envelope::{smoothing_coeff, EnvelopeDetector};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandMode {
    Downward,
    Upward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterKind {
    Peak,
    LowShelf,
    HighShelf,
}

impl BandMode {
    pub const fn all() -> [Self; 2] {
        [BandMode::Downward, BandMode::Upward]
    }
}

impl fmt::Display for BandMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandMode::Downward => write!(f, "下行压缩"),
            BandMode::Upward => write!(f, "上行扩展"),
        }
    }
}

impl FilterKind {
    pub const fn all() -> [Self; 3] {
        [
            FilterKind::Peak,
            FilterKind::LowShelf,
            FilterKind::HighShelf,
        ]
    }
}

impl fmt::Display for FilterKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterKind::Peak => write!(f, "峰值"),
            FilterKind::LowShelf => write!(f, "低搁架"),
            FilterKind::HighShelf => write!(f, "高搁架"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BandSettings {
    pub label: &'static str,
    pub frequency_hz: f32,
    pub q: f32,
    pub detector_q: f32,
    pub threshold_db: f32,
    pub ratio: f32,
    pub max_gain_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub mode: BandMode,
    pub filter: FilterKind,
    pub makeup_db: f32,
    pub static_gain_db: f32,
}

impl BandSettings {
    pub fn detector_q(&self) -> f32 {
        if self.detector_q <= 0.0 {
            self.q
        } else {
            self.detector_q
        }
    }
}

#[derive(Clone, Debug)]
pub struct DynamicBand {
    settings: BandSettings,
    detector: Biquad,
    filter: Biquad,
    envelope: EnvelopeDetector,
    sample_rate: f32,
    current_gain_db: f32,
    user_gain_db: f32,
    rms_ema: f32,
}

impl DynamicBand {
    pub fn new(sample_rate: f32, settings: BandSettings) -> Self {
        let detector = Biquad::new(
            BiquadType::BandPass,
            sample_rate,
            settings.frequency_hz,
            settings.detector_q(),
            0.0,
        );
        let filter = create_filter(sample_rate, &settings);
        Self {
            settings,
            detector,
            filter,
            envelope: EnvelopeDetector::new(sample_rate, settings.attack_ms, settings.release_ms),
            sample_rate,
            current_gain_db: 0.0,
            user_gain_db: settings.static_gain_db,
            rms_ema: 0.0,
        }
    }

    pub fn configure(&mut self, settings: BandSettings) {
        *self = Self::new(self.sample_rate, settings);
    }

    pub fn label(&self) -> &'static str {
        self.settings.label
    }

    pub fn analyze(&mut self, block: &[f32]) -> f32 {
        if block.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0;
        for &sample in block {
            let filtered = self.detector.process(sample);
            acc += filtered * filtered;
        }
        let mean_sq = acc / block.len().max(1) as f32;
        mean_sq.max(1e-20).sqrt()
    }

    pub fn update(&mut self, rms: f32, block_len: usize) {
        if block_len == 0 {
            return;
        }
        // 块级 RMS 做 EMA 平滑，降低对块长的敏感度
        let alpha_rms = smoothing_coeff(50.0, block_len, self.sample_rate); // ~50ms 平滑
        if self.rms_ema == 0.0 {
            self.rms_ema = rms;
        } else {
            self.rms_ema += alpha_rms * (rms - self.rms_ema);
        }
        let level_db = linear_to_db(self.rms_ema);
        let env_db = self.envelope.process(level_db, block_len);
        let target = self.target_gain(env_db);
        let coeff = if target > self.current_gain_db {
            smoothing_coeff(self.settings.attack_ms, block_len, self.sample_rate)
        } else {
            smoothing_coeff(self.settings.release_ms, block_len, self.sample_rate)
        };
        self.current_gain_db += coeff * (target - self.current_gain_db);
        self.refresh_filter_gain();
    }

    pub fn apply(&mut self, samples: &mut [f32]) {
        for sample in samples {
            *sample = self.filter.process(*sample);
        }
    }

    pub fn gain_db(&self) -> f32 {
        self.total_gain()
    }

    fn target_gain(&self, level_db: f32) -> f32 {
        let ratio = self.settings.ratio.max(1.0001);
        match self.settings.mode {
            BandMode::Downward => {
                if level_db > self.settings.threshold_db {
                    let over = level_db - self.settings.threshold_db;
                    let reduction = over * (1.0 - 1.0 / ratio);
                    -(reduction.clamp(0.0, self.settings.max_gain_db)) + self.settings.makeup_db
                } else {
                    self.settings.makeup_db
                }
            }
            BandMode::Upward => {
                if level_db < self.settings.threshold_db {
                    let diff = self.settings.threshold_db - level_db;
                    let boost = diff * (1.0 - 1.0 / ratio);
                    boost.clamp(0.0, self.settings.max_gain_db) + self.settings.makeup_db
                } else {
                    self.settings.makeup_db
                }
            }
        }
    }

    pub fn set_user_gain(&mut self, gain_db: f32) {
        self.user_gain_db = gain_db;
        self.refresh_filter_gain();
    }

    pub fn set_frequency(&mut self, frequency_hz: f32) {
        let nyquist = self.sample_rate * 0.5 - 100.0;
        let upper = nyquist.max(200.0);
        self.settings.frequency_hz = frequency_hz.clamp(20.0, upper);
        self.rebuild_filter();
        self.rebuild_detector();
    }

    pub fn set_q(&mut self, q: f32) {
        self.settings.q = q.clamp(0.1, 5.0);
        self.rebuild_filter();
    }

    pub fn set_detector_q(&mut self, detector_q: f32) {
        self.settings.detector_q = detector_q.clamp(0.1, 5.0);
        self.rebuild_detector();
    }

    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.settings.threshold_db = threshold_db.clamp(-60.0, 0.0);
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        self.settings.ratio = ratio.clamp(1.0, 10.0);
    }

    pub fn set_max_gain(&mut self, max_gain_db: f32) {
        self.settings.max_gain_db = max_gain_db.clamp(0.0, 20.0);
    }

    pub fn set_attack(&mut self, attack_ms: f32) {
        self.settings.attack_ms = attack_ms.clamp(1.0, 100.0);
        self.envelope = EnvelopeDetector::new(
            self.sample_rate,
            self.settings.attack_ms,
            self.settings.release_ms,
        );
    }

    pub fn set_release(&mut self, release_ms: f32) {
        self.settings.release_ms = release_ms.clamp(10.0, 500.0);
        self.envelope = EnvelopeDetector::new(
            self.sample_rate,
            self.settings.attack_ms,
            self.settings.release_ms,
        );
    }

    pub fn set_makeup_gain(&mut self, makeup_db: f32) {
        self.settings.makeup_db = makeup_db.clamp(-12.0, 12.0);
    }

    pub fn set_mode(&mut self, mode: BandMode) {
        self.settings.mode = mode;
    }

    pub fn set_filter_kind(&mut self, filter: FilterKind) {
        self.settings.filter = filter;
        self.rebuild_filter();
    }

    fn total_gain(&self) -> f32 {
        (self.current_gain_db + self.user_gain_db).clamp(-20.0, 20.0)
    }

    fn refresh_filter_gain(&mut self) {
        self.filter.set_gain_db(self.total_gain());
    }

    fn rebuild_filter(&mut self) {
        let kind = match self.settings.filter {
            FilterKind::Peak => BiquadType::Peaking,
            FilterKind::LowShelf => BiquadType::LowShelf,
            FilterKind::HighShelf => BiquadType::HighShelf,
        };
        self.filter = Biquad::new(
            kind,
            self.sample_rate,
            self.settings.frequency_hz,
            self.settings.q,
            0.0,
        );
        self.refresh_filter_gain();
    }

    fn rebuild_detector(&mut self) {
        self.detector = Biquad::new(
            BiquadType::BandPass,
            self.sample_rate,
            self.settings.frequency_hz,
            self.settings.detector_q(),
            0.0,
        );
    }
}

fn linear_to_db(value: f32) -> f32 {
    20.0 * value.max(1e-5).log10()
}

fn create_filter(sample_rate: f32, settings: &BandSettings) -> Biquad {
    let kind = match settings.filter {
        FilterKind::Peak => BiquadType::Peaking,
        FilterKind::LowShelf => BiquadType::LowShelf,
        FilterKind::HighShelf => BiquadType::HighShelf,
    };
    Biquad::new(kind, sample_rate, settings.frequency_hz, settings.q, 0.0)
}
