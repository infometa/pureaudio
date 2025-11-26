use super::dynamic_band::{BandMode, DynamicBand, FilterKind};
use super::presets::{EqPresetKind, MAX_EQ_BANDS};

// Soft limiter is kept only as a last resort inside EQ; set higher to avoid unnecessary compression
const SOFT_LIMIT_THRESHOLD: f32 = 0.97;
const SOFT_LIMIT_CEILING: f32 = 0.995;

#[derive(Debug, Clone)]
pub enum EqControl {
    SetEnabled(bool),
    SetPreset(EqPresetKind),
    SetDryWet(f32),
    SetBandGain(usize, f32),
    SetBandFrequency(usize, f32),
    SetBandQ(usize, f32),
    SetBandDetectorQ(usize, f32),
    SetBandThreshold(usize, f32),
    SetBandRatio(usize, f32),
    SetBandMaxGain(usize, f32),
    SetBandAttack(usize, f32),
    SetBandRelease(usize, f32),
    SetBandMakeup(usize, f32),
    SetBandMode(usize, BandMode),
    SetBandFilter(usize, FilterKind),
}

#[derive(Debug, Clone, Copy)]
pub struct EqProcessMetrics {
    pub gain_db: [f32; MAX_EQ_BANDS],
    pub enabled: bool,
}

impl Default for EqProcessMetrics {
    fn default() -> Self {
        Self {
            gain_db: [0.0; MAX_EQ_BANDS],
            enabled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicEq {
    sample_rate: f32,
    dry_wet: f32,
    enabled: bool,
    preset: EqPresetKind,
    bands: Vec<DynamicBand>,
    analysis_buf: Vec<f32>,
    user_band_gains: Vec<f32>,
}

impl DynamicEq {
    pub fn new(sample_rate: f32, preset: EqPresetKind) -> Self {
        let mut eq = Self {
            sample_rate,
            dry_wet: preset.default_mix(),
            enabled: true,
            preset,
            bands: Vec::new(),
            analysis_buf: Vec::new(),
            user_band_gains: vec![0.0; MAX_EQ_BANDS],
        };
        eq.load_preset_gains();
        eq.rebuild_bands();
        eq
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn dry_wet(&self) -> f32 {
        self.dry_wet
    }

    pub fn preset(&self) -> EqPresetKind {
        self.preset
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.dry_wet = dry_wet.clamp(0.0, 1.0);
    }

    pub fn apply_preset(&mut self, preset: EqPresetKind) {
        if self.preset == preset {
            return;
        }
        self.preset = preset;
        self.load_preset_gains();
        self.rebuild_bands();
    }

    pub fn set_band_gain(&mut self, band_idx: usize, gain_db: f32) {
        if band_idx >= MAX_EQ_BANDS {
            return;
        }
        let gain = gain_db.clamp(-12.0, 12.0);
        self.user_band_gains[band_idx] = gain;
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_user_gain(gain);
        }
    }

    pub fn set_band_frequency(&mut self, band_idx: usize, frequency_hz: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_frequency(frequency_hz);
        }
    }

    pub fn set_band_q(&mut self, band_idx: usize, q: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_q(q);
        }
    }

    pub fn set_band_detector_q(&mut self, band_idx: usize, detector_q: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_detector_q(detector_q);
        }
    }

    pub fn set_band_threshold(&mut self, band_idx: usize, threshold_db: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_threshold(threshold_db);
        }
    }

    pub fn set_band_ratio(&mut self, band_idx: usize, ratio: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_ratio(ratio);
        }
    }

    pub fn set_band_max_gain(&mut self, band_idx: usize, max_gain_db: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_max_gain(max_gain_db);
        }
    }

    pub fn set_band_attack(&mut self, band_idx: usize, attack_ms: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_attack(attack_ms);
        }
    }

    pub fn set_band_release(&mut self, band_idx: usize, release_ms: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_release(release_ms);
        }
    }

    pub fn set_band_makeup(&mut self, band_idx: usize, makeup_db: f32) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_makeup_gain(makeup_db);
        }
    }

    pub fn set_band_mode(&mut self, band_idx: usize, mode: BandMode) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_mode(mode);
        }
    }

    pub fn set_band_filter(&mut self, band_idx: usize, filter: FilterKind) {
        if let Some(band) = self.bands.get_mut(band_idx) {
            band.set_filter_kind(filter);
        }
    }

    pub fn process_block(&mut self, samples: &mut [f32]) -> EqProcessMetrics {
        let mut metrics = EqProcessMetrics::default();
        metrics.enabled = self.enabled && !self.bands.is_empty();
        if !metrics.enabled || samples.is_empty() {
            return metrics;
        }
        let len = samples.len();
        if self.analysis_buf.len() < len {
            self.analysis_buf.resize(len, 0.0);
        }
        self.analysis_buf[..len].copy_from_slice(samples);
        for (idx, band) in self.bands.iter_mut().enumerate() {
            let rms = band.analyze(&self.analysis_buf[..len]);
            band.update(rms, len);
            if idx < MAX_EQ_BANDS {
                metrics.gain_db[idx] = band.gain_db();
            }
        }
        for band in self.bands.iter_mut() {
            band.apply(samples);
        }
        let peak = samples.iter().fold(0.0f32, |acc, sample| acc.max(sample.abs()));
        if peak > 1.0 {
            let scale = 0.85 / peak;
            for sample in samples.iter_mut() {
                *sample *= scale;
            }
            log::warn!(
                "EQ 输出过载 ({:.2}x)，自动降低 {:.1} dB",
                peak,
                20.0 * (1.0 / scale).log10()
            );
        }
        if should_soft_limit(samples) {
            apply_soft_limiter(samples);
        }
        let wet = self.dry_wet;
        if wet < 0.999 {
            let dry = 1.0 - wet;
            for (idx, sample) in samples.iter_mut().enumerate().take(len) {
                *sample = *sample * wet + self.analysis_buf[idx] * dry;
            }
            if should_soft_limit(samples) {
                apply_soft_limiter(samples);
            }
        }
        metrics
    }

    fn rebuild_bands(&mut self) {
        let preset = self.preset.preset();
        self.bands = preset
            .bands
            .iter()
            .cloned()
            .map(|cfg| DynamicBand::new(self.sample_rate, cfg))
            .collect();
        for (idx, band) in self.bands.iter_mut().enumerate() {
            let gain = self.user_band_gains.get(idx).copied().unwrap_or(0.0);
            band.set_user_gain(gain);
        }
    }

    fn load_preset_gains(&mut self) {
        let preset = self.preset.preset();
        for (idx, band) in preset.bands.iter().enumerate() {
            if idx < self.user_band_gains.len() {
                self.user_band_gains[idx] = band.static_gain_db;
            }
        }
    }
}

fn apply_soft_limiter(samples: &mut [f32]) {
    for sample in samples.iter_mut() {
        *sample = soft_clip(*sample, SOFT_LIMIT_THRESHOLD, SOFT_LIMIT_CEILING);
    }
}

fn should_soft_limit(samples: &[f32]) -> bool {
    samples.iter().any(|s| s.abs() > SOFT_LIMIT_THRESHOLD)
}

fn soft_clip(x: f32, threshold: f32, ceiling: f32) -> f32 {
    let abs_x = x.abs();
    if abs_x <= threshold {
        x
    } else if abs_x < ceiling {
        let t = (abs_x - threshold) / (ceiling - threshold);
        let soft = threshold + (ceiling - threshold) * (3.0 * t.powi(2) - 2.0 * t.powi(3));
        x.signum() * soft
    } else {
        x.signum() * ceiling
    }
}
