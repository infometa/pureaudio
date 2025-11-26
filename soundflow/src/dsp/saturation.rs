//! Simple saturation effect adapted from the original soundflow project.
//!
//! Saturation introduces harmonic distortion by pushing the signal through a
//! non‑linear transfer function (tanh) and then mixes the distorted signal
//! back with the original. A makeup gain (in dB) can compensate for level
//! loss, and compensation can be enabled to equalize the magnitude of the
//! dry and wet signals.

use log::warn;

/// Saturation effect structure.
#[derive(Clone, Debug)]
pub struct Saturation {
    drive: f32,
    makeup_db: f32,
    mix: f32,
    compensate: bool,
}

impl Saturation {
    /// Create a new saturation with default settings.
    pub fn new() -> Self {
        Self {
            drive: 1.2,
            makeup_db: -0.5,
            mix: 1.0,
            compensate: true,
        }
    }

    /// Set the drive (0.5–2.0). Higher drive increases distortion.
    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.clamp(0.5, 2.0);
    }

    /// Set the makeup gain in dB (-12–6). Positive values boost the output.
    pub fn set_makeup(&mut self, makeup_db: f32) {
        self.makeup_db = makeup_db.clamp(-12.0, 6.0);
    }

    /// Set the wet/dry mix (0–1). 1 means fully saturated.
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Enable or disable compensation. When enabled, the output level is
    /// approximately equal to the input level for drive values around 1.
    pub fn set_compensate(&mut self, enable: bool) {
        self.compensate = enable;
    }

    /// Process a block of samples in place.
    pub fn process(&self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        if sanitize_samples("Saturation", samples) {
            return;
        }
        let drive = self.drive;
        let makeup = db_to_linear(self.makeup_db);
        let wet_ratio = self.mix;
        let dry_ratio = 1.0 - wet_ratio;
        for sample in samples.iter_mut() {
            let dry = *sample;
            let driven = if self.compensate {
                (dry * drive).tanh() / drive
            } else {
                (dry * drive).tanh()
            };
            let driven = driven * makeup;
            *sample = driven * wet_ratio + dry * dry_ratio;
        }
    }
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
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
        warn!("{tag} 检测到非法音频数据 (NaN/Inf)，跳过本帧处理");
    }
    found
}
