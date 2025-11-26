//! Simple multi‑band equalizer using biquad filters.
//!
//! This module provides a simplified equalizer for timbre shaping. Unlike the
//! dynamic EQ in the original `soundflow` project, this EQ uses fixed
//! coefficients and gains without envelope detection or compression. Each
//! preset defines a set of bands (low shelf, peak and high shelf) with
//! specific frequency, Q and static gain. The EQ can mix the wet (filtered)
//! signal with the dry (original) signal.

use super::biquad::{Biquad, BiquadType};

/// A single EQ band with a biquad filter and static gain in dB.
#[derive(Clone, Debug)]
struct EqBand {
    filter: Biquad,
}

impl EqBand {
    fn new(kind: BiquadType, sample_rate: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let mut filter = Biquad::new(kind, sample_rate, freq, q, 0.0);
        // Apply static gain for non‑bandpass filters. For bandpass we ignore gain.
        filter.set_gain_db(gain_db);
        Self { filter }
    }

    fn process_inplace(&mut self, samples: &mut [f32]) {
        for v in samples.iter_mut() {
            *v = self.filter.process(*v);
        }
    }
}

/// Available EQ presets. Each preset defines five bands tuned for typical
/// speech enhancement scenarios. These presets are loosely based on the
/// dynamic EQ presets in the original project.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqPreset {
    /// A balanced preset suitable for most voices.
    Natural,
    /// A broadcast preset with thick low end and bright highs.
    Broadcast,
    /// A meeting preset focusing on intelligibility and de‑essing.
    Meeting,
}

impl EqPreset {
    /// Return the band definitions for this preset. Each band is a tuple of
    /// `(filter_type, frequency_hz, q, gain_db)`.
    fn bands(self) -> &'static [(BiquadType, f32, f32, f32)] {
        match self {
            EqPreset::Natural => &NATURAL_BANDS,
            EqPreset::Broadcast => &BROADCAST_BANDS,
            EqPreset::Meeting => &MEETING_BANDS,
        }
    }
}

// Define the band settings for each preset. These values were derived from
// the static gains in the original dynamic EQ presets and tuned manually.
const NATURAL_BANDS: [(BiquadType, f32, f32, f32); 5] = [
    // Low shelf to clean up low end slightly.
    (BiquadType::LowShelf, 100.0, 0.7, -0.5),
    // Peak to reduce muddiness around 260 Hz.
    (BiquadType::Peaking, 260.0, 1.2, -0.5),
    // Peak to enhance clarity around 3 kHz.
    (BiquadType::Peaking, 3100.0, 1.1, 0.8),
    // Peak to control sibilance around 7 kHz.
    (BiquadType::Peaking, 7000.0, 2.0, -1.0),
    // High shelf to add air above 12 kHz.
    (BiquadType::HighShelf, 12000.0, 0.8, 0.5),
];

const BROADCAST_BANDS: [(BiquadType, f32, f32, f32); 5] = [
    // Low shelf for chest and warmth.
    (BiquadType::LowShelf, 80.0, 0.7, 2.0),
    // Peak for low mid chest.
    (BiquadType::Peaking, 180.0, 1.0, 3.0),
    // Peak to reduce muddiness.
    (BiquadType::Peaking, 450.0, 1.5, 0.0),
    // Peak to enhance clarity.
    (BiquadType::Peaking, 3500.0, 1.2, 2.0),
    // High shelf to boost air.
    (BiquadType::HighShelf, 8000.0, 0.7, 4.5),
];

const MEETING_BANDS: [(BiquadType, f32, f32, f32); 5] = [
    // Low shelf to tame low noise.
    (BiquadType::LowShelf, 110.0, 0.8, -2.0),
    // Peak to reduce low mids.
    (BiquadType::Peaking, 230.0, 1.0, -1.0),
    // Peak to focus voice.
    (BiquadType::Peaking, 2700.0, 1.3, 0.5),
    // Peak to tame sibilance.
    (BiquadType::Peaking, 7200.0, 2.1, -2.0),
    // High shelf to add some sparkle.
    (BiquadType::HighShelf, 11000.0, 0.9, 0.3),
];

/// A multi‑band equalizer with fixed bands and adjustable wet/dry mix.
#[derive(Clone, Debug)]
pub struct Eq {
    bands: Vec<EqBand>,
    dry_wet: f32,
}

impl Eq {
    /// Create a new EQ using the given preset and sample rate. The wet/dry
    /// parameter controls how much of the filtered signal is mixed in. A value
    /// of 1.0 means fully wet (only filtered), while 0.0 means bypass.
    pub fn new(sample_rate: f32, preset: EqPreset, dry_wet: f32) -> Self {
        let mut bands = Vec::new();
        for (kind, freq, q, gain_db) in preset.bands() {
            bands.push(EqBand::new(*kind, sample_rate, *freq, *q, *gain_db));
        }
        Self {
            bands,
            dry_wet: dry_wet.clamp(0.0, 1.0),
        }
    }

    /// Process the samples in place. If the wet amount is 0, the audio is left
    /// untouched. Otherwise, each band is applied sequentially and then the
    /// result is blended with the original according to `dry_wet`.
    pub fn process_inplace(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        let wet = self.dry_wet;
        if wet <= 0.0 {
            return;
        }
        // Copy original samples to a buffer for dry/wet mixing.
        let mut dry = samples.to_vec();
        // Apply each band in sequence to `samples`.
        for band in self.bands.iter_mut() {
            band.process_inplace(samples);
        }
        // Mix dry and wet.
        let dry_ratio = 1.0 - wet;
        for (out, in_dry) in samples.iter_mut().zip(dry.iter()) {
            *out = *out * wet + *in_dry * dry_ratio;
        }
    }

    /// Set a new wet/dry mix.
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.dry_wet = dry_wet.clamp(0.0, 1.0);
    }
}
