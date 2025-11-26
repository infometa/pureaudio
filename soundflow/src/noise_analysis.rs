use df::Complex32;
use ndarray::prelude::*;

/// Features computed from a noise-only spectrogram segment.
///
/// * `energy_db` – average energy of the noise in dB. Values closer to 0 indicate louder noise.
/// * `spectral_flatness` – ratio of geometric mean to arithmetic mean of the power spectrum. Values near 1.0 indicate
///   flat, broadband noise (e.g. human chatter), while lower values indicate tonal or mechanical noise.
/// * `spectral_centroid` – normalized centroid of the spectrum, in the range `[0.0, 1.0]`, where 0.0 corresponds
///   to low‑frequency dominated noise and 1.0 to high‑frequency dominated noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoiseFeatures {
    pub energy_db: f32,
    pub spectral_flatness: f32,
    pub spectral_centroid: f32,
}

/// High‑level classification of ambient noise environment.
///
/// These categories are coarse and based on heuristic thresholds; they should be refined with empirical testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentType {
    /// Very low noise floor; typical of quiet rooms.
    Quiet,
    /// Stationary or tonal mechanical noise (air conditioner, fans, hums).
    Mechanical,
    /// Non‑stationary human or crowd noise (cafes, subway platforms).
    Human,
    /// Extremely loud or unpredictable noise.
    Extreme,
}

impl Default for EnvironmentType {
    fn default() -> Self {
        EnvironmentType::Quiet
    }
}

/// Maintains recent noise measurements and provides environment classification.
///
/// The analyser accumulates a short history of features to smooth out momentary fluctuations. The `history_size`
/// parameter controls how many past measurements are used when computing a smoothed classification.
pub struct NoiseAnalyzer {
    history: std::collections::VecDeque<NoiseFeatures>,
    history_size: usize,
}

impl NoiseAnalyzer {
    /// Create a new analyser with a given history length.
    pub fn new(history_size: usize) -> Self {
        Self {
            history: std::collections::VecDeque::with_capacity(history_size),
            history_size,
        }
    }

    /// Compute noise features from a single‑channel spectrogram slice.
    ///
    /// The input should be the magnitude or complex spectrum of shape `(1, freq_bins)`. Only the first row is
    /// considered. The values are treated as complex; power is computed from their squared magnitudes.
    pub fn compute_features(&self, spec: &ArrayView2<Complex32>) -> NoiseFeatures {
        // Expect shape (1, freq)
        let (_, freq_len) = spec.dim();
        let freq_len_f32 = freq_len as f32;
        let mut sum_power = 0.0f32;
        let mut sum_log_power = 0.0f32;
        let mut weighted_sum = 0.0f32;
        let eps = 1e-12f32;
        let row = spec.row(0);
        for (i, &c) in row.iter().enumerate() {
            let power = c.norm_sqr().max(eps);
            sum_power += power;
            sum_log_power += power.ln();
            weighted_sum += power * i as f32;
        }
        // Energy in dB: convert mean power (linear) to dBFS. Because the absolute scale depends on STFT windowing
        // and normalization, we only use it relative to threshold values.
        let mean_power = sum_power / freq_len_f32;
        let energy_db = 10.0 * mean_power.max(eps).log10();
        // Spectral flatness: ratio of geometric mean to arithmetic mean of power
        let geometric_mean = (sum_log_power / freq_len_f32).exp();
        let spectral_flatness = geometric_mean / mean_power.max(eps);
        // Spectral centroid normalized by frequency count
        let centroid = if sum_power > 0.0 {
            (weighted_sum / sum_power) / freq_len_f32
        } else {
            0.0
        };
        NoiseFeatures {
            energy_db: energy_db as f32,
            spectral_flatness: spectral_flatness as f32,
            spectral_centroid: centroid as f32,
        }
    }

    /// Push a new measurement into the history buffer.
    fn push_features(&mut self, features: NoiseFeatures) {
        if self.history.len() == self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(features);
    }

    /// Classify the environment based on the most recent noise measurement. This method also pushes the
    /// measurement into the history for smoothing. The returned category is the mode of the last `history_size`
    /// classifications (i.e., the most frequent category), providing some inertia to the environment state.
    pub fn classify(&mut self, features: NoiseFeatures) -> EnvironmentType {
        self.push_features(features);
        // Simple threshold based on instantaneous measurement
        let inst_class = Self::classify_once(features);
        // Count frequency of each class in the history
        let mut counters = [0usize; 4];
        for f in self.history.iter() {
            let cls = Self::classify_once(*f);
            let idx = match cls {
                EnvironmentType::Quiet => 0,
                EnvironmentType::Mechanical => 1,
                EnvironmentType::Human => 2,
                EnvironmentType::Extreme => 3,
            };
            counters[idx] += 1;
        }
        // Determine the mode of the history counts
        let (max_idx, _) = counters
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .unwrap();
        match max_idx {
            0 => EnvironmentType::Quiet,
            1 => EnvironmentType::Mechanical,
            2 => EnvironmentType::Human,
            _ => EnvironmentType::Extreme,
        }
    }

    /// Classify a single measurement into an environment category without updating history.
    fn classify_once(features: NoiseFeatures) -> EnvironmentType {
        // Heuristic thresholds derived from typical noise spectra. Adjust based on evaluation.
        if features.energy_db < -40.0 {
            EnvironmentType::Quiet
        } else if features.spectral_flatness < 0.3 {
            EnvironmentType::Mechanical
        } else if features.spectral_flatness > 0.6 {
            // High flatness implies broadband, human‑like noise. Use centroid to refine: high centroid indicates
            // significant high‑frequency content (like hiss or crowd), else treat as human noise.
            if features.spectral_centroid > 0.5 {
                EnvironmentType::Human
            } else {
                EnvironmentType::Mechanical
            }
        } else {
            EnvironmentType::Extreme
        }
    }
}