//! High level processor that wraps DeepFilterNet and ancillary DSP modules.
//!
//! The `Processor` struct glues together the noise analysis, environment preset
//! selection, DeepFilterNet, high‑pass filtering, automatic gain control and
//! limiting. It operates on blocks of audio samples and produces processed
//! audio in real time.

use std::path::Path;

use crate::dsp::{agc::Agc, bus_limiter::BusLimiter, highpass::Highpass};
use crate::dsp::{eq::{Eq, EqPreset}, saturation::Saturation, transient::TransientShaper};
use crate::env_preset::get_preset;
use crate::noise_analysis::NoiseAnalyzer;
use df::DfTract;
use ndarray::prelude::*;

/// Real‑time audio processor combining DeepFilterNet with adaptive controls.
pub struct Processor {
    /// DeepFilterNet inference engine.
    pub df: DfTract,
    /// High‑pass filter applied after dry/wet mixing.
    highpass: Highpass,
    /// Automatic gain control to normalize loudness.
    agc: Agc,
    /// Bus limiter to tame peaks and prevent clipping.
    limiter: BusLimiter,
    /// Analyzer for estimating noise characteristics and classifying environment.
    analyzer: NoiseAnalyzer,
    /// Current dry/wet mix ratio for DF output.
    df_mix: f32,
    /// Equalizer for timbre shaping.
    eq: Eq,
    /// Transient shaper for enhancing or softening attacks.
    transient: TransientShaper,
    /// Saturation effect for harmonic enhancement.
    saturation: Saturation,
}

impl Processor {
    /// Load the model and initialize processing modules. If `model_path` is `None`,
    /// a default built‑in model is used (as defined by the `df` crate). The history
    /// length of the noise analyzer controls how quickly the environment adapts;
    /// here it is fixed at 8 frames.
    pub fn new(model_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        // Instantiate DeepFilterNet. The df crate accepts an optional path to
        // a `.df` model; `None` loads the default shipped with the crate.
        let mut df = DfTract::new(model_path)?;
        // Retrieve sample rate to configure the high‑pass filter. The df API
        // exposes the sample rate via the `sample_rate` field. If this API
        // changes, adjust accordingly based on compile errors.
        let sr = df.sample_rate as f32;
        // Initialize DSP modules with default parameters.
        let highpass = Highpass::new(80.0, sr);
        let agc = Agc::new();
        let limiter = BusLimiter::new();
        let analyzer = NoiseAnalyzer::new(8);
        // Initialise EQ with a default preset and 100% wet. The preset can be
        // adjusted if you wish to experiment with different voicings.
        let eq = Eq::new(sr, EqPreset::Natural, 1.0);
        // Create transient shaper and saturation with defaults.
        let transient = TransientShaper::new(sr);
        let saturation = Saturation::new();
        Ok(Self {
            df,
            highpass,
            agc,
            limiter,
            analyzer,
            df_mix: 1.0,
            eq,
            transient,
            saturation,
        })
    }

    /// Return the sample rate used by the processor.
    pub fn sample_rate(&self) -> usize {
        self.df.sample_rate as usize
    }

    /// Return the hop size (frame size) used internally by DeepFilterNet.
    pub fn hop_size(&self) -> usize {
        self.df.hop_size as usize
    }

    /// Process a block of audio samples and return the processed block. The input
    /// slice is not modified. The block size should match `self.hop_size()`.
    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        // Prepare buffers for DF input and output. Copy input because df.process
        // works in place on its input slice.
        let mut in_buf = input.to_vec();
        let mut df_out = vec![0.0f32; input.len()];
        // Run DeepFilterNet inference. This fills `df_out` with the denoised
        // samples and returns an LSNR estimate. The LSNR is currently unused.
        let _lsnr = self.df.process(&mut in_buf, &mut df_out);
        // Obtain the noisy spectrogram for feature extraction. DeepFilterNet
        // exposes the most recent spectrogram via `get_spec_noisy`.
        let spec_noisy = self.df.get_spec_noisy();
        let features = self.analyzer.compute_features(&spec_noisy.view());
        let env = self.analyzer.classify(features);
        let preset = get_preset(env);
        // Apply preset to DF and DSP modules. Access to these fields may
        // change in future versions of df; adjust the code if compilation fails.
        self.df.atten_lim = preset.atten_lim;
        self.df.min_df_thresh_db = preset.min_thresh_db;
        self.df.max_df_thresh_db = preset.max_df_thresh_db;
        self.df_mix = preset.df_mix;
        self.highpass.set_cutoff(preset.highpass_cutoff);
        // Mix DF output and original input according to df_mix.
        let mut mixed: Vec<f32> = Vec::with_capacity(df_out.len());
        for (wet, dry) in df_out.iter().zip(input.iter()) {
            mixed.push(self.df_mix * *wet + (1.0 - self.df_mix) * *dry);
        }
        // Apply transient shaping and saturation before tone shaping and loudness control.
        self.transient.process(&mut mixed);
        self.saturation.process(&mut mixed);
        // Apply EQ for timbre shaping. The EQ mixes dry/wet internally.
        self.eq.process_inplace(&mut mixed);
        // Apply high‑pass filter to remove rumble, then AGC and bus limiting.
        self.highpass.process_inplace(&mut mixed);
        self.agc.process_inplace(&mut mixed);
        self.limiter.process_inplace(&mut mixed);
        // Final hard limiting to ensure samples are in [-1, 1].
        for v in mixed.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        mixed
    }
}