use super::noise_analysis::EnvironmentType;

/// A set of parameters tuned for a specific noise environment.
///
/// These fields map directly onto controls in the audio processing chain. They are conservative defaults and
/// should be adjusted based on listening tests. The fields include:
///
/// * `atten_lim` – attenuation limit for DeepFilterNet, in dB. Higher values allow stronger noise suppression.
/// * `min_thresh_db` – minimum threshold for the DF noise floor estimator, in dB.
/// * `max_df_thresh_db` – maximum threshold for DF estimator, in dB.
/// * `df_mix` – dry/wet mix ratio; 1.0 uses only DF output, 0.0 bypasses DF entirely.
/// * `highpass_cutoff` – cutoff frequency for the high‑pass filter, in Hz.
#[derive(Debug, Clone, Copy)]
pub struct EnvPreset {
    pub atten_lim: f32,
    pub min_thresh_db: f32,
    pub max_df_thresh_db: f32,
    pub df_mix: f32,
    pub highpass_cutoff: f32,
}

/// Return a preset based on the classified environment. These presets provide a reasonable starting point for
/// automatically adjusting noise suppression and tone‑shaping parameters. They are intentionally conservative to
/// avoid over‑suppression in quiet environments and aggressive enough for extreme noise.
pub fn get_preset(env: EnvironmentType) -> EnvPreset {
    match env {
        EnvironmentType::Quiet => EnvPreset {
            atten_lim: 10.0,        // gentle suppression
            min_thresh_db: -80.0,   // base noise floor estimate
            max_df_thresh_db: -20.0, // avoid over‑attenuation
            df_mix: 0.6,            // preserve more original signal
            highpass_cutoff: 60.0,  // low cutoff, preserve bass
        },
        EnvironmentType::Mechanical => EnvPreset {
            atten_lim: 15.0,
            min_thresh_db: -85.0,
            max_df_thresh_db: -25.0,
            df_mix: 0.8,
            highpass_cutoff: 80.0,
        },
        EnvironmentType::Human => EnvPreset {
            atten_lim: 18.0,
            min_thresh_db: -90.0,
            max_df_thresh_db: -30.0,
            df_mix: 0.9,
            highpass_cutoff: 100.0,
        },
        EnvironmentType::Extreme => EnvPreset {
            atten_lim: 24.0,
            min_thresh_db: -95.0,
            max_df_thresh_db: -35.0,
            df_mix: 1.0,
            highpass_cutoff: 120.0,
        },
    }
}