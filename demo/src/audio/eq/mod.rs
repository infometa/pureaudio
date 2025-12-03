mod biquad;
mod dynamic_band;
mod dynamic_eq;
mod envelope;
mod presets;

// Biquad kept internal; no external re-export to avoid未用警告
#[cfg(feature = "ui")]
#[allow(unused_imports)]
pub use dynamic_band::{BandMode, FilterKind};
pub use dynamic_eq::{DynamicEq, EqControl, EqProcessMetrics};
pub use presets::{EqPresetKind, MAX_EQ_BANDS};
