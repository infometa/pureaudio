use std::borrow::Cow;
use std::env;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::process::{exit, Command as StdCommand};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueHint};
use cpal::traits::{DeviceTrait, HostTrait};
use crossbeam_channel::unbounded;
use hound;
use iced::widget::tooltip::{self, Position};
use iced::widget::{
    self, column, container, image, pick_list, row, scrollable, slider, text, text_input, toggler,
    Container, Image,
};
use iced::{
    alignment, executor, Alignment, Application, Color, Command, ContentFit, Element, Font, Length,
    Settings, Subscription, Theme,
};
use std::collections::HashMap;
use image_rs::{Rgba, RgbaImage};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use serde_json;

mod audio;
mod capture;
mod cmap;
mod scene;
mod ui;
use audio::eq::{BandMode, EqControl, EqPresetKind, FilterKind, MAX_EQ_BANDS};
use capture::*;
use capture::{EnvStatus, RecvEnvStatus};
use scene::ScenePreset;
use ui::tooltips;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
}

pub fn main() -> iced::Result {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    if args.model.is_some() {
        capture::set_model_path(args.model.clone());
    }

    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default())
            .filter_level(level)
            .filter_module("tract_onnx", tract_level)
            .filter_module("tract_hir", tract_level)
            .filter_module("tract_core", tract_level)
            .filter_module("tract_linalg", tract_level)
            .filter_module("iced_winit", log::LevelFilter::Error)
            .filter_module("iced_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .filter_module("crossfont", log::LevelFilter::Error)
            .filter_module("cosmic_text", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    let mut settings = Settings::default();
    settings.fonts.push(Cow::Borrowed(UI_FONT_BYTES));
    settings.default_font = UI_FONT;
    SpecView::run(settings)
}

const SPEC_DISPLAY_WIDTH: u16 = 1000;
const SPEC_DISPLAY_HEIGHT: u16 = 250;
const OUTPUT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/output");
const TIMBRE_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/timbre_restore.onnx");
const CONFIG_VERSION: u32 = 1;
const UI_FONT: Font = Font::with_name("Source Han Sans CN");
const UI_FONT_BYTES: &[u8] = include_bytes!("../fonts/SourceHanSansCN-Regular.ttf");
const EQ_BAND_LABELS: [&str; MAX_EQ_BANDS] = ["BAND1", "BAND2", "BAND3", "BAND4", "BAND5"];

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_auto_play_file() -> Option<PathBuf> {
    Some(PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/audio/kh.wav"
    )))
}

fn list_audio_devices() -> (Vec<String>, Vec<String>, Option<String>, Option<String>) {
    let host = cpal::default_host();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut default_in = host.default_input_device().and_then(|dev| dev.name().ok());
    let mut default_out = host.default_output_device().and_then(|dev| dev.name().ok());

    match host.input_devices() {
        Ok(devices) => {
            for dev in devices {
                if let Ok(name) = dev.name() {
                    if default_in.is_none() {
                        default_in = Some(name.clone());
                    }
                    if !inputs.contains(&name) {
                        inputs.push(name);
                    }
                }
            }
        }
        Err(err) => log::warn!("枚举输入设备失败: {}", err),
    }

    match host.output_devices() {
        Ok(devices) => {
            for dev in devices {
                if let Ok(name) = dev.name() {
                    if default_out.is_none() {
                        default_out = Some(name.clone());
                    }
                    if !outputs.contains(&name) {
                        outputs.push(name);
                    }
                }
            }
        }
        Err(err) => log::warn!("枚举输出设备失败: {}", err),
    }

    (inputs, outputs, default_in, default_out)
}

struct SpecView {
    df_worker: Option<DeepFilterCapture>,
    lsnr: f32,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    df_mix: f32,
    headroom_gain: f32,
    show_spectrum: bool,
    rt60_enabled: bool,
    final_limiter_enabled: bool,
    post_trim_gain: f32,
    noisy_img: image::Handle,
    enh_img: image::Handle,
    noisy_spec: Arc<Mutex<SpecImage>>,
    enh_spec: Arc<Mutex<SpecImage>>,
    r_lsnr: Option<RecvLsnr>,
    r_noisy: Option<RecvSpec>,
    r_enh: Option<RecvSpec>,
    s_controls: Option<SendControl>,
    r_eq_status: Option<RecvEqStatus>,
    r_env_status: Option<RecvEnvStatus>,
    eq_enabled: bool,
    eq_preset: EqPresetKind,
    eq_dry_wet: f32,
    eq_status: EqStatus,
    eq_show_advanced: bool,
    eq_band_gains: [f32; MAX_EQ_BANDS],
    eq_band_frequencies: [f32; MAX_EQ_BANDS],
    eq_band_qs: [f32; MAX_EQ_BANDS],
    eq_band_detector_qs: [f32; MAX_EQ_BANDS],
    eq_band_thresholds: [f32; MAX_EQ_BANDS],
    eq_band_ratios: [f32; MAX_EQ_BANDS],
    eq_band_max_gains: [f32; MAX_EQ_BANDS],
    eq_band_attacks: [f32; MAX_EQ_BANDS],
    eq_band_releases: [f32; MAX_EQ_BANDS],
    eq_band_makeups: [f32; MAX_EQ_BANDS],
    eq_band_modes: [BandMode; MAX_EQ_BANDS],
    eq_band_filters: [FilterKind; MAX_EQ_BANDS],
    eq_band_expanded: [bool; MAX_EQ_BANDS],
    eq_band_show_advanced: [bool; MAX_EQ_BANDS],
    mute_playback: bool,
    auto_play_enabled: bool,
    auto_play_file: Option<PathBuf>,
    auto_play_pid: Option<u32>,
    highpass_enabled: bool,
    highpass_cutoff: f32,
    transient_enabled: bool,
    transient_gain: f32,
    transient_sustain: f32,
    transient_mix: f32,
    show_transient_advanced: bool,
    saturation_enabled: bool,
    saturation_drive: f32,
    saturation_makeup: f32,
    saturation_mix: f32,
    show_saturation_advanced: bool,
    agc_enabled: bool,
    timbre_enabled: bool,
    agc_current_gain: f32,
    agc_target_db: f32,
    agc_max_gain_db: f32,
    agc_max_atten_db: f32,
    agc_window_sec: f32,
    agc_attack_ms: f32,
    agc_release_ms: f32,
    show_agc_advanced: bool,
    aec_aggressive: bool,
    aec_enabled: bool,
    aec_delay_ms: f32,
    sys_auto_volume: bool,
    env_auto_enabled: bool,
    sysvol_monitor: Option<capture::SysVolMonitorHandle>,
    exciter_enabled: bool,
    exciter_mix: f32,
    bypass_enabled: bool,
    user_selected_input: bool,
    user_selected_output: bool,
    vad_enabled: bool,
    env_status_label: String,
    noise_show_advanced: bool,
    scene_preset: ScenePreset,
    model_path: Option<PathBuf>,
    recording: Option<RecordingHandle>,
    is_running: bool,
    is_saving: bool,
    status_text: String,
    last_saved: Option<(PathBuf, PathBuf, PathBuf, PathBuf)>,
    input_buffers: HashMap<String, String>,
    spec_frames: u32,
    spec_freqs: u32,
    input_device_filter: String,
    output_device_filter: String,
    input_devices: Vec<String>,
    output_devices: Vec<String>,
    selected_input_device: Option<String>,
    selected_output_device: Option<String>,
    show_device_selector: bool,
}

#[derive(Debug, Clone)]
pub enum SliderTarget {
    AttenLim,
    PostFilterBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
    HighpassCutoff,
    SaturationDrive,
    SaturationMakeup,
    SaturationMix,
    ExciterMix,
    TransientGain,
    TransientSustain,
    TransientMix,
    HeadroomGain,
    AecDelay,
    AgcTarget,
    AgcMaxGain,
    AgcMaxAtten,
    AgcWindow,
    AgcAttack,
    AgcRelease,
}

#[derive(Debug, Clone)]
pub enum Message {
    None,
    Tick,
    LsnrChanged(f32),
    NoisyChanged,
    EnhChanged,
    HeadroomChanged(f32),
    SpectrumToggled(bool),
    Rt60Toggled(bool),
    FinalLimiterToggled(bool),
    AttenLimChanged(f32),
    PostFilterChanged(f32),
    MinThreshDbChanged(f32),
    MaxErbThreshDbChanged(f32),
    MaxDfThreshDbChanged(f32),
    DfMixChanged(f32),
    SliderInputChanged {
        key: String,
        raw: String,
        target: SliderTarget,
        min: f32,
        max: f32,
        precision: usize,
    },
    NoiseAdvancedToggled,
    EqEnabledChanged(bool),
    EqPresetSelected(EqPresetKind),
    EqDryWetChanged(f32),
    EqBandGainChanged(usize, f32),
    EqBandFrequencyChanged(usize, f32),
    EqBandQChanged(usize, f32),
    EqBandDetectorQChanged(usize, f32),
    EqBandThresholdChanged(usize, f32),
    EqBandRatioChanged(usize, f32),
    EqBandMaxGainChanged(usize, f32),
    EqBandAttackChanged(usize, f32),
    EqBandReleaseChanged(usize, f32),
    EqBandMakeupChanged(usize, f32),
    EqBandModeChanged(usize, BandMode),
    EqBandFilterChanged(usize, FilterKind),
    EqBandToggleExpand(usize),
    EqBandToggleAdvanced(usize),
    EqToggleAdvanced,
    EqResetBands,
    EqStatusUpdated(EqStatus),
    InputDeviceChanged(String),
    OutputDeviceChanged(String),
    InputDeviceSelected(String),
    OutputDeviceSelected(String),
    DevicePanelToggled(bool),
    ScenePresetChanged(ScenePreset),
    MutePlaybackChanged(bool),
    BypassToggled(bool),
    AutoPlayToggled(bool),
    AutoPlayPickRequested,
    AutoPlayFilePicked(Option<PathBuf>),
    PlaybackFinished(Result<(), String>),
    HighpassToggled(bool),
    HighpassCutoffChanged(f32),
    SaturationToggled(bool),
    SaturationDriveChanged(f32),
    SaturationMakeupChanged(f32),
    SaturationMixChanged(f32),
    SaturationToggleAdvanced,
    TransientToggled(bool),
    TransientGainChanged(f32),
    TransientSustainChanged(f32),
    TransientMixChanged(f32),
    TransientToggleAdvanced,
    VadToggled(bool),
    AgcToggled(bool),
    AgcTargetChanged(f32),
    AgcMaxGainChanged(f32),
    AgcMaxAttenChanged(f32),
    AgcWindowChanged(f32),
    AgcAttackChanged(f32),
    AgcReleaseChanged(f32),
    AgcToggleAdvanced,
    SysAutoVolumeToggled(bool),
    EnvAutoToggled(bool),
    ExciterToggled(bool),
    ExciterMixChanged(f32),
    AecToggled(bool),
    AecDelayChanged(f32),
    AecAggressiveToggled(bool),
    TimbreToggled(bool),
    EnvStatusUpdated(EnvStatus),
    StartProcessing,
    StopProcessing,
    SaveFinished(Result<(PathBuf, PathBuf, PathBuf, PathBuf), String>),
    SaveConfigRequested,
    LoadConfigRequested,
    ConfigSaveFinished(Result<PathBuf, String>),
    ConfigLoadFinished(Result<UserConfig, String>),
    Exit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConfig {
    version: u32,
    scene_preset: ScenePreset,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    df_mix: f32,
    post_trim_gain: f32,
    eq_enabled: bool,
    eq_preset: EqPresetKind,
    eq_dry_wet: f32,
    eq_band_gains: [f32; MAX_EQ_BANDS],
    eq_band_frequencies: [f32; MAX_EQ_BANDS],
    eq_band_qs: [f32; MAX_EQ_BANDS],
    eq_band_detector_qs: [f32; MAX_EQ_BANDS],
    eq_band_thresholds: [f32; MAX_EQ_BANDS],
    eq_band_ratios: [f32; MAX_EQ_BANDS],
    eq_band_max_gains: [f32; MAX_EQ_BANDS],
    eq_band_attacks: [f32; MAX_EQ_BANDS],
    eq_band_releases: [f32; MAX_EQ_BANDS],
    eq_band_makeups: [f32; MAX_EQ_BANDS],
    eq_band_modes: [BandMode; MAX_EQ_BANDS],
    eq_band_filters: [FilterKind; MAX_EQ_BANDS],
    eq_show_advanced: bool,
    eq_band_show_advanced: [bool; MAX_EQ_BANDS],
    eq_band_expanded: [bool; MAX_EQ_BANDS],
    noise_show_advanced: bool,
    mute_playback: bool,
    #[serde(default)]
    auto_play_enabled: bool,
    #[serde(default = "default_auto_play_file")]
    auto_play_file: Option<PathBuf>,
    highpass_enabled: bool,
    highpass_cutoff: f32,
    saturation_enabled: bool,
    saturation_drive: f32,
    saturation_makeup: f32,
    saturation_mix: f32,
    show_saturation_advanced: bool,
    transient_enabled: bool,
    transient_gain: f32,
    transient_sustain: f32,
    transient_mix: f32,
    show_transient_advanced: bool,
    // 输出头间距，默认 0.9，UI 控制
    headroom_gain: f32,
    // UI: 是否显示/推送频谱，降低性能占用
    #[serde(default = "default_true")]
    show_spectrum: bool,
    // UI: 是否计算/显示 RT60（仅自适应时），可关掉减少计算
    #[serde(default = "default_true")]
    rt60_enabled: bool,
    #[serde(default = "default_true")]
    final_limiter_enabled: bool,
    agc_enabled: bool,
    #[serde(default)]
    timbre_enabled: bool,
    agc_target_db: f32,
    agc_max_gain_db: f32,
    agc_max_atten_db: f32,
    agc_window_sec: f32,
    agc_attack_ms: f32,
    agc_release_ms: f32,
    show_agc_advanced: bool,
    #[serde(default)]
    aec_enabled: bool,
    #[serde(default)]
    aec_delay_ms: f32,
    #[serde(default = "default_true")]
    aec_aggressive: bool,
    #[serde(default)]
    sys_auto_volume: bool,
    #[serde(default = "default_false")]
    env_auto_enabled: bool,
    #[serde(default = "default_false")]
    vad_enabled: bool,
}

fn write_config_file(cfg: &UserConfig, path: &Path) -> Result<PathBuf, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let data = serde_json::to_vec_pretty(cfg).map_err(|e| e.to_string())?;
    fs::write(path, data).map_err(|e| e.to_string())?;
    Ok(path.to_path_buf())
}

fn read_config_file(path: &Path) -> Result<UserConfig, String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    serde_json::from_slice(&data).map_err(|e| e.to_string())
}

async fn save_config_with_dialog(cfg: UserConfig) -> Result<PathBuf, String> {
    let Some(path) = FileDialog::new()
        .add_filter("配置文件", &["json"])
        .set_file_name("df_config.json")
        .save_file()
    else {
        return Err("用户取消".to_string());
    };
    write_config_file(&cfg, &path)
}

fn load_config_with_dialog(
) -> impl Future<Output = Result<UserConfig, String>> + 'static + Send + Sync {
    async move {
        let Some(path) = FileDialog::new().add_filter("配置文件", &["json"]).pick_file() else {
            return Err("用户取消".to_string());
        };
        read_config_file(&path)
    }
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
    write_pos: usize,
    frames_written: usize,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            im: RgbaImage::new(n_frames, n_freqs),
            n_frames,
            n_freqs,
            vmin,
            vmax,
            write_pos: 0,
            frames_written: 0,
        }
    }
    fn capacity(&self) -> usize {
        self.n_frames as usize
    }
    fn update<I>(&mut self, specs: I)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        for spec in specs.take(self.capacity()) {
            self.push_column(&spec);
        }
    }
    fn push_column(&mut self, spec: &[f32]) {
        let freq_bins = self.n_freqs as usize;
        let column = self.write_pos;
        for (freq_idx, &sample) in spec.iter().take(freq_bins).enumerate() {
            self.im.put_pixel(column as u32, freq_idx as u32, self.color_for_value(sample));
        }
        if spec.len() < freq_bins {
            for freq_idx in spec.len()..freq_bins {
                self.im.put_pixel(column as u32, freq_idx as u32, Rgba([0, 0, 0, 255]));
            }
        }
        self.write_pos = (self.write_pos + 1) % self.capacity();
        self.frames_written = (self.frames_written + 1).min(self.capacity());
    }
    fn color_for_value(&self, value: f32) -> Rgba<u8> {
        let v = (value.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
        Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize])
    }
    fn image_handle(&self) -> image::Handle {
        image::Handle::from_pixels(self.n_frames, self.n_freqs, self.ordered_bytes())
    }
    fn ordered_bytes(&self) -> Vec<u8> {
        let width = self.n_frames as usize;
        let height = self.n_freqs as usize;
        let mut buf = vec![0; width * height * 4];
        if width == 0 || height == 0 {
            return buf;
        }
        let raw = self.im.as_raw();
        let filled = self.frames_written.min(width);
        if filled == 0 {
            return buf;
        }
        for out_x in 0..width {
            let has_full_buffer = self.frames_written >= width;
            let src_x = if has_full_buffer {
                (self.write_pos + out_x) % width
            } else if out_x < filled {
                out_x
            } else {
                continue;
            };
            for y in 0..height {
                let src_idx = ((y * width) + src_x) * 4;
                let dst_idx = ((y * width) + out_x) * 4;
                buf[dst_idx..dst_idx + 4].copy_from_slice(&raw[src_idx..src_idx + 4]);
            }
        }
        buf
    }
}

fn create_spec_storage(
    spec_frames: u32,
    spec_freqs: u32,
) -> (Arc<Mutex<SpecImage>>, image::Handle) {
    let spec = Arc::new(Mutex::new(SpecImage::new(
        spec_frames,
        spec_freqs,
        -100.,
        -10.,
    )));
    let handle = spec_image_handle(&spec, spec_frames, spec_freqs);
    (spec, handle)
}

fn spec_image_handle(
    spec: &Arc<Mutex<SpecImage>>,
    spec_frames: u32,
    spec_freqs: u32,
) -> image::Handle {
    match spec.lock() {
        Ok(guard) => guard.image_handle(),
        Err(err) => {
            log::error!("无法锁定频谱缓存: {}", err);
            image::Handle::from_pixels(
                spec_frames,
                spec_freqs,
                vec![0; spec_frames as usize * spec_freqs as usize * 4],
            )
        }
    }
}

impl Application for SpecView {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let model_path = current_model_path();
        let (sr, frame_size, freq_size) =
            capture::model_dimensions(model_path.clone(), 1).unwrap_or_else(|_| (48000, 960, 512));
        let spec_frames = ((sr / frame_size) * 10).max(1) as u32;
        let freq_res = (sr / 2) / (freq_size.saturating_sub(1).max(1));
        let spec_freqs = (8000 / freq_res.max(1)).max(1) as u32;
        let (noisy_spec, noisy_img) = create_spec_storage(spec_frames, spec_freqs);
        let (enh_spec, enh_img) = create_spec_storage(spec_frames, spec_freqs);
        let eq_preset = EqPresetKind::default();
        let _eq_dry_wet = 1.0;
        let preset_config = eq_preset.preset();
        let mut eq_band_gains = [0.0; MAX_EQ_BANDS];
        let mut eq_band_frequencies = [0.0; MAX_EQ_BANDS];
        let mut eq_band_qs = [0.0; MAX_EQ_BANDS];
        let mut eq_band_detector_qs = [0.0; MAX_EQ_BANDS];
        let mut eq_band_thresholds = [0.0; MAX_EQ_BANDS];
        let mut eq_band_ratios = [1.0; MAX_EQ_BANDS];
        let mut eq_band_max_gains = [0.0; MAX_EQ_BANDS];
        let mut eq_band_attacks = [0.0; MAX_EQ_BANDS];
        let mut eq_band_releases = [0.0; MAX_EQ_BANDS];
        let mut eq_band_makeups = [0.0; MAX_EQ_BANDS];
        let mut eq_band_modes = [BandMode::Downward; MAX_EQ_BANDS];
        let mut eq_band_filters = [FilterKind::Peak; MAX_EQ_BANDS];
        for (i, band) in preset_config.bands.iter().enumerate() {
            eq_band_gains[i] = band.static_gain_db;
            eq_band_frequencies[i] = band.frequency_hz;
            eq_band_qs[i] = band.q;
            eq_band_detector_qs[i] = band.detector_q();
            eq_band_thresholds[i] = band.threshold_db;
            eq_band_ratios[i] = band.ratio;
            eq_band_max_gains[i] = band.max_gain_db;
            eq_band_attacks[i] = band.attack_ms;
            eq_band_releases[i] = band.release_ms;
            eq_band_makeups[i] = band.makeup_db;
            eq_band_modes[i] = band.mode;
            eq_band_filters[i] = band.filter;
        }
        let mut eq_band_expanded = [false; MAX_EQ_BANDS];
        if !eq_band_expanded.is_empty() {
            eq_band_expanded[0] = true;
        }
        let eq_band_show_advanced = [false; MAX_EQ_BANDS];
        let (input_devices, output_devices, default_input_device, default_output_device) =
            list_audio_devices();
        (
            Self {
                df_worker: None,
                lsnr: 0.,
                atten_lim: 30.,
                post_filter_beta: 0.,
                min_threshdb: -60.,
                max_erbthreshdb: 20.,
                max_dfthreshdb: 20.,
                df_mix: 1.0,
                headroom_gain: 0.9,
                show_spectrum: true,
                rt60_enabled: true,
                final_limiter_enabled: false,
                aec_aggressive: true,
                post_trim_gain: 1.0,
                noisy_img,
                enh_img,
                noisy_spec,
                enh_spec,
                r_lsnr: None,
                r_noisy: None,
                r_enh: None,
                s_controls: None,
                r_eq_status: None,
                r_env_status: None,
                eq_enabled: false,
                eq_preset,
                eq_dry_wet: 1.0,
                eq_status: EqStatus::default(),
                eq_show_advanced: false,
                noise_show_advanced: false,
                eq_band_gains,
                eq_band_frequencies,
                eq_band_qs,
                eq_band_detector_qs,
                eq_band_thresholds,
                eq_band_ratios,
                eq_band_max_gains,
                eq_band_attacks,
                eq_band_releases,
                eq_band_makeups,
                eq_band_modes,
                eq_band_filters,
                eq_band_expanded,
                eq_band_show_advanced,
                mute_playback: false,
                auto_play_enabled: false,
                auto_play_file: default_auto_play_file(),
                auto_play_pid: None,
                highpass_enabled: true,
                highpass_cutoff: 60.0,
                transient_enabled: false,
                transient_gain: 3.5,
                transient_sustain: 0.0,
                transient_mix: 100.0,
                show_transient_advanced: false,
                saturation_enabled: false,
                saturation_drive: 1.2,
                saturation_makeup: -0.5,
                saturation_mix: 100.0,
                show_saturation_advanced: false,
                agc_enabled: true,
                timbre_enabled: false,
                agc_current_gain: 0.0,
                agc_target_db: -16.0,
                agc_max_gain_db: 12.0,
                agc_max_atten_db: 12.0,
                agc_window_sec: 0.6,
                agc_attack_ms: 500.0,
                agc_release_ms: 2000.0,
                show_agc_advanced: false,
                aec_enabled: false,
                aec_delay_ms: 60.0,
                sys_auto_volume: false,
                env_auto_enabled: false,
                vad_enabled: false,
                exciter_enabled: false,
                exciter_mix: 0.0,
                bypass_enabled: false,
                user_selected_input: false,
                user_selected_output: false,
                env_status_label: "自适应降噪: 关闭".to_string(),
                sysvol_monitor: None,
                scene_preset: ScenePreset::OpenOfficeMeeting,
                model_path,
                recording: None,
                is_running: false,
                is_saving: false,
                status_text: "待机".to_string(),
                last_saved: None,
                input_buffers: HashMap::new(),
                spec_frames,
                spec_freqs,
                input_device_filter: String::new(),
                output_device_filter: String::new(),
                input_devices,
                output_devices,
                selected_input_device: default_input_device,
                selected_output_device: default_output_device,
                show_device_selector: false,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "DeepFilterNet 拾音演示".to_string()
    }

    // fn theme(&self) -> Self::Theme {
    //     Theme::Dark
    // }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::None => (),
            Message::StartProcessing => return self.start_processing(),
            Message::StopProcessing => return self.stop_processing(),
            Message::SaveFinished(result) => {
                self.is_saving = false;
                match result {
                    Ok((raw, denoised, timbre, processed)) => {
                        self.status_text = "音频已保存".to_string();
                        self.last_saved = Some((raw, denoised, timbre, processed));
                        if let Some((raw, _, _, _)) = &self.last_saved {
                            if let Some(dir) = raw.parent() {
                                match self.save_config_to_path(&dir.join("config.json")) {
                                    Ok(_) => {}
                                    Err(err) => {
                                        log::warn!("自动保存配置失败: {}", err);
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        self.last_saved = None;
                        self.status_text = format!("保存失败: {}", err);
                        log::error!("录音保存失败: {}", err);
                    }
                }
            }
            Message::SaveConfigRequested => {
                let cfg = self.to_user_config();
                self.status_text = "正在保存配置...".to_string();
                return Command::perform(
                    async move { save_config_with_dialog(cfg).await },
                    Message::ConfigSaveFinished,
                );
            }
            Message::LoadConfigRequested => {
                self.status_text = "选择配置文件...".to_string();
                return Command::perform(load_config_with_dialog(), Message::ConfigLoadFinished);
            }
            Message::ConfigSaveFinished(result) => match result {
                Ok(path) => {
                    self.status_text = format!("配置已保存: {}", path.display());
                }
                Err(err) => {
                    self.status_text = format!("保存配置失败: {}", err);
                    log::warn!("保存配置失败: {}", err);
                }
            },
            Message::ConfigLoadFinished(result) => match result {
                Ok(cfg) => {
                    self.apply_user_config(cfg);
                    self.status_text = "配置已加载".to_string();
                }
                Err(err) => {
                    self.status_text = format!("加载配置失败: {}", err);
                    log::warn!("加载配置失败: {}", err);
                }
            },
            Message::Exit => {
                if let Some(mut worker) = self.df_worker.take() {
                    if let Err(err) = worker.should_stop() {
                        log::error!("停止音频处理失败: {}", err);
                    }
                }
                if let Some(handle) = self.sysvol_monitor.take() {
                    handle.stop();
                }
                exit(0);
            }
            Message::EqEnabledChanged(enabled) => {
                self.eq_enabled = enabled;
                self.send_eq_control(EqControl::SetEnabled(enabled));
            }
            Message::EqPresetSelected(preset) => {
                self.eq_preset = preset;
                self.eq_dry_wet = 1.0;
                self.apply_eq_preset_config(preset);
                self.send_eq_control(EqControl::SetPreset(preset));
                self.send_eq_control(EqControl::SetDryWet(self.eq_dry_wet));
                self.broadcast_eq_parameters();
                self.reset_eq_band_gains();
            }
            Message::EqDryWetChanged(_value) => {
                // 固定全湿，忽略滑条值，避免并行相位起伏
                self.eq_dry_wet = 1.0;
                self.send_eq_control(EqControl::SetDryWet(self.eq_dry_wet));
            }
            Message::EqBandGainChanged(idx, value) => self.set_eq_band_gain(idx, value),
            Message::EqBandFrequencyChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_frequencies[idx] = value;
                    self.send_eq_control(EqControl::SetBandFrequency(idx, value));
                }
            }
            Message::EqBandQChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_qs[idx] = value;
                    self.send_eq_control(EqControl::SetBandQ(idx, value));
                }
            }
            Message::EqBandDetectorQChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_detector_qs[idx] = value;
                    self.send_eq_control(EqControl::SetBandDetectorQ(idx, value));
                }
            }
            Message::EqBandThresholdChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_thresholds[idx] = value;
                    self.send_eq_control(EqControl::SetBandThreshold(idx, value));
                }
            }
            Message::EqBandRatioChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_ratios[idx] = value;
                    self.send_eq_control(EqControl::SetBandRatio(idx, value));
                }
            }
            Message::EqBandMaxGainChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_max_gains[idx] = value;
                    self.send_eq_control(EqControl::SetBandMaxGain(idx, value));
                }
            }
            Message::EqBandAttackChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_attacks[idx] = value;
                    self.send_eq_control(EqControl::SetBandAttack(idx, value));
                }
            }
            Message::EqBandReleaseChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_releases[idx] = value;
                    self.send_eq_control(EqControl::SetBandRelease(idx, value));
                }
            }
            Message::EqBandMakeupChanged(idx, value) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_makeups[idx] = value;
                    self.send_eq_control(EqControl::SetBandMakeup(idx, value));
                }
            }
            Message::EqBandModeChanged(idx, mode) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_modes[idx] = mode;
                    self.send_eq_control(EqControl::SetBandMode(idx, mode));
                }
            }
            Message::EqBandFilterChanged(idx, filter) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_filters[idx] = filter;
                    self.send_eq_control(EqControl::SetBandFilter(idx, filter));
                }
            }
            Message::InputDeviceChanged(name) => {
                self.input_device_filter = name;
            }
            Message::OutputDeviceChanged(name) => {
                self.output_device_filter = name;
            }
            Message::InputDeviceSelected(name) => {
                self.selected_input_device = Some(name);
                self.user_selected_input = true;
                if self.sys_auto_volume {
                    self.restart_sys_volume_monitor();
                }
            }
            Message::OutputDeviceSelected(name) => {
                self.selected_output_device = Some(name);
                self.user_selected_output = true;
            }
            Message::DevicePanelToggled(show) => {
                self.show_device_selector = show;
            }
            Message::EqBandToggleExpand(idx) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_expanded[idx] = !self.eq_band_expanded[idx];
                }
            }
            Message::EqBandToggleAdvanced(idx) => {
                if idx < MAX_EQ_BANDS {
                    self.eq_band_show_advanced[idx] = !self.eq_band_show_advanced[idx];
                }
            }
            Message::EqToggleAdvanced => {
                self.eq_show_advanced = !self.eq_show_advanced;
            }
            Message::EqResetBands => self.reset_eq_band_gains(),
            Message::EqStatusUpdated(status) => {
                self.agc_current_gain = status.agc_gain_db;
                self.eq_status = status;
            }
            Message::ScenePresetChanged(scene) => {
                self.scene_preset = scene;
                self.apply_scene(scene);
            }
            Message::MutePlaybackChanged(muted) => {
                self.mute_playback = muted;
                self.send_control_message(ControlMessage::MutePlayback(muted));
            }
            Message::BypassToggled(enabled) => {
                self.bypass_enabled = enabled;
                self.send_control_message(ControlMessage::BypassEnabled(enabled));
            }
            Message::AutoPlayToggled(enabled) => {
                self.auto_play_enabled = enabled;
            }
            Message::AutoPlayPickRequested => {
                let picked = rfd::FileDialog::new()
                    .add_filter("音频", &["wav", "mp3", "flac", "m4a"])
                    .pick_file();
                return Command::perform(async move { picked }, Message::AutoPlayFilePicked);
            }
            Message::AutoPlayFilePicked(path) => {
                self.auto_play_file = path;
            }
            Message::HighpassToggled(enabled) => {
                self.highpass_enabled = enabled;
                self.send_control_message(ControlMessage::HighpassEnabled(enabled));
                if enabled {
                    self.send_control_message(ControlMessage::HighpassCutoff(self.highpass_cutoff));
                }
            }
            Message::HighpassCutoffChanged(freq) => {
                self.highpass_cutoff = freq;
                self.send_control_message(ControlMessage::HighpassCutoff(freq));
            }
            Message::HeadroomChanged(v) => {
                self.headroom_gain = v;
                self.send_control_message(ControlMessage::HeadroomGain(v));
            }
            Message::SpectrumToggled(enabled) => {
                self.show_spectrum = enabled;
                if !enabled {
                    self.reset_spec_images();
                }
                self.send_control_message(ControlMessage::SpecEnabled(enabled));
            }
            Message::Rt60Toggled(enabled) => {
                self.rt60_enabled = enabled;
                self.send_control_message(ControlMessage::Rt60Enabled(enabled));
            }
            Message::FinalLimiterToggled(enabled) => {
                self.final_limiter_enabled = enabled;
                self.send_control_message(ControlMessage::FinalLimiterEnabled(enabled));
            }
            Message::SaturationToggled(enabled) => {
                self.saturation_enabled = enabled;
                self.send_control_message(ControlMessage::SaturationEnabled(enabled));
            }
            Message::SaturationDriveChanged(v) => {
                self.saturation_drive = v;
                self.send_control_message(ControlMessage::SaturationDrive(v));
            }
            Message::SaturationMakeupChanged(v) => {
                self.saturation_makeup = v;
                self.send_control_message(ControlMessage::SaturationMakeup(v));
            }
            Message::SaturationMixChanged(v) => {
                self.saturation_mix = v;
                self.send_control_message(ControlMessage::SaturationMix(v));
            }
            Message::SaturationToggleAdvanced => {
                self.show_saturation_advanced = !self.show_saturation_advanced;
            }
            Message::TransientSustainChanged(db) => {
                self.transient_sustain = db;
                self.send_control_message(ControlMessage::TransientSustain(db));
            }
            Message::TransientToggled(enabled) => {
                self.transient_enabled = enabled;
                self.send_control_message(ControlMessage::TransientEnabled(enabled));
            }
            Message::TransientGainChanged(db) => {
                self.transient_gain = db;
                self.send_control_message(ControlMessage::TransientGain(db));
            }
            Message::TransientMixChanged(ratio) => {
                self.transient_mix = ratio;
                self.send_control_message(ControlMessage::TransientMix(ratio));
            }
            Message::TransientToggleAdvanced => {
                self.show_transient_advanced = !self.show_transient_advanced;
            }
            Message::AgcToggled(enabled) => {
                self.agc_enabled = enabled;
                self.send_control_message(ControlMessage::AgcEnabled(enabled));
            }
            Message::AgcTargetChanged(value) => {
                self.agc_target_db = value;
                self.send_control_message(ControlMessage::AgcTargetLevel(value));
            }
            Message::AgcMaxGainChanged(value) => {
                self.agc_max_gain_db = value;
                self.send_control_message(ControlMessage::AgcMaxGain(value));
            }
            Message::AgcMaxAttenChanged(value) => {
                self.agc_max_atten_db = value;
                self.send_control_message(ControlMessage::AgcMaxAttenuation(value));
            }
            Message::AgcWindowChanged(value) => {
                self.agc_window_sec = value;
                self.send_control_message(ControlMessage::AgcWindowSeconds(value));
            }
            Message::AgcAttackChanged(value) => {
                self.agc_attack_ms = value;
                self.send_control_message(ControlMessage::AgcAttackRelease(
                    value,
                    self.agc_release_ms,
                ));
            }
            Message::AgcReleaseChanged(value) => {
                self.agc_release_ms = value;
                self.send_control_message(ControlMessage::AgcAttackRelease(
                    self.agc_attack_ms,
                    value,
                ));
            }
            Message::AgcToggleAdvanced => {
                self.show_agc_advanced = !self.show_agc_advanced;
            }
            Message::SysAutoVolumeToggled(enabled) => {
                self.sys_auto_volume = enabled;
                self.send_control_message(ControlMessage::SysAutoVolumeEnabled(enabled));
                self.ensure_sys_volume_monitor();
            }
            Message::EnvAutoToggled(enabled) => {
                self.env_auto_enabled = enabled;
                self.env_status_label = if enabled {
                    "自适应降噪: 正常".to_string()
                } else {
                    "自适应降噪: 关闭".to_string()
                };
                self.send_control_message(ControlMessage::EnvAutoEnabled(enabled));
            }
            Message::VadToggled(enabled) => {
                self.vad_enabled = enabled;
                self.send_control_message(ControlMessage::VadEnabled(enabled));
            }
            Message::ExciterToggled(enabled) => {
                self.exciter_enabled = enabled;
                self.send_control_message(ControlMessage::ExciterEnabled(enabled));
            }
            Message::ExciterMixChanged(value) => {
                self.exciter_mix = value.clamp(0.0, 0.5);
                self.send_control_message(ControlMessage::ExciterMix(self.exciter_mix));
            }
            Message::AecToggled(enabled) => {
                self.aec_enabled = enabled;
                self.send_control_message(ControlMessage::AecEnabled(enabled));
            }
            Message::AecAggressiveToggled(enabled) => {
                self.aec_aggressive = enabled;
                self.send_control_message(ControlMessage::AecAggressive(enabled));
            }
            Message::AecDelayChanged(v) => {
                self.aec_delay_ms = v;
                self.send_control_message(ControlMessage::AecDelayMs(v as i32));
            }
            Message::TimbreToggled(enabled) => {
                self.timbre_enabled = enabled;
                self.send_control_message(ControlMessage::TimbreEnabled(enabled));
            }
            Message::Tick => {
                // 确保后台系统音量监测在未降噪时也保持运行
                self.ensure_sys_volume_monitor();
                let mut commands = Vec::new();
                if let Some(task) = self.update_lsnr() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_noisy() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_enh() {
                    commands.push(Command::perform(task, move |message| message))
                }
                if let Some(task) = self.update_eq_status() {
                    commands.push(Command::perform(task, move |message| message));
                }
                if let Some(task) = self.update_env_status() {
                    commands.push(Command::perform(task, move |message| message));
                }
                return Command::batch(commands);
            }
            Message::EnvStatusUpdated(status) => {
                self.env_status_label = match status {
                    EnvStatus::Normal => "自适应降噪: 正常".to_string(),
                    EnvStatus::Soft => "自适应降噪: 柔和".to_string(),
                };
            }
            Message::LsnrChanged(lsnr) => self.lsnr = lsnr,
            Message::NoisyChanged => {
                self.noisy_img =
                    spec_image_handle(&self.noisy_spec, self.spec_frames, self.spec_freqs);
            }
            Message::EnhChanged => {
                self.enh_img = spec_image_handle(&self.enh_spec, self.spec_frames, self.spec_freqs);
            }
            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                self.send_df_control(DfControl::AttenLim, v);
            }
            Message::PostFilterChanged(v) => {
                self.post_filter_beta = v;
                self.send_df_control(DfControl::PostFilterBeta, v);
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                self.send_df_control(DfControl::MinThreshDb, v);
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                self.send_df_control(DfControl::MaxErbThreshDb, v);
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                self.send_df_control(DfControl::MaxDfThreshDb, v);
            }
            Message::SliderInputChanged {
                key,
                raw,
                target,
                min,
                max,
                precision,
            } => {
                self.set_buffer(&key, raw.clone());
                if let Ok(parsed) = raw.parse::<f32>() {
                    let clamped = parsed.clamp(min, max);
                    self.apply_slider_value(target, clamped);
                    self.set_buffer(&key, format!("{:.precision$}", clamped));
                }
            }
            Message::DfMixChanged(_v) => {
                // 强制全湿，忽略传入比例，避免干/湿并行带来的相位梳状
                self.df_mix = 1.0;
                self.send_control_message(ControlMessage::DfMix(self.df_mix));
            }
            Message::NoiseAdvancedToggled => {
                self.noise_show_advanced = !self.noise_show_advanced;
            }
            Message::PlaybackFinished(result) => {
                if let Err(err) = result {
                    log::warn!("播放测试音频失败: {}", err);
                }
                self.auto_play_pid = None;
                if self.is_running {
                    return self.stop_processing();
                }
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let mut start_btn_widget = button("开始降噪");
        if !self.is_running && !self.is_saving {
            start_btn_widget = start_btn_widget.on_press(Message::StartProcessing);
        }
        let start_btn = apply_tooltip(
            start_btn_widget.width(Length::Shrink),
            tooltips::START_BUTTON,
        );
        let mut stop_btn_widget = button("结束降噪");
        if self.is_running && !self.is_saving {
            stop_btn_widget = stop_btn_widget.on_press(Message::StopProcessing);
        }
        let stop_btn = apply_tooltip(stop_btn_widget.width(Length::Shrink), tooltips::STOP_BUTTON);
        let save_cfg_btn = button("保存配置").on_press(Message::SaveConfigRequested);
        let load_cfg_btn = button("加载配置").on_press(Message::LoadConfigRequested);
        let exit_btn = apply_tooltip(
            button("退出").on_press(Message::Exit),
            tooltips::EXIT_BUTTON,
        );

        let mute_label = if self.mute_playback {
            "🔇 播放已静音"
        } else {
            "🔊 播放正常"
        };
        let mute_toggle = apply_tooltip(
            toggler(
                Some(mute_label.to_string()),
                self.mute_playback,
                Message::MutePlaybackChanged,
            ),
            tooltips::MUTE_PLAYBACK,
        );
        let auto_play_path = self
            .auto_play_file
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "未选择文件".to_string());
        let auto_play_btn = button("选择播放音频").on_press(Message::AutoPlayPickRequested);
        let auto_play_row = row![
            toggler(None, self.auto_play_enabled, Message::AutoPlayToggled),
            text("自动播放测试音频").size(14),
            text(auto_play_path).size(12).width(Length::Fill),
            auto_play_btn
        ]
        .spacing(8)
        .align_items(Alignment::Center);

        let device_toggle = toggler(
            Some("显示设备选择".to_string()),
            self.show_device_selector,
            Message::DevicePanelToggled,
        );
        let device_row = if self.show_device_selector {
            let input_filter = self.input_device_filter.to_lowercase();
            let output_filter = self.output_device_filter.to_lowercase();
            let filtered_inputs: Vec<String> = self
                .input_devices
                .iter()
                .filter(|name| name.to_lowercase().contains(&input_filter))
                .cloned()
                .collect();
            let filtered_outputs: Vec<String> = self
                .output_devices
                .iter()
                .filter(|name| name.to_lowercase().contains(&output_filter))
                .cloned()
                .collect();
            row![
                column![
                    text("输入设备过滤:").size(12),
                    widget::text_input("关键字匹配输入设备", &self.input_device_filter)
                        .on_input(Message::InputDeviceChanged)
                        .padding(6)
                        .size(14),
                    pick_list(
                        filtered_inputs.clone(),
                        self.selected_input_device.clone(),
                        Message::InputDeviceSelected
                    )
                    .placeholder("选择输入设备")
                    .width(Length::Fill),
                ]
                .spacing(4)
                .width(Length::Fill),
                column![
                    text("输出设备过滤:").size(12),
                    widget::text_input("关键字匹配输出设备", &self.output_device_filter)
                        .on_input(Message::OutputDeviceChanged)
                        .padding(6)
                        .size(14),
                    pick_list(
                        filtered_outputs.clone(),
                        self.selected_output_device.clone(),
                        Message::OutputDeviceSelected
                    )
                    .placeholder("选择输出设备")
                    .width(Length::Fill),
                ]
                .spacing(4)
                .width(Length::Fill),
            ]
            .spacing(12)
        } else {
            row![].spacing(0)
        };
        let mut header = column![
            row![
                text("DeepFilterNet 实时拾音").size(40).width(Length::Fill),
                exit_btn,
            ]
            .spacing(20)
            .align_items(Alignment::Center),
            row![
                start_btn,
                stop_btn,
                save_cfg_btn,
                load_cfg_btn,
                mute_toggle,
                toggler(String::new(), self.bypass_enabled, Message::BypassToggled),
                text("全链路旁路").size(16),
                text(self.env_status_label.clone()).size(18),
                text(format!("状态: {}", self.status_text))
                    .size(20)
                    .width(Length::Fill)
                    .horizontal_alignment(alignment::Horizontal::Right)
            ]
            .spacing(20)
            .align_items(Alignment::Center),
            row![auto_play_row].spacing(12).align_items(Alignment::Center),
            row![device_toggle, container(device_row).width(Length::Fill)]
                .spacing(12)
                .align_items(Alignment::Center),
        ];

        if let Some((raw, _, _, _)) = &self.last_saved {
            if let Some(dir) = raw.parent() {
                header = header.push(text(format!("文件已保存至: {}", dir.display())).size(16));
            } else {
                header = header.push(text("文件已保存").size(16));
            }
        }

        let controls_panel = scrollable(
            container(self.create_eq_panel())
                .width(Length::Fixed(420.0))
                .height(Length::Fill),
        )
        .width(Length::Fixed(420.0))
        .height(Length::Fill);

        let main_content = row![
            controls_panel,
            container(self.create_spectrum_panel())
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(20),
        ]
        .spacing(20)
        .height(Length::Fill);

        container(column![header, main_content].spacing(20))
            .padding(20)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(20)).map(|_| Message::Tick)
    }
}

impl SpecView {
    fn refresh_devices(&mut self) {
        let (inputs, outputs, default_in, default_out) = list_audio_devices();
        self.input_devices = inputs;
        self.output_devices = outputs;
        if self
            .selected_input_device
            .as_ref()
            .map(|d| !self.input_devices.contains(d))
            .unwrap_or(true)
            || !self.user_selected_input
        {
            self.selected_input_device = default_in;
        }
        if self
            .selected_output_device
            .as_ref()
            .map(|d| !self.output_devices.contains(d))
            .unwrap_or(true)
            || !self.user_selected_output
        {
            self.selected_output_device = default_out;
        }
    }

    fn start_processing(&mut self) -> Command<Message> {
        if self.is_running || self.is_saving {
            return Command::none();
        }
        // 刷新设备列表，确保使用当前设备
        self.refresh_devices();
        let (s_lsnr, r_lsnr) = unbounded();
        let (s_noisy, r_noisy) = unbounded();
        let (s_enh, r_enh) = unbounded();
        let (s_controls, r_controls) = unbounded();
        let (s_eq_status, r_eq_status) = unbounded();
        let (s_env_status, r_env_status) = unbounded();
        let model_path = current_model_path().or_else(|| self.model_path.clone());
        self.model_path = model_path.clone();
        let input_device = self.selected_input_device.clone().or_else(|| {
            let trimmed = self.input_device_filter.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        let output_device = self.selected_output_device.clone().or_else(|| {
            let trimmed = self.output_device_filter.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        let spec_channels = if self.show_spectrum {
            Some((s_noisy, s_enh))
        } else {
            None
        };
        let r_noisy = if self.show_spectrum { Some(r_noisy) } else { None };
        let r_enh = if self.show_spectrum { Some(r_enh) } else { None };

        let df_worker = match DeepFilterCapture::new(
            model_path,
            Some(s_lsnr),
            spec_channels.as_ref().map(|(n, _)| n.clone()),
            spec_channels.as_ref().map(|(_, e)| e.clone()),
            Some(r_controls),
            Some(s_eq_status),
            Some(s_env_status),
            input_device,
            output_device,
        ) {
            Ok(worker) => worker,
            Err(err) => {
                self.status_text = format!("启动失败: {}", err);
                log::error!("启动降噪失败: {}", err);
                return Command::none();
            }
        };
        let recording = df_worker.recording();
        self.recording = Some(recording);
        self.df_worker = Some(df_worker);
        self.r_lsnr = Some(r_lsnr);
        self.r_noisy = r_noisy;
        self.r_enh = r_enh;
        self.r_eq_status = Some(r_eq_status);
        self.r_env_status = Some(r_env_status);
        self.s_controls = Some(s_controls);
        self.is_running = true;
        self.status_text = "实时降噪中".to_string();
        self.last_saved = None;
        self.reset_spec_images();
        self.eq_status = EqStatus::default();
        self.send_df_control(DfControl::AttenLim, self.atten_lim);
        self.send_df_control(DfControl::PostFilterBeta, self.post_filter_beta);
        self.send_df_control(DfControl::MinThreshDb, self.min_threshdb);
        self.send_df_control(DfControl::MaxErbThreshDb, self.max_erbthreshdb);
        self.send_df_control(DfControl::MaxDfThreshDb, self.max_dfthreshdb);
        self.send_control_message(ControlMessage::DfMix(self.df_mix));
        self.send_control_message(ControlMessage::HeadroomGain(self.headroom_gain));
        self.send_control_message(ControlMessage::PostTrimGain(self.post_trim_gain));
        self.send_eq_control(EqControl::SetEnabled(self.eq_enabled));
        self.send_eq_control(EqControl::SetPreset(self.eq_preset));
        self.send_eq_control(EqControl::SetDryWet(self.eq_dry_wet));
        self.broadcast_eq_parameters();
        self.broadcast_eq_band_gains();
        self.send_control_message(ControlMessage::HighpassEnabled(self.highpass_enabled));
        self.send_control_message(ControlMessage::HighpassCutoff(self.highpass_cutoff));
        self.send_control_message(ControlMessage::SaturationEnabled(self.saturation_enabled));
        self.send_control_message(ControlMessage::SaturationDrive(self.saturation_drive));
        self.send_control_message(ControlMessage::SaturationMakeup(self.saturation_makeup));
        self.send_control_message(ControlMessage::SaturationMix(self.saturation_mix));
        self.send_control_message(ControlMessage::ExciterEnabled(self.exciter_enabled));
        self.send_control_message(ControlMessage::ExciterMix(self.exciter_mix));
        self.send_control_message(ControlMessage::TimbreEnabled(self.timbre_enabled));
        self.send_control_message(ControlMessage::BypassEnabled(self.bypass_enabled));
        self.send_control_message(ControlMessage::TransientEnabled(self.transient_enabled));
        self.send_control_message(ControlMessage::TransientGain(self.transient_gain));
        self.send_control_message(ControlMessage::TransientSustain(self.transient_sustain));
        self.send_control_message(ControlMessage::TransientMix(self.transient_mix));
        self.send_control_message(ControlMessage::HeadroomGain(self.headroom_gain));
        self.send_control_message(ControlMessage::FinalLimiterEnabled(
            self.final_limiter_enabled,
        ));
        self.send_control_message(ControlMessage::SpecEnabled(self.show_spectrum));
        self.send_control_message(ControlMessage::Rt60Enabled(self.rt60_enabled));
        self.send_control_message(ControlMessage::AecAggressive(self.aec_aggressive));
        self.send_control_message(ControlMessage::AgcTargetLevel(self.agc_target_db));
        self.send_control_message(ControlMessage::AgcMaxGain(self.agc_max_gain_db));
        self.send_control_message(ControlMessage::AgcMaxAttenuation(self.agc_max_atten_db));
        self.send_control_message(ControlMessage::AgcWindowSeconds(self.agc_window_sec));
        self.send_control_message(ControlMessage::AgcAttackRelease(
            self.agc_attack_ms,
            self.agc_release_ms,
        ));
        self.send_control_message(ControlMessage::AgcEnabled(self.agc_enabled));
        self.send_control_message(ControlMessage::AecEnabled(self.aec_enabled));
        self.send_control_message(ControlMessage::AecDelayMs(self.aec_delay_ms as i32));
        self.send_control_message(ControlMessage::MutePlayback(self.mute_playback));
        self.send_control_message(ControlMessage::SysAutoVolumeEnabled(self.sys_auto_volume));
        self.send_control_message(ControlMessage::EnvAutoEnabled(self.env_auto_enabled));
        self.send_control_message(ControlMessage::VadEnabled(self.vad_enabled));
        let mut cmds: Vec<Command<Message>> = Vec::new();
        if self.auto_play_enabled {
            if let Some(path) = self.auto_play_file.clone() {
                if path.exists() {
                    match StdCommand::new("afplay").arg(&path).spawn() {
                        Ok(mut child) => {
                            self.auto_play_pid = Some(child.id());
                            cmds.push(Command::perform(
                                async move {
                                    let status = child.wait().map_err(|e| e.to_string())?;
                                    if status.success() {
                                        Ok(())
                                    } else {
                                        Err(format!("播放进程退出状态 {}", status))
                                    }
                                },
                                Message::PlaybackFinished,
                            ));
                        }
                        Err(err) => {
                            log::warn!("启动播放失败: {}", err);
                        }
                    }
                } else {
                    log::warn!("自动播放文件不存在: {}", path.display());
                }
            } else {
                log::warn!("已启用自动播放测试音频，但未选择文件");
            }
        }
        Command::batch(cmds)
    }

    fn stop_processing(&mut self) -> Command<Message> {
        if !self.is_running {
            return Command::none();
        }
        if let Some(pid) = self.auto_play_pid.take() {
            kill_pid(pid);
        }
        let mut worker = if let Some(worker) = self.df_worker.take() {
            worker
        } else {
            return Command::none();
        };
        self.is_running = false;
        self.status_text = "正在停止...".to_string();
        if let Err(err) = worker.should_stop() {
            self.status_text = format!("停止失败: {}", err);
            log::error!("停止降噪失败: {}", err);
            return Command::none();
        }
        let recording = self.recording.take().unwrap_or_else(|| worker.recording());
        self.s_controls = None;
        self.r_lsnr = None;
        self.r_noisy = None;
        self.r_enh = None;
        self.r_eq_status = None;
        self.r_env_status = None;
        self.eq_status = EqStatus::default();
        let sample_rate = recording.sample_rate() as u32;
        let (noisy, denoised, timbre, processed) = recording.take_samples();
        self.status_text = "正在保存音频...".to_string();
        self.last_saved = None;
        self.is_saving = true;
        let timbre_enabled = self.timbre_enabled;
        Command::perform(
            async move {
                save_recordings(noisy, denoised, timbre, processed, sample_rate, timbre_enabled)
            },
            Message::SaveFinished,
        )
    }

    fn reset_spec_images(&mut self) {
        let (noisy_spec, noisy_img) = create_spec_storage(self.spec_frames, self.spec_freqs);
        let (enh_spec, enh_img) = create_spec_storage(self.spec_frames, self.spec_freqs);
        self.noisy_spec = noisy_spec;
        self.enh_spec = enh_spec;
        self.noisy_img = noisy_img;
        self.enh_img = enh_img;
    }

    fn send_df_control(&self, control: DfControl, value: f32) {
        self.send_control_message(ControlMessage::DeepFilter(control, value));
    }

    fn send_eq_control(&self, control: EqControl) {
        self.send_control_message(ControlMessage::Eq(control));
    }

    fn send_control_message(&self, message: ControlMessage) {
        if let Some(sender) = self.s_controls.as_ref() {
            if let Err(err) = sender.send(message) {
                log::warn!("发送控制参数失败: {}", err);
            }
        }
    }

    fn update_lsnr(&mut self) -> Option<impl Future<Output = Message>> {
        let Some(recv) = self.r_lsnr.as_ref().cloned() else {
            return None;
        };
        if recv.is_empty() {
            return None;
        }
        Some(async move {
            sleep(Duration::from_millis(100));
            let mut lsnr = 0.;
            let mut n = 0;
            while let Ok(v) = recv.try_recv() {
                lsnr += v;
                n += 1;
            }
            if n > 0 {
                lsnr /= n as f32;
                Message::LsnrChanged(lsnr)
            } else {
                Message::None
            }
        })
    }

    fn update_noisy(&mut self) -> Option<impl Future<Output = Message>> {
        let Some(recv) = self.r_noisy.as_ref().cloned() else {
            return None;
        };
        if recv.is_empty() {
            return None;
        }
        let spec = self.noisy_spec.clone();
        Some(async move {
            let n = recv.len();
            if let Ok(mut guard) = spec.lock() {
                guard.update(recv.iter().take(n));
            } else {
                log::error!("无法锁定拾音频谱缓存");
            }
            Message::NoisyChanged
        })
    }

    fn update_enh(&mut self) -> Option<impl Future<Output = Message>> {
        let Some(recv) = self.r_enh.as_ref().cloned() else {
            return None;
        };
        if recv.is_empty() {
            return None;
        }
        let spec = self.enh_spec.clone();
        Some(async move {
            let n = recv.len();
            if let Ok(mut guard) = spec.lock() {
                guard.update(recv.iter().take(n));
            } else {
                log::error!("无法锁定降噪频谱缓存");
            }
            Message::EnhChanged
        })
    }

    fn apply_eq_preset_config(&mut self, preset: EqPresetKind) {
        let config = preset.preset();
        for (i, band) in config.bands.iter().enumerate() {
            if i >= MAX_EQ_BANDS {
                break;
            }
            self.eq_band_gains[i] = band.static_gain_db;
            self.eq_band_frequencies[i] = band.frequency_hz;
            self.eq_band_qs[i] = band.q;
            self.eq_band_detector_qs[i] = band.detector_q();
            self.eq_band_thresholds[i] = band.threshold_db;
            self.eq_band_ratios[i] = band.ratio;
            self.eq_band_max_gains[i] = band.max_gain_db;
            self.eq_band_attacks[i] = band.attack_ms;
            self.eq_band_releases[i] = band.release_ms;
            self.eq_band_makeups[i] = band.makeup_db;
            self.eq_band_modes[i] = band.mode;
            self.eq_band_filters[i] = band.filter;
        }
    }

    fn update_eq_status(&mut self) -> Option<impl Future<Output = Message>> {
        let Some(recv) = self.r_eq_status.as_ref().cloned() else {
            return None;
        };
        if recv.is_empty() {
            return None;
        }
        Some(async move {
            let mut latest = None;
            while let Ok(status) = recv.try_recv() {
                latest = Some(status);
            }
            latest.map_or(Message::None, Message::EqStatusUpdated)
        })
    }

    fn update_env_status(&mut self) -> Option<impl Future<Output = Message>> {
        let Some(recv) = self.r_env_status.as_ref().cloned() else {
            return None;
        };
        if recv.is_empty() {
            return None;
        }
        Some(async move {
            let mut latest = None;
            while let Ok(status) = recv.try_recv() {
                latest = Some(status);
            }
            latest.map_or(Message::None, Message::EnvStatusUpdated)
        })
    }

    fn eq_status_label(&self) -> String {
        if !self.eq_enabled {
            "⚪️ 已旁路".to_string()
        } else if !self.eq_status.enabled {
            "🟡 等待音频".to_string()
        } else if self.eq_status.cpu_load > 40.0 {
            format!("🟡 CPU {:.1}%", self.eq_status.cpu_load.min(100.0))
        } else {
            "🟢 正常工作".to_string()
        }
    }

    fn set_eq_band_gain(&mut self, index: usize, gain_db: f32) {
        if index >= MAX_EQ_BANDS {
            return;
        }
        const STEP: f32 = 0.5;
        let clamped = gain_db.clamp(-12.0, 12.0);
        let quantized = (clamped / STEP).round() * STEP;
        let gain = quantized.clamp(-12.0, 12.0);
        let needs_send = (self.eq_band_gains[index] - gain).abs() > 1e-3;
        self.eq_band_gains[index] = gain;
        if needs_send {
            self.send_eq_control(EqControl::SetBandGain(index, gain));
        }
    }

    fn reset_eq_band_gains(&mut self) {
        let preset = self.eq_preset.preset();
        for (idx, band) in preset.bands.iter().enumerate() {
            self.set_eq_band_gain(idx, band.static_gain_db);
        }
    }

    fn apply_scene(&mut self, scene: ScenePreset) {
        match scene {
            ScenePreset::ConferenceHall => {
                self.highpass_cutoff = 50.0;
                self.atten_lim = 20.0;
                self.min_threshdb = -50.0;
                self.max_erbthreshdb = 20.0;
                self.max_dfthreshdb = 20.0;
                self.df_mix = 1.0;
                self.headroom_gain = 1.0;
                self.post_trim_gain = 1.0;
                self.transient_gain = 2.0;
                self.transient_sustain = -4.0;
                self.transient_mix = 100.0;
                self.saturation_drive = 1.3;
                self.saturation_makeup = -0.5;
                self.saturation_mix = 100.0;
                self.agc_target_db = -3.0;
                self.agc_max_gain_db = 6.0;
                self.eq_preset = EqPresetKind::ConferenceHall;
                self.eq_dry_wet = 1.0;
                self.apply_eq_preset_config(self.eq_preset);
            }
            ScenePreset::OpenOfficeMeeting => {
                // 开放式办公区：重降噪、高通抬升，激励保守
                self.highpass_cutoff = 120.0;
                self.atten_lim = 80.0;
                self.min_threshdb = -55.0;
                self.max_erbthreshdb = 10.0;
                self.max_dfthreshdb = 10.0;
                self.df_mix = 1.0;
                self.headroom_gain = 1.0;
                self.post_trim_gain = 1.0;
                self.transient_gain = 3.0;
                self.transient_sustain = -2.0;
                self.transient_mix = 100.0;
                self.saturation_drive = 1.0;
                self.saturation_makeup = -0.5;
                self.saturation_mix = 100.0;
                self.agc_target_db = -6.0;
                self.agc_max_gain_db = 12.0;
                self.agc_max_atten_db = 12.0;
                self.agc_window_sec = 0.6;
                self.agc_attack_ms = 400.0;
                self.agc_release_ms = 1500.0;
                self.eq_preset = EqPresetKind::OpenOffice;
                self.eq_dry_wet = 1.0;
                self.apply_eq_preset_config(self.eq_preset);
                self.exciter_enabled = true;
                self.exciter_mix = 0.1;
                self.env_auto_enabled = true;
                self.vad_enabled = true;
            }
            ScenePreset::OpenOfficeHeadset => {
                // 开放办公-耳机：抑制呼吸/衣物摩擦，保持近讲清晰
                self.highpass_cutoff = 150.0;
                self.atten_lim = 85.0;
                self.min_threshdb = -55.0;
                self.max_erbthreshdb = 10.0;
                self.max_dfthreshdb = 10.0;
                self.df_mix = 1.0;
                self.headroom_gain = 0.85;
                self.post_trim_gain = 1.0;
                self.transient_gain = 2.0;
                self.transient_sustain = -3.0;
                self.transient_mix = 80.0;
                self.saturation_drive = 1.0;
                self.saturation_makeup = -0.5;
                self.saturation_mix = 100.0;
                self.agc_target_db = -6.0;
                self.agc_max_gain_db = 12.0;
                self.agc_max_atten_db = 12.0;
                self.agc_window_sec = 0.5;
                self.agc_attack_ms = 300.0;
                self.agc_release_ms = 1200.0;
                self.eq_preset = EqPresetKind::OpenOfficeHeadset;
                self.eq_dry_wet = 1.0;
                self.apply_eq_preset_config(self.eq_preset);
                self.exciter_enabled = true;
                self.exciter_mix = 0.08; // 耳机轻激励，防止嘶声
                self.env_auto_enabled = true;
                self.vad_enabled = true;
                self.highpass_enabled = true;
                self.agc_enabled = true;
                self.saturation_enabled = true;
                self.transient_enabled = true;
            }
        }
        self.input_buffers.clear();
        self.sync_runtime_controls();
    }

    fn sync_runtime_controls(&self) {
        self.send_control_message(ControlMessage::HighpassEnabled(self.highpass_enabled));
        self.send_control_message(ControlMessage::HighpassCutoff(self.highpass_cutoff));
        self.send_control_message(ControlMessage::DfMix(self.df_mix));
        self.send_control_message(ControlMessage::HeadroomGain(self.headroom_gain));
        self.send_control_message(ControlMessage::SpecEnabled(self.show_spectrum));
        self.send_control_message(ControlMessage::Rt60Enabled(self.rt60_enabled));
        self.send_control_message(ControlMessage::PostTrimGain(self.post_trim_gain));
        self.send_control_message(ControlMessage::SaturationEnabled(self.saturation_enabled));
        self.send_control_message(ControlMessage::SaturationDrive(self.saturation_drive));
        self.send_control_message(ControlMessage::SaturationMakeup(self.saturation_makeup));
        self.send_control_message(ControlMessage::SaturationMix(self.saturation_mix));
        self.send_control_message(ControlMessage::ExciterEnabled(self.exciter_enabled));
        self.send_control_message(ControlMessage::ExciterMix(self.exciter_mix));
        self.send_control_message(ControlMessage::TransientEnabled(self.transient_enabled));
        self.send_control_message(ControlMessage::TransientGain(self.transient_gain));
        self.send_control_message(ControlMessage::TransientSustain(self.transient_sustain));
        self.send_control_message(ControlMessage::TransientMix(self.transient_mix));
        self.send_df_control(DfControl::AttenLim, self.atten_lim);
        self.send_df_control(DfControl::MinThreshDb, self.min_threshdb);
        self.send_df_control(DfControl::MaxErbThreshDb, self.max_erbthreshdb);
        self.send_df_control(DfControl::MaxDfThreshDb, self.max_dfthreshdb);
        self.send_control_message(ControlMessage::AecEnabled(self.aec_enabled));
        self.send_control_message(ControlMessage::AecDelayMs(self.aec_delay_ms as i32));
        self.send_eq_control(EqControl::SetEnabled(self.eq_enabled));
        self.send_eq_control(EqControl::SetPreset(self.eq_preset));
        self.send_eq_control(EqControl::SetDryWet(self.eq_dry_wet));
        self.broadcast_eq_parameters();
        self.broadcast_eq_band_gains();
        self.send_control_message(ControlMessage::AgcEnabled(self.agc_enabled));
        self.send_control_message(ControlMessage::AgcTargetLevel(self.agc_target_db));
        self.send_control_message(ControlMessage::AgcMaxGain(self.agc_max_gain_db));
        self.send_control_message(ControlMessage::AgcMaxAttenuation(self.agc_max_atten_db));
        self.send_control_message(ControlMessage::AgcWindowSeconds(self.agc_window_sec));
        self.send_control_message(ControlMessage::AgcAttackRelease(
            self.agc_attack_ms,
            self.agc_release_ms,
        ));
        self.send_control_message(ControlMessage::SysAutoVolumeEnabled(self.sys_auto_volume));
        self.send_control_message(ControlMessage::EnvAutoEnabled(self.env_auto_enabled));
        self.send_control_message(ControlMessage::VadEnabled(self.vad_enabled));
    }

    fn ensure_sys_volume_monitor(&mut self) {
        if self.sys_auto_volume {
            if self.sysvol_monitor.is_none() {
                match capture::start_sys_volume_monitor(self.selected_input_device.clone()) {
                    Some(handle) => {
                        log::info!("后台系统音量监测已启动");
                        self.sysvol_monitor = Some(handle);
                    }
                    None => {
                        log::warn!("后台系统音量监测未启动（可能无权限或无可用设备）");
                    }
                }
            }
        } else if let Some(handle) = self.sysvol_monitor.take() {
            handle.stop();
            log::info!("后台系统音量监测已停止");
        }
    }

    fn restart_sys_volume_monitor(&mut self) {
        if let Some(handle) = self.sysvol_monitor.take() {
            handle.stop();
        }
        self.ensure_sys_volume_monitor();
    }

    fn broadcast_eq_parameters(&self) {
        for idx in 0..MAX_EQ_BANDS {
            self.send_eq_control(EqControl::SetBandFrequency(
                idx,
                self.eq_band_frequencies[idx],
            ));
            self.send_eq_control(EqControl::SetBandQ(idx, self.eq_band_qs[idx]));
            self.send_eq_control(EqControl::SetBandDetectorQ(
                idx,
                self.eq_band_detector_qs[idx],
            ));
            self.send_eq_control(EqControl::SetBandThreshold(
                idx,
                self.eq_band_thresholds[idx],
            ));
            self.send_eq_control(EqControl::SetBandRatio(idx, self.eq_band_ratios[idx]));
            self.send_eq_control(EqControl::SetBandMaxGain(idx, self.eq_band_max_gains[idx]));
            self.send_eq_control(EqControl::SetBandAttack(idx, self.eq_band_attacks[idx]));
            self.send_eq_control(EqControl::SetBandRelease(idx, self.eq_band_releases[idx]));
            self.send_eq_control(EqControl::SetBandMakeup(idx, self.eq_band_makeups[idx]));
            self.send_eq_control(EqControl::SetBandMode(idx, self.eq_band_modes[idx]));
            self.send_eq_control(EqControl::SetBandFilter(idx, self.eq_band_filters[idx]));
        }
    }

    fn broadcast_eq_band_gains(&self) {
        for (idx, gain) in self.eq_band_gains.iter().enumerate() {
            self.send_eq_control(EqControl::SetBandGain(idx, *gain));
        }
    }

    fn to_user_config(&self) -> UserConfig {
        UserConfig {
            version: CONFIG_VERSION,
            scene_preset: self.scene_preset,
            atten_lim: self.atten_lim,
            post_filter_beta: self.post_filter_beta,
            min_threshdb: self.min_threshdb,
            max_erbthreshdb: self.max_erbthreshdb,
            max_dfthreshdb: self.max_dfthreshdb,
            df_mix: 1.0,
            headroom_gain: self.headroom_gain,
            show_spectrum: self.show_spectrum,
            rt60_enabled: self.rt60_enabled,
            final_limiter_enabled: self.final_limiter_enabled,
            aec_aggressive: self.aec_aggressive,
            post_trim_gain: self.post_trim_gain,
            eq_enabled: self.eq_enabled,
            eq_preset: self.eq_preset,
            eq_dry_wet: 1.0,
            eq_band_gains: self.eq_band_gains,
            eq_band_frequencies: self.eq_band_frequencies,
            eq_band_qs: self.eq_band_qs,
            eq_band_detector_qs: self.eq_band_detector_qs,
            eq_band_thresholds: self.eq_band_thresholds,
            eq_band_ratios: self.eq_band_ratios,
            eq_band_max_gains: self.eq_band_max_gains,
            eq_band_attacks: self.eq_band_attacks,
            eq_band_releases: self.eq_band_releases,
            eq_band_makeups: self.eq_band_makeups,
            eq_band_modes: self.eq_band_modes,
            eq_band_filters: self.eq_band_filters,
            eq_show_advanced: self.eq_show_advanced,
            eq_band_show_advanced: self.eq_band_show_advanced,
            eq_band_expanded: self.eq_band_expanded,
            noise_show_advanced: self.noise_show_advanced,
            mute_playback: self.mute_playback,
            auto_play_enabled: self.auto_play_enabled,
            auto_play_file: self.auto_play_file.clone(),
            highpass_enabled: self.highpass_enabled,
            highpass_cutoff: self.highpass_cutoff,
            saturation_enabled: self.saturation_enabled,
            saturation_drive: self.saturation_drive,
            saturation_makeup: self.saturation_makeup,
            saturation_mix: self.saturation_mix,
            show_saturation_advanced: self.show_saturation_advanced,
            transient_enabled: self.transient_enabled,
            transient_gain: self.transient_gain,
            transient_sustain: self.transient_sustain,
            transient_mix: self.transient_mix,
            show_transient_advanced: self.show_transient_advanced,
            agc_enabled: self.agc_enabled,
            timbre_enabled: self.timbre_enabled,
            agc_target_db: self.agc_target_db,
            agc_max_gain_db: self.agc_max_gain_db,
            agc_max_atten_db: self.agc_max_atten_db,
            agc_window_sec: self.agc_window_sec,
            agc_attack_ms: self.agc_attack_ms,
            agc_release_ms: self.agc_release_ms,
            show_agc_advanced: self.show_agc_advanced,
            aec_enabled: self.aec_enabled,
            aec_delay_ms: self.aec_delay_ms,
            sys_auto_volume: self.sys_auto_volume,
            env_auto_enabled: self.env_auto_enabled,
            vad_enabled: self.vad_enabled,
        }
    }

    fn apply_user_config(&mut self, cfg: UserConfig) {
        self.scene_preset = cfg.scene_preset;
        self.atten_lim = cfg.atten_lim;
        self.post_filter_beta = cfg.post_filter_beta;
        self.min_threshdb = cfg.min_threshdb;
        self.max_erbthreshdb = cfg.max_erbthreshdb;
        self.max_dfthreshdb = cfg.max_dfthreshdb;
        self.df_mix = 1.0;
        self.headroom_gain = cfg.headroom_gain;
        self.show_spectrum = cfg.show_spectrum;
        self.rt60_enabled = cfg.rt60_enabled;
        self.final_limiter_enabled = cfg.final_limiter_enabled;
        self.aec_aggressive = cfg.aec_aggressive;
        self.post_trim_gain = cfg.post_trim_gain;
        self.eq_enabled = cfg.eq_enabled;
        self.eq_preset = cfg.eq_preset;
        self.eq_dry_wet = 1.0;
        self.eq_band_gains = cfg.eq_band_gains;
        self.eq_band_frequencies = cfg.eq_band_frequencies;
        self.eq_band_qs = cfg.eq_band_qs;
        self.eq_band_detector_qs = cfg.eq_band_detector_qs;
        self.eq_band_thresholds = cfg.eq_band_thresholds;
        self.eq_band_ratios = cfg.eq_band_ratios;
        self.eq_band_max_gains = cfg.eq_band_max_gains;
        self.eq_band_attacks = cfg.eq_band_attacks;
        self.eq_band_releases = cfg.eq_band_releases;
        self.eq_band_makeups = cfg.eq_band_makeups;
        self.eq_band_modes = cfg.eq_band_modes;
        self.eq_band_filters = cfg.eq_band_filters;
        self.eq_show_advanced = cfg.eq_show_advanced;
        self.eq_band_show_advanced = cfg.eq_band_show_advanced;
        self.eq_band_expanded = cfg.eq_band_expanded;
        self.input_buffers.clear();
        if !self.eq_band_expanded.is_empty() {
            self.eq_band_expanded[0] = true;
        }
        self.noise_show_advanced = cfg.noise_show_advanced;
        self.mute_playback = cfg.mute_playback;
        self.auto_play_enabled = cfg.auto_play_enabled;
        self.auto_play_file = cfg.auto_play_file.or_else(default_auto_play_file);
        self.auto_play_pid = None;
        self.highpass_enabled = cfg.highpass_enabled;
        self.highpass_cutoff = cfg.highpass_cutoff;
        self.saturation_enabled = cfg.saturation_enabled;
        self.saturation_drive = cfg.saturation_drive;
        self.saturation_makeup = cfg.saturation_makeup;
        self.saturation_mix = cfg.saturation_mix;
        self.show_saturation_advanced = cfg.show_saturation_advanced;
        self.transient_enabled = cfg.transient_enabled;
        self.transient_gain = cfg.transient_gain;
        self.transient_sustain = cfg.transient_sustain;
        self.transient_mix = cfg.transient_mix;
        self.show_transient_advanced = cfg.show_transient_advanced;
        self.agc_enabled = cfg.agc_enabled;
        self.timbre_enabled = cfg.timbre_enabled;
        self.agc_target_db = cfg.agc_target_db;
        self.agc_max_gain_db = cfg.agc_max_gain_db;
        self.agc_max_atten_db = cfg.agc_max_atten_db;
        self.agc_window_sec = cfg.agc_window_sec;
        self.agc_attack_ms = cfg.agc_attack_ms;
        self.agc_release_ms = cfg.agc_release_ms;
        self.show_agc_advanced = cfg.show_agc_advanced;
        self.aec_enabled = cfg.aec_enabled;
        self.aec_delay_ms = cfg.aec_delay_ms;
        self.sys_auto_volume = cfg.sys_auto_volume;
        self.env_auto_enabled = cfg.env_auto_enabled;
        self.vad_enabled = cfg.vad_enabled;
        self.sync_runtime_controls();
        self.ensure_sys_volume_monitor();
    }

    fn save_config_to_path(&self, path: &Path) -> Result<PathBuf, String> {
        let cfg = self.to_user_config();
        write_config_file(&cfg, path)
    }
}

fn spec_view(title: &str, im: image::Handle, width: u16, height: u16) -> Element<'_, Message> {
    column![
        text(title).size(24).width(Length::Fill),
        spec_raw(im, width, height)
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}
fn spec_raw<'a>(im: image::Handle, width: u16, height: u16) -> Container<'a, Message> {
    container(Image::new(im).width(width).height(height).content_fit(ContentFit::Fill))
        .max_width(width)
        .max_height(height)
        .width(Length::Fill)
        .center_x()
        .center_y()
}

impl SpecView {
    fn band_label(&self, idx: usize) -> String {
        let base = EQ_BAND_LABELS[idx];
        let freq = self.eq_band_frequencies[idx].abs();
        let q = self.eq_band_qs[idx].max(1e-3);
        let bw = freq / q;
        let mut low = (freq - bw / 2.0).max(0.0);
        let mut high = freq + bw / 2.0;
        if idx == 0 {
            low = 0.0;
        }
        if idx == MAX_EQ_BANDS - 1 {
            high = high.max(freq * 1.5);
        }
        format!(
            "{} {:.0}-{:.0} Hz",
            base,
            low.clamp(0.0, 20_000.0),
            high.clamp(0.0, 20_000.0)
        )
    }

    fn create_eq_panel(&self) -> Element<'_, Message> {
        let presets = EqPresetKind::all().to_vec();
        let scene_presets = ScenePreset::all().to_vec();
        let toggle_label = if self.eq_enabled {
            "动态 EQ 已开启"
        } else {
            "动态 EQ 已关闭"
        };
        let toggle = apply_tooltip(
            toggler(
                Some(toggle_label.to_string()),
                self.eq_enabled,
                Message::EqEnabledChanged,
            ),
            tooltips::EQ_ENABLED,
        );
        let preset_picker = apply_tooltip(
            pick_list(presets, Some(self.eq_preset), Message::EqPresetSelected)
                .placeholder("选择预设")
                .width(Length::Fill),
            self.eq_preset.tooltip_text(),
        );
        let mut general = column![
            row![toggle].align_items(Alignment::Center),
            row![text("预设:").size(14).width(60), preset_picker,]
                .spacing(8)
                .align_items(Alignment::Center),
            text(self.eq_preset.description()).size(13).width(Length::Fill),
            row![
                text("混合:").size(14).width(60),
                text("100% (固定全湿，避免相位梳状)")
                    .size(14)
                    .width(Length::Fill),
            ]
            .spacing(8)
            .align_items(Alignment::Center),
            row![
                text("状态:").size(14).width(60),
                text(self.eq_status_label()).size(14).width(Length::Fill),
                text(format!("CPU {:>4.1}%", self.eq_status.cpu_load.min(100.0)))
                    .size(14)
                    .width(110)
                    .horizontal_alignment(alignment::Horizontal::Right),
            ]
            .spacing(8)
            .align_items(Alignment::Center),
        ]
        .spacing(12);

        general = general
            .push(self.slider_view(
                "atten_lim",
                "噪声抑制 [dB]",
                self.atten_lim,
                0.,
                100.,
                SliderTarget::AttenLim,
                Message::AttenLimChanged,
                420,
                0,
                3.,
                Some(tooltips::NOISE_SUPPRESSION),
            ))
            .push(apply_tooltip(
                row![
                    text("降噪混合 (%)").size(14).width(120),
                    text("100% (固定全湿，避免梳状)").size(14).width(Length::Fill)
                ]
                .spacing(8)
                .align_items(Alignment::Center),
                tooltips::DF_MIX,
            ))
            .push(self.slider_view(
                "post_filter_beta",
                "后滤波 Beta",
                self.post_filter_beta,
                0.,
                1.,
                SliderTarget::PostFilterBeta,
                Message::PostFilterChanged,
                420,
                3,
                0.001,
                Some(tooltips::POST_FILTER),
            ));

        let noise_toggle_text = if self.noise_show_advanced {
            "▼ 隐藏高级降噪参数"
        } else {
            "► 显示高级降噪参数"
        };
        let noise_toggle = apply_tooltip(
            widget::button(noise_toggle_text)
                .width(Length::Fill)
                .on_press(Message::NoiseAdvancedToggled),
            tooltips::NOISE_ADVANCED_TOGGLE,
        );
        general = general.push(noise_toggle);

        if self.noise_show_advanced {
            let advanced_thresholds = column![
                self.slider_view(
                    "min_thresh",
                    "阈值下限 [dB]",
                    self.min_threshdb,
                    -60.,
                    35.,
                    SliderTarget::MinThreshDb,
                    Message::MinThreshDbChanged,
                    420,
                    0,
                    1.,
                    Some(tooltips::MIN_THRESHOLD),
                ),
                self.slider_view(
                    "max_erb",
                    "ERB 阈值上限 [dB]",
                    self.max_erbthreshdb,
                    -15.,
                    35.,
                    SliderTarget::MaxErbThreshDb,
                    Message::MaxErbThreshDbChanged,
                    420,
                    0,
                    1.,
                    Some(tooltips::MAX_ERB_THRESHOLD),
                ),
                self.slider_view(
                    "max_df",
                    "DF 阈值上限 [dB]",
                    self.max_dfthreshdb,
                    -15.,
                    35.,
                    SliderTarget::MaxDfThreshDb,
                    Message::MaxDfThreshDbChanged,
                    420,
                    0,
                    1.,
                    Some(tooltips::MAX_DF_THRESHOLD),
                ),
                text("这些参数影响噪声判断，调节需谨慎。推荐范围：-20~40 dB。")
                    .size(13)
                    .width(Length::Fill),
            ]
            .spacing(12);
            general = general.push(
                container(advanced_thresholds).padding(12).style(iced::theme::Container::Box),
            );
        }

        general = general.push(self.create_audio_enhancement_panel());

        let mut bands = column![];
        for idx in 0..MAX_EQ_BANDS {
            bands = bands.push(self.create_band_panel(idx));
        }
        let band_scroll = scrollable(bands.spacing(12)).height(Length::Fill);

        let reset_button = apply_tooltip(
            button("重置所有频段").on_press(Message::EqResetBands).width(Length::Fill),
            tooltips::EQ_RESET_BANDS,
        );

        let scene_picker = row![
            text("场景:").size(14).width(60),
            pick_list(
                scene_presets,
                Some(self.scene_preset),
                Message::ScenePresetChanged
            )
            .placeholder("选择场景")
            .width(Length::Fill),
        ]
        .spacing(8)
        .align_items(Alignment::Center);

        container(column![scene_picker, general, band_scroll, reset_button,].spacing(16))
            .padding(16)
            .style(iced::theme::Container::Box)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn create_audio_enhancement_panel(&self) -> Element<'_, Message> {
        let highpass_row = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.highpass_enabled,
                    Message::HighpassToggled
                ),
                text(format!("高通滤波器 ({:.0}Hz)", self.highpass_cutoff)).size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            tooltips::HIGHPASS_FILTER,
        );
        let aec_row = row![
            toggler(String::new(), self.aec_enabled, Message::AecToggled),
            text("AEC3 回声消除").size(14),
            toggler(String::new(), self.aec_aggressive, Message::AecAggressiveToggled),
            text("强力模式").size(13),
        ]
        .spacing(10)
        .align_items(Alignment::Center);
        let aec_delay_slider = if self.aec_enabled {
            self.slider_view(
                "aec_delay",
                "回声延迟补偿 [ms]",
                self.aec_delay_ms,
                0.0,
                200.0,
                SliderTarget::AecDelay,
                Message::AecDelayChanged,
                380,
                0,
                5.0,
                Some("播放链路延迟补偿，外放时可尝试 50~120 ms"),
            )
        } else {
            widget::Column::new().into()
        };
        let highpass_controls: Element<'_, Message> = if self.highpass_enabled {
            self.slider_view(
                "highpass_cutoff",
                "截止频率 [Hz]",
                self.highpass_cutoff,
                0.0,
                300.0,
                SliderTarget::HighpassCutoff,
                Message::HighpassCutoffChanged,
                380,
                0,
                1.0,
                Some(tooltips::HIGHPASS_FILTER),
            )
        } else {
            widget::Column::new().into()
        };

        let saturation_toggle = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.saturation_enabled,
                    Message::SaturationToggled
                ),
                text("饱和/谐波增强").size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            tooltips::SATURATION,
        );

        let saturation_controls: Element<'_, Message> = if self.saturation_enabled {
            let toggle_button = button(if self.show_saturation_advanced {
                "▼ 饱和参数"
            } else {
                "► 饱和参数"
            })
            .on_press(Message::SaturationToggleAdvanced)
            .width(Length::Fill);

            let mut advanced = widget::Column::new();
            if self.show_saturation_advanced {
                advanced = advanced
                    .push(self.slider_view(
                        "saturation_drive",
                        "驱动 (Drive)",
                        self.saturation_drive,
                        0.8,
                        1.8,
                        SliderTarget::SaturationDrive,
                        Message::SaturationDriveChanged,
                        380,
                        2,
                        0.02,
                        Some(tooltips::SATURATION_DRIVE),
                    ))
                    .push(self.slider_view(
                        "saturation_makeup",
                        "补偿增益 [dB]",
                        self.saturation_makeup,
                        -6.0,
                        3.0,
                        SliderTarget::SaturationMakeup,
                        Message::SaturationMakeupChanged,
                        380,
                        1,
                        0.1,
                        Some(tooltips::SATURATION_MAKEUP),
                    ))
                    .push(self.slider_view(
                        "saturation_mix",
                        "混合比例 [%]",
                        self.saturation_mix,
                        0.0,
                        100.0,
                        SliderTarget::SaturationMix,
                        Message::SaturationMixChanged,
                        380,
                        0,
                        1.0,
                        Some(tooltips::SATURATION_MIX),
                    ));
            }
            widget::Column::new().spacing(8).push(toggle_button).push(advanced).into()
        } else {
            widget::Column::new().into()
        };

        let exciter_toggle = row![
            toggler(String::new(), self.exciter_enabled, Message::ExciterToggled),
            text("谐波激励").size(14),
        ]
        .spacing(10)
        .align_items(Alignment::Center);
        let exciter_controls: Element<'_, Message> = if self.exciter_enabled {
            widget::Column::new()
                .spacing(8)
                .push(self.slider_view(
                    "exciter_mix",
                    "激励混合 [%]",
                    self.exciter_mix * 100.0,
                    0.0,
                    50.0,
                    SliderTarget::ExciterMix,
                    |v| Message::ExciterMixChanged(v / 100.0),
                    380,
                    0,
                    1.0,
                    Some("高频补偿，默认 25%"),
                ))
                .into()
        } else {
            widget::Column::new().into()
        };

        let timbre_toggle = row![
            toggler(String::new(), self.timbre_enabled, Message::TimbreToggled),
            text("音色修复 (ONNX)").size(14),
        ]
        .spacing(10)
        .align_items(Alignment::Center);

        let transient_toggle = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.transient_enabled,
                    Message::TransientToggled
                ),
                text("瞬态增强").size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            tooltips::TRANSIENT_SHAPER,
        );

        let transient_controls: Element<'_, Message> = if self.transient_enabled {
            let toggle_button = button(if self.show_transient_advanced {
                "▼ 瞬态参数"
            } else {
                "► 瞬态参数"
            })
            .on_press(Message::TransientToggleAdvanced)
            .width(Length::Fill);

            let mut advanced = widget::Column::new();
            if self.show_transient_advanced {
                advanced = advanced
                    .push(self.slider_view(
                        "transient_gain",
                        "瞬态增益 [dB]",
                        self.transient_gain,
                        0.,
                        12.,
                        SliderTarget::TransientGain,
                        Message::TransientGainChanged,
                        380,
                        1,
                        0.5,
                        Some(tooltips::TRANSIENT_GAIN),
                    ))
                    .push(self.slider_view(
                        "transient_sustain",
                        "释放增益 [dB]",
                        self.transient_sustain,
                        -12.0,
                        6.0,
                        SliderTarget::TransientSustain,
                        Message::TransientSustainChanged,
                        380,
                        1,
                        0.5,
                        Some(tooltips::TRANSIENT_SUSTAIN),
                    ))
                    .push(self.slider_view(
                        "transient_mix",
                        "混合比例 [%]",
                        self.transient_mix,
                        0.,
                        100.,
                        SliderTarget::TransientMix,
                        Message::TransientMixChanged,
                        380,
                        0,
                        5.,
                        Some(tooltips::TRANSIENT_MIX),
                    ));
            }
            widget::Column::new().spacing(8).push(toggle_button).push(advanced).into()
        } else {
            widget::Column::new().into()
        };

        let agc_row = apply_tooltip(
            row![
                toggler(String::new(), self.agc_enabled, Message::AgcToggled),
                text("自动增益控制").size(14),
                text(format!("({:+.1} dB)", self.agc_current_gain))
                    .size(12)
                    .style(Color::from_rgb(0.5, 0.5, 0.5)),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            tooltips::AGC,
        );

        let agc_controls: Element<'_, Message> = if self.agc_enabled {
            let toggle_button = button(if self.show_agc_advanced {
                "▼ AGC 参数"
            } else {
                "► AGC 参数"
            })
            .on_press(Message::AgcToggleAdvanced)
            .width(Length::Fill);
            let mut advanced = widget::Column::new();
            if self.show_agc_advanced {
                advanced = advanced
                    .push(self.create_slider_row(
                        "目标电平（距满刻度）[dB]",
                        Some("agc_target"),
                        self.agc_target_db,
                        -20.0,
                        0.0,
                        1.0,
                        1,
                        Some(SliderTarget::AgcTarget),
                        Message::AgcTargetChanged,
                        |v| format!("{:.0} dBFS", v),
                        "WebRTC AGC 目标：距满刻度的正值（0~31），-10 表示目标 -10 dBFS。",
                    ))
                    .push(self.create_slider_row(
                        "最大增益 [dB]",
                        Some("agc_max_gain"),
                        self.agc_max_gain_db,
                        0.0,
                        18.0,
                        0.5,
                        1,
                        Some(SliderTarget::AgcMaxGain),
                        Message::AgcMaxGainChanged,
                        |v| format!("{:+.1} dB", v),
                        tooltips::AGC_MAX_GAIN,
                    ))
                    // WebRTC AGC 内部自适应，以下参数无效，收起避免误解
                    ;
            }
            widget::Column::new().spacing(8).push(toggle_button).push(advanced).into()
        } else {
            widget::Column::new().into()
        };

        let headroom_row = self.create_slider_row(
            "Headroom (线性倍率)",
            Some("headroom_gain"),
            self.headroom_gain,
            0.6,
            1.05,
            0.01,
            2,
            Some(SliderTarget::HeadroomGain),
            Message::HeadroomChanged,
            |v| format!("{:.0}%", v * 100.0),
            "输出前置头间距，降低可减少啸叫/削顶，升高可提高响度（默认 90%）",
        );

        let sys_auto_volume_toggle = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.sys_auto_volume,
                    Message::SysAutoVolumeToggled
                ),
                text("自动系统音量保护（削波时下调输入增益）").size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            "检测连续削波后自动降低系统输入音量，冷却期内不再调整",
        );
        let env_auto_toggle = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.env_auto_enabled,
                    Message::EnvAutoToggled
                ),
                text("自适应降噪（根据噪声自动调整降噪和高通）").size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            "开启后根据噪声特征自动调整 DF 阈值/高通/混合，关闭则完全按手动参数",
        );
        let vad_toggle = apply_tooltip(
            row![
                toggler(
                    String::new(),
                    self.vad_enabled,
                    Message::VadToggled
                ),
                text("VAD 语音检测（Silero，仅非语音段更新噪声/RT60）").size(14),
            ]
            .spacing(10)
            .align_items(Alignment::Center),
            "关闭可对比效果；关闭后噪声地板与 RT60 更新不再用 VAD 门控（当前使用 Silero VAD）",
        );

        container(
            column![
                text("音频增强").size(16),
                highpass_row,
                highpass_controls,
                aec_row,
                aec_delay_slider,
                saturation_toggle,
                saturation_controls,
                exciter_toggle,
                exciter_controls,
                timbre_toggle,
                transient_toggle,
                transient_controls,
                agc_row,
                agc_controls,
                headroom_row,
                row![
                    toggler(
                        String::new(),
                        self.final_limiter_enabled,
                        |v| Message::FinalLimiterToggled(v)
                    ),
                    text("最终限幅器（保护用，可关闭做纯降噪对比）").size(14)
                ]
                .spacing(10)
                .align_items(Alignment::Center),
                sys_auto_volume_toggle,
                env_auto_toggle,
                vad_toggle
            ]
            .spacing(12),
        )
        .padding(12)
        .style(iced::theme::Container::Box)
        .into()
    }

    fn create_spectrum_panel(&self) -> Element<'_, Message> {
        let toggles = row![
            toggler(String::new(), self.show_spectrum, Message::SpectrumToggled),
            text("显示频谱/RT60").size(14),
            toggler(String::new(), self.rt60_enabled, Message::Rt60Toggled),
            text("RT60 估计").size(14),
        ]
        .spacing(12)
        .align_items(Alignment::Center);

        let spectrums: Element<'_, Message> = if self.show_spectrum {
            column![
                spec_view(
                    "拾音频谱",
                    self.noisy_img.clone(),
                    SPEC_DISPLAY_WIDTH,
                    SPEC_DISPLAY_HEIGHT,
                ),
                spec_view(
                    "降噪后频谱",
                    self.enh_img.clone(),
                    SPEC_DISPLAY_WIDTH,
                    SPEC_DISPLAY_HEIGHT,
                ),
            ]
            .spacing(20)
            .into()
        } else {
            text("频谱/RT60 显示已关闭（可在左侧开关恢复）")
                .size(14)
                .width(Length::Fill)
                .into()
        };

        let info = column![
            row![
                text("当前信噪比:").size(16),
                text(format!("{:>5.1} dB", self.lsnr))
                    .size(16)
                    .width(120)
                    .horizontal_alignment(alignment::Horizontal::Right),
            ]
            .spacing(12),
            row![
                text("EQ状态:").size(16),
                text(self.eq_status_label()).size(16).width(Length::Fill),
                text(format!("CPU {:>4.1}%", self.eq_status.cpu_load.min(100.0)))
                    .size(16)
                    .width(120)
                    .horizontal_alignment(alignment::Horizontal::Right),
            ]
            .spacing(12),
        ]
        .spacing(10);

        column![toggles, spectrums, info]
            .spacing(16)
            .width(Length::Fill)
            .into()
    }

    fn create_band_panel(&self, idx: usize) -> Element<'_, Message> {
        let band_label = self.band_label(idx);
        let is_expanded = self.eq_band_expanded[idx];
        let expand_icon = if is_expanded { "▼" } else { "►" };
        let header = row![
            widget::button(text(format!("{expand_icon} {band_label}")))
                .padding(6)
                .on_press(Message::EqBandToggleExpand(idx))
                .width(Length::Fill)
                .style(iced::theme::Button::Secondary),
            text(format!(
                "动态 {:+.1} dB",
                self.eq_status.gain_reduction_db[idx]
            ))
            .size(13)
            .width(110)
            .horizontal_alignment(alignment::Horizontal::Right),
        ]
        .spacing(8)
        .align_items(Alignment::Center);

        if !is_expanded {
            let gain_slider = self.create_slider_row(
                "增益 (dB)",
                None,
                self.eq_band_gains[idx],
                -12.0,
                12.0,
                0.5,
                1,
                None,
                move |v| Message::EqBandGainChanged(idx, v),
                |v| format!("{:+.1} dB", v),
                tooltips::EQ_PARAM_GAIN,
            );
            return container(column![header, gain_slider].spacing(8))
                .padding(10)
                .style(iced::theme::Container::Box)
                .width(Length::Fill)
                .into();
        }

        let core = column![
            self.create_slider_row(
                "增益 (dB)",
                None,
                self.eq_band_gains[idx],
                -12.0,
                12.0,
                0.5,
                1,
                None,
                move |v| Message::EqBandGainChanged(idx, v),
                |v| format!("{:+.1} dB", v),
                tooltips::EQ_PARAM_GAIN,
            ),
            self.create_slider_row(
                "频率 (Hz)",
                None,
                self.eq_band_frequencies[idx],
                20.0,
                20000.0,
                1.0,
                0,
                None,
                move |v| Message::EqBandFrequencyChanged(idx, v),
                |v| format!("{:.0} Hz", v),
                tooltips::EQ_PARAM_FREQUENCY,
            ),
            self.create_slider_row(
                "阈值 (dB)",
                None,
                self.eq_band_thresholds[idx],
                -60.0,
                0.0,
                0.5,
                1,
                None,
                move |v| Message::EqBandThresholdChanged(idx, v),
                |v| format!("{:+.1} dB", v),
                tooltips::EQ_PARAM_THRESHOLD,
            ),
            self.create_slider_row(
                "比率",
                None,
                self.eq_band_ratios[idx],
                1.0,
                10.0,
                0.1,
                1,
                None,
                move |v| Message::EqBandRatioChanged(idx, v),
                |v| ratio_short_text(self.eq_band_modes[idx], v),
                tooltips::EQ_PARAM_RATIO,
            ),
        ]
        .spacing(8);

        let advanced_toggle_text = if self.eq_band_show_advanced[idx] {
            "🔼 隐藏高级参数"
        } else {
            "🔧 显示高级参数"
        };
        let advanced_toggle =
            widget::button(advanced_toggle_text).on_press(Message::EqBandToggleAdvanced(idx));

        let mut advanced = column![advanced_toggle].spacing(8);
        if self.eq_band_show_advanced[idx] {
            advanced = advanced.push(
                column![
                    self.create_slider_row(
                        "Q 值",
                        None,
                        self.eq_band_qs[idx],
                        0.1,
                        5.0,
                        0.01,
                        2,
                        None,
                        move |v| Message::EqBandQChanged(idx, v),
                        |v| format!("{:.2}", v),
                        tooltips::EQ_PARAM_Q,
                    ),
                    self.create_slider_row(
                        "检测器 Q",
                        None,
                        self.eq_band_detector_qs[idx],
                        0.1,
                        5.0,
                        0.01,
                        2,
                        None,
                        move |v| Message::EqBandDetectorQChanged(idx, v),
                        |v| format!("{:.2}", v),
                        tooltips::EQ_PARAM_DETECTOR_Q,
                    ),
                    self.create_slider_row(
                        "最大增益 (dB)",
                        None,
                        self.eq_band_max_gains[idx],
                        0.0,
                        20.0,
                        0.5,
                        1,
                        None,
                        move |v| Message::EqBandMaxGainChanged(idx, v),
                        |v| format!("{:.1} dB", v),
                        tooltips::EQ_PARAM_MAX_GAIN,
                    ),
                    self.create_slider_row(
                        "起音 (ms)",
                        None,
                        self.eq_band_attacks[idx],
                        1.0,
                        100.0,
                        1.0,
                        0,
                        None,
                        move |v| Message::EqBandAttackChanged(idx, v),
                        |v| format!("{:.0} ms", v),
                        tooltips::EQ_PARAM_ATTACK,
                    ),
                    self.create_slider_row(
                        "释放 (ms)",
                        None,
                        self.eq_band_releases[idx],
                        10.0,
                        500.0,
                        5.0,
                        0,
                        None,
                        move |v| Message::EqBandReleaseChanged(idx, v),
                        |v| format!("{:.0} ms", v),
                        tooltips::EQ_PARAM_RELEASE,
                    ),
                    self.create_slider_row(
                        "补偿 (dB)",
                        None,
                        self.eq_band_makeups[idx],
                        -6.0,
                        6.0,
                        0.1,
                        1,
                        None,
                        move |v| Message::EqBandMakeupChanged(idx, v),
                        |v| format!("{:+.1} dB", v),
                        tooltips::EQ_PARAM_MAKEUP,
                    ),
                    apply_tooltip(
                        row![
                            text("模式").size(13).width(80),
                            pick_list(
                                BandMode::all().to_vec(),
                                Some(self.eq_band_modes[idx]),
                                move |mode| Message::EqBandModeChanged(idx, mode),
                            )
                            .width(Length::Fill),
                        ]
                        .spacing(8)
                        .align_items(Alignment::Center),
                        tooltips::EQ_PARAM_MODE,
                    ),
                    apply_tooltip(
                        row![
                            text("滤波器").size(13).width(80),
                            pick_list(
                                FilterKind::all().to_vec(),
                                Some(self.eq_band_filters[idx]),
                                move |filter| Message::EqBandFilterChanged(idx, filter),
                            )
                            .width(Length::Fill),
                        ]
                        .spacing(8)
                        .align_items(Alignment::Center),
                        tooltips::EQ_PARAM_FILTER,
                    ),
                ]
                .spacing(8),
            );
        }

        container(column![header, core, advanced].spacing(10))
            .padding(12)
            .style(iced::theme::Container::Box)
            .width(Length::Fill)
            .into()
    }

    fn buffer_value(&self, key: &str, value: f32, precision: usize) -> String {
        self.input_buffers
            .get(key)
            .cloned()
            .unwrap_or_else(|| format!("{:.precision$}", value))
    }

    fn set_buffer(&mut self, key: &str, value: String) {
        self.input_buffers.insert(key.to_string(), value);
    }

    fn apply_slider_value(&mut self, target: SliderTarget, value: f32) {
        match target {
            SliderTarget::AttenLim => {
                self.atten_lim = value;
                self.send_df_control(DfControl::AttenLim, value);
            }
            SliderTarget::PostFilterBeta => {
                self.post_filter_beta = value;
                self.send_df_control(DfControl::PostFilterBeta, value);
            }
            SliderTarget::MinThreshDb => {
                self.min_threshdb = value;
                self.send_df_control(DfControl::MinThreshDb, value);
            }
            SliderTarget::MaxErbThreshDb => {
                self.max_erbthreshdb = value;
                self.send_df_control(DfControl::MaxErbThreshDb, value);
            }
            SliderTarget::MaxDfThreshDb => {
                self.max_dfthreshdb = value;
                self.send_df_control(DfControl::MaxDfThreshDb, value);
            }
            SliderTarget::HighpassCutoff => {
                self.highpass_cutoff = value;
                self.send_control_message(ControlMessage::HighpassCutoff(value));
            }
            SliderTarget::SaturationDrive => {
                self.saturation_drive = value;
                self.send_control_message(ControlMessage::SaturationDrive(value));
            }
            SliderTarget::SaturationMakeup => {
                self.saturation_makeup = value;
                self.send_control_message(ControlMessage::SaturationMakeup(value));
            }
            SliderTarget::SaturationMix => {
                self.saturation_mix = value;
                self.send_control_message(ControlMessage::SaturationMix(value));
            }
            SliderTarget::ExciterMix => {
                let mix = (value / 100.0).clamp(0.0, 0.5);
                self.exciter_mix = mix;
                self.send_control_message(ControlMessage::ExciterMix(mix));
            }
            SliderTarget::TransientGain => {
                self.transient_gain = value;
                self.send_control_message(ControlMessage::TransientGain(value));
            }
            SliderTarget::TransientSustain => {
                self.transient_sustain = value;
                self.send_control_message(ControlMessage::TransientSustain(value));
            }
            SliderTarget::TransientMix => {
                self.transient_mix = value;
                self.send_control_message(ControlMessage::TransientMix(value));
            }
            SliderTarget::HeadroomGain => {
                self.headroom_gain = value;
                self.send_control_message(ControlMessage::HeadroomGain(value));
            }
            SliderTarget::AecDelay => {
                self.aec_delay_ms = value;
                self.send_control_message(ControlMessage::AecDelayMs(value as i32));
            }
            SliderTarget::AgcTarget => {
                self.agc_target_db = value;
                self.send_control_message(ControlMessage::AgcTargetLevel(value));
            }
            SliderTarget::AgcMaxGain => {
                self.agc_max_gain_db = value;
                self.send_control_message(ControlMessage::AgcMaxGain(value));
            }
            SliderTarget::AgcMaxAtten => {
                self.agc_max_atten_db = value;
                self.send_control_message(ControlMessage::AgcMaxAttenuation(value));
            }
            SliderTarget::AgcWindow => {
                self.agc_window_sec = value;
                self.send_control_message(ControlMessage::AgcWindowSeconds(value));
            }
            SliderTarget::AgcAttack => {
                self.agc_attack_ms = value;
                self.send_control_message(ControlMessage::AgcAttackRelease(
                    value,
                    self.agc_release_ms,
                ));
            }
            SliderTarget::AgcRelease => {
                self.agc_release_ms = value;
                self.send_control_message(ControlMessage::AgcAttackRelease(
                    self.agc_attack_ms,
                    value,
                ));
            }
        }
    }

    fn create_slider_row<F, G>(
        &self,
        label: &str,
        key: Option<&str>,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        precision: usize,
        target: Option<SliderTarget>,
        on_change: F,
        formatter: G,
        tooltip_text: &'static str,
    ) -> Element<'_, Message>
    where
        F: 'static + Copy + Fn(f32) -> Message,
        G: Fn(f32) -> String,
    {
        let key_owned = key.map(|k| k.to_string());
        let (display, on_input): (String, Box<dyn Fn(String) -> Message + 'static>) =
            if let (Some(k), Some(t)) = (key_owned.clone(), target.clone()) {
                let d = self.buffer_value(&k, value, precision);
                let handler = move |s: String| Message::SliderInputChanged {
                    key: k.clone(),
                    raw: s,
                    target: t.clone(),
                    min,
                    max,
                    precision,
                };
                (d, Box::new(handler))
            } else {
                let d = formatter(value);
                let handler = move |s: String| {
                    if let Ok(parsed) = s.parse::<f32>() {
                        let clamped = parsed.clamp(min, max);
                        on_change(clamped)
                    } else {
                        Message::None
                    }
                };
                (d, Box::new(handler))
            };
        let row_element = row![
            text(label).size(13).width(110),
            slider(min..=max, value, on_change).step(step).width(Length::Fill),
            text_input("", &display)
                .on_input(on_input)
                .on_submit(Message::None)
                .padding(6)
                .size(13)
                .width(90),
        ]
        .spacing(8)
        .align_items(Alignment::Center);
        apply_tooltip(row_element, tooltip_text)
    }

    #[allow(clippy::too_many_arguments)]
    fn slider_view<'a>(
        &self,
        key: impl Into<String>,
        title: &str,
        value: f32,
        min: f32,
        max: f32,
        target: SliderTarget,
        message: impl Fn(f32) -> Message + Copy + 'a,
        width: u16,
        precision: usize,
        step: f32,
        tooltip_text: Option<&'static str>,
    ) -> Element<'a, Message> {
        let key: String = key.into();
        let slider_widget = slider(min..=max, value, message).step(step);
        let slider_element = if let Some(text) = tooltip_text {
            apply_tooltip(slider_widget, text)
        } else {
            slider_widget.into()
        };
        let display = self.buffer_value(&key, value, precision);
        let on_input = move |s: String| Message::SliderInputChanged {
            key: key.clone(),
            raw: s,
            target: target.clone(),
            min,
            max,
            precision,
        };
        let input = text_input("", &display)
            .on_input(on_input)
            .on_submit(Message::None)
            .padding(6)
            .size(16)
            .width(90);
        column![
            text(title).size(18).width(Length::Fill),
            row![container(slider_element).width(Length::Fill), input,]
        ]
        .max_width(width)
        .width(Length::Fill)
        .into()
    }

}

fn apply_tooltip<'a>(
    content: impl Into<Element<'a, Message>>,
    text: &'static str,
) -> Element<'a, Message> {
    tooltip::Tooltip::new(content, text, Position::Bottom)
        .gap(6)
        .padding(10)
        .style(iced::theme::Container::Box)
        .into()
}

fn ratio_short_text(mode: BandMode, ratio: f32) -> String {
    let formatted = if (ratio.round() - ratio).abs() < 0.05 {
        format!("{:.0}", ratio)
    } else {
        format!("{:.1}", ratio)
    };
    match mode {
        BandMode::Downward => format!("比率 {}:1", formatted),
        BandMode::Upward => format!("比率 1:{}", formatted),
    }
}

fn button(text: &str) -> widget::Button<'_, Message> {
    widget::button(text).padding(10)
}

fn current_model_path() -> Option<PathBuf> {
    env::var("DF_MODEL").ok().map(PathBuf::from).or_else(capture::get_model_path)
}

fn save_recordings(
    noisy: Vec<f32>,
    denoised: Vec<f32>,
    timbre: Vec<f32>,
    processed: Vec<f32>,
    sample_rate: u32,
    timbre_enabled: bool,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let dir = PathBuf::from(OUTPUT_DIR);
    let timestamp = current_timestamp();
    let folder = dir.join(timestamp);
    fs::create_dir_all(&folder).map_err(|err| err.to_string())?;
    let raw_path = folder.join("raw.wav");
    let denoised_path = folder.join("nc.wav");
    let timbre_path = folder.join("timbre.wav");
    let processed_path = folder.join("final.wav");
    let mut created = Vec::new();
    let cleanup = |paths: &[PathBuf], folder: &Path| {
        for path in paths {
            if let Err(err) = fs::remove_file(path) {
                log::warn!("无法删除残留文件 {}: {}", path.display(), err);
            }
        }
        if let Err(err) = fs::remove_dir(folder) {
            log::warn!("无法删除残留目录 {}: {}", folder.display(), err);
        }
    };
    created.push(raw_path.clone());
    if let Err(err) = write_wav(&raw_path, &noisy, sample_rate) {
        cleanup(&created, &folder);
        return Err(err);
    }
    created.push(denoised_path.clone());
    if let Err(err) = write_wav(&denoised_path, &denoised, sample_rate) {
        cleanup(&created, &folder);
        return Err(err);
    }
    // 优先使用实时链路记录的音色修复数据；若未记录则回退到离线处理降噪后的音频
    let mut timbre_out = if timbre_enabled { timbre } else { Vec::new() };
    if timbre_enabled && timbre_out.is_empty() {
        timbre_out = denoised.clone();
        if let Some(mut tr) = crate::audio::timbre_restore::load_default_timbre(TIMBRE_MODEL) {
            let frame = 480.min(timbre_out.len().max(1)); // 约 20ms 帧长
            for chunk in timbre_out.chunks_mut(frame) {
                if let Err(e) = tr.process_frame(chunk) {
                    log::warn!("离线音色修复失败: {e}");
                    break;
                }
            }
        } else {
            log::warn!("音色修复加载模型失败，跳过离线处理");
            timbre_out.clear();
        }
    }
    created.push(timbre_path.clone());
    if let Err(err) = write_wav(&timbre_path, &timbre_out, sample_rate) {
        cleanup(&created, &folder);
        return Err(err);
    }
    created.push(processed_path.clone());
    if let Err(err) = write_wav(&processed_path, &processed, sample_rate) {
        cleanup(&created, &folder);
        return Err(err);
    }
    Ok((raw_path, denoised_path, timbre_path, processed_path))
}

#[cfg(target_os = "macos")]
async fn play_test_audio(path: PathBuf) -> Result<(), String> {
    log::info!("开始播放测试音频: {}", path.display());
    let status = StdCommand::new("afplay")
        .arg(&path)
        .status()
        .map_err(|e| format!("启动播放失败: {}", e))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("播放进程退出状态 {}", status))
    }
}

#[cfg(not(target_os = "macos"))]
async fn play_test_audio(_path: PathBuf) -> Result<(), String> {
    Err("自动播放仅在 macOS 示例中启用".to_string())
}

#[cfg(target_os = "macos")]
fn kill_pid(pid: u32) {
    let _ = StdCommand::new("kill").arg("-TERM").arg(pid.to_string()).status();
}

#[cfg(not(target_os = "macos"))]
fn kill_pid(_pid: u32) {}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let data: Vec<f32> = samples.to_vec();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|err| err.to_string())?;
    let mut clip_count = 0usize;
    let mut max_amp = 0.0f32;
    for &sample in data.iter() {
        let limited = if sample > 1.0 || sample < -1.0 {
            clip_count += 1;
            max_amp = max_amp.max(sample.abs());
            sample.clamp(-1.0, 1.0)
        } else {
            sample
        };
        writer.write_sample(limited).map_err(|err| err.to_string())?;
    }
    if clip_count > 0 && !samples.is_empty() {
        log::warn!(
            "检测到 {} 个削波样本 ({:.2}%)，最大幅度 {:.2}",
            clip_count,
            100.0 * clip_count as f32 / samples.len() as f32,
            max_amp
        );
        log::warn!("建议降低 EQ 频段增益或减小混合强度");
    }
    writer.finalize().map_err(|err| err.to_string())
}

fn current_timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
        .to_string()
}
