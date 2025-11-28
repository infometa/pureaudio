use std::env;
use std::fmt::Display;
use std::io::{self, stdout, Write};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use crate::audio::agc::AutoGainControl;
use crate::audio::eq::{DynamicEq, EqControl, EqPresetKind, MAX_EQ_BANDS};
use crate::audio::exciter::HarmonicExciter;
use crate::audio::highpass::HighpassFilter;
use crate::audio::saturation::Saturation;
use crate::audio::stft_eq::StftStaticEq;
use crate::audio::transient_shaper::TransientShaper;
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, SupportedStreamConfigRange};
use crossbeam_channel::{unbounded, Receiver, Sender};
use df::{tract::*, Complex32};
use ndarray::prelude::*;
use once_cell::sync::Lazy;
use ringbuf::{producer::PostponedProducer, Consumer, HeapRb, SharedRb};
use rubato::{FftFixedIn, FftFixedOut, Resampler};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvClass {
    Quiet,
    Office,
    Noisy,
}

#[derive(Debug, Clone, Copy, Default)]
struct NoiseFeatures {
    energy_db: f32,
    spectral_flatness: f32,
    spectral_centroid: f32,
}

pub type RbProd = PostponedProducer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type RbCons = Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type SendLsnr = Sender<f32>;
pub type RecvLsnr = Receiver<f32>;
pub type SendSpec = Sender<Box<[f32]>>;
pub type RecvSpec = Receiver<Box<[f32]>>;
pub type SendControl = Sender<ControlMessage>;
pub type RecvControl = Receiver<ControlMessage>;
pub type SendEqStatus = Sender<EqStatus>;
pub type RecvEqStatus = Receiver<EqStatus>;
pub type SendEnvStatus = Sender<EnvStatus>;
pub type RecvEnvStatus = Receiver<EnvStatus>;

#[derive(Debug, Clone)]
pub struct EqStatus {
    pub gain_reduction_db: [f32; MAX_EQ_BANDS],
    pub cpu_load: f32,
    pub enabled: bool,
    pub dry_wet: f32,
    pub agc_gain_db: f32,
}

impl Default for EqStatus {
    fn default() -> Self {
        Self {
            gain_reduction_db: [0.0; MAX_EQ_BANDS],
            cpu_load: 0.0,
            enabled: false,
            dry_wet: 0.0,
            agc_gain_db: 0.0,
        }
    }
}

pub(crate) static INIT_LOGGER: Once = Once::new();
pub(crate) static MODEL_PATH: Lazy<Mutex<Option<PathBuf>>> = Lazy::new(|| Mutex::new(None));
static MODEL_CACHE: Lazy<Mutex<Option<ModelMetadata>>> = Lazy::new(|| Mutex::new(None));
static SYS_VOL_MON_ACTIVE: AtomicBool = AtomicBool::new(false);

const DEFAULT_MAX_RECORDING_DURATION_SECS: usize = 900;
const MAX_RECORDING_DURATION_CAP_SECS: usize = 3600;
const MIN_RECORDING_DURATION_SECS: usize = 60;
const DEFAULT_RECORDING_RESERVE_SECS: usize = 60;

static RECORDING_LIMIT_SECS: Lazy<usize> = Lazy::new(|| {
    env::var("DF_MAX_RECORD_SECS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(MIN_RECORDING_DURATION_SECS, MAX_RECORDING_DURATION_CAP_SECS))
        .unwrap_or(DEFAULT_MAX_RECORDING_DURATION_SECS)
});

static RECORDING_RESERVE_SECS: Lazy<usize> = Lazy::new(|| {
    env::var("DF_RECORD_RESERVE_SECS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_RECORDING_RESERVE_SECS)
        .min(*RECORDING_LIMIT_SECS)
});

#[derive(Clone)]
struct StreamSelection {
    config: StreamConfig,
    format: cpal::SampleFormat,
}

#[derive(Debug, Default)]
pub struct RecordingState {
    sr: usize,
    max_samples: usize,
    noisy: Mutex<Vec<f32>>,
    denoised: Mutex<Vec<f32>>,
    processed: Mutex<Vec<f32>>,
}

impl RecordingState {
    pub fn new(sr: usize) -> Self {
        let max_samples = sr.saturating_mul(*RECORDING_LIMIT_SECS);
        let reserve = sr.saturating_mul(*RECORDING_RESERVE_SECS);
        Self {
            sr,
            max_samples,
            noisy: Mutex::new(Vec::with_capacity(reserve)),
            denoised: Mutex::new(Vec::with_capacity(reserve)),
            processed: Mutex::new(Vec::with_capacity(reserve)),
        }
    }

    pub fn sample_rate(&self) -> usize {
        self.sr
    }

    pub fn append_noisy(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.noisy.lock() {
            self.append_with_limit(&mut buf, samples, "noisy");
        }
    }

    pub fn append_denoised(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.denoised.lock() {
            self.append_with_limit(&mut buf, samples, "denoised");
        }
    }

    pub fn append_processed(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.processed.lock() {
            self.append_with_limit(&mut buf, samples, "processed");
        }
    }

    fn append_with_limit(&self, buf: &mut Vec<f32>, samples: &[f32], tag: &str) {
        if buf.len() >= self.max_samples {
            // 长时间运行时录音缓冲达到上限，静默丢弃，避免日志刷屏
            return;
        }
        let available = self.max_samples - buf.len();
        let to_copy = available.min(samples.len());
        buf.extend_from_slice(&samples[..to_copy]);
        // 达到容量后继续静默丢弃，避免日志刷屏
    }

    pub fn take_samples(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let noisy = self.noisy.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        let denoised =
            self.denoised.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        let processed =
            self.processed.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        (noisy, denoised, processed)
    }
}

pub type RecordingHandle = Arc<RecordingState>;

#[derive(Clone)]
struct ModelMetadata {
    params: DfParams,
    sr: usize,
    frame_size: usize,
    freq_size: usize,
}

pub fn set_model_path(path: Option<PathBuf>) {
    if let Ok(mut slot) = MODEL_PATH.lock() {
        *slot = path;
    }
    if let Ok(mut cache) = MODEL_CACHE.lock() {
        cache.take();
    }
}

pub fn get_model_path() -> Option<PathBuf> {
    MODEL_PATH.lock().ok().and_then(|slot| slot.clone())
}

fn resolve_model_metadata(model_path: Option<PathBuf>) -> Result<ModelMetadata> {
    if model_path.is_none() {
        if let Ok(cache) = MODEL_CACHE.lock() {
            if let Some(meta) = cache.as_ref() {
                return Ok(meta.clone());
            }
        }
    }
    let params = if let Some(path) = model_path {
        DfParams::new(path)?
    } else if let Ok(cache) = MODEL_CACHE.lock() {
        if let Some(meta) = cache.as_ref() {
            return Ok(meta.clone());
        }
        DfParams::default()
    } else {
        DfParams::default()
    };
    let (sr, frame_size, freq_size) = params.audio_dimensions()?;
    let meta = ModelMetadata {
        params,
        sr,
        frame_size,
        freq_size,
    };
    if let Ok(mut cache) = MODEL_CACHE.lock() {
        *cache = Some(meta.clone());
    }
    Ok(meta)
}

pub struct AudioSink {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
    sample_format: cpal::SampleFormat,
}
pub struct AudioSource {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
    sample_format: cpal::SampleFormat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DfControl {
    AttenLim,
    PostFilterBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvStatus {
    Normal,
    Soft,
}

#[derive(Debug, Clone)]
pub enum ControlMessage {
    DeepFilter(DfControl, f32),
    Eq(EqControl),
    MutePlayback(bool),
    BypassEnabled(bool),
    HighpassEnabled(bool),
    DfMix(f32),
    HeadroomGain(f32),
    PostTrimGain(f32),
    TransientEnabled(bool),
    TransientGain(f32),
    TransientSustain(f32),
    TransientMix(f32),
    HighpassCutoff(f32),
    SaturationEnabled(bool),
    SaturationDrive(f32),
    SaturationMakeup(f32),
    SaturationMix(f32),
    AgcEnabled(bool),
    AgcTargetLevel(f32),
    AgcMaxGain(f32),
    AgcMaxAttenuation(f32),
    AgcWindowSeconds(f32),
    AgcAttackRelease(f32, f32),
    SysAutoVolumeEnabled(bool),
    EnvAutoEnabled(bool),
    ExciterEnabled(bool),
    ExciterMix(f32),
    StftEqEnabled(bool),
    StftEqHfGain(f32),
    StftEqAirGain(f32),
    StftEqTilt(f32),
    StftEqBand(usize, f32),
}

pub fn model_dimensions(model_path: Option<PathBuf>, _channels: usize) -> (usize, usize, usize) {
    let path = model_path.or_else(get_model_path);
    let meta = resolve_model_metadata(path).expect("Failed to load DeepFilterNet metadata");
    (meta.sr, meta.frame_size, meta.freq_size)
}

#[derive(Clone, Copy)]
enum StreamDirection {
    Input,
    Output,
}
impl Display for StreamDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamDirection::Input => write!(f, "input"),
            StreamDirection::Output => write!(f, "output"),
        }
    }
}

fn get_all_configs(
    device: &Device,
    direction: StreamDirection,
) -> Result<Vec<SupportedStreamConfigRange>> {
    let configs = match direction {
        StreamDirection::Input => device
            .supported_input_configs()
            .context("failed to query input configs")?
            .collect::<Vec<SupportedStreamConfigRange>>(),
        StreamDirection::Output => device
            .supported_output_configs()
            .context("failed to query output configs")?
            .collect::<Vec<SupportedStreamConfigRange>>(),
    };
    Ok(configs)
}

fn get_stream_config(
    device: &Device,
    sample_rate: u32,
    direction: StreamDirection,
    frame_size: usize,
) -> Result<StreamSelection> {
    let mut configs = Vec::new();
    let all_configs = get_all_configs(device, direction)?;
    for c in all_configs.iter() {
        if c.channels() == 1 {
            log::debug!("Found audio {} config: {:?}", direction, &c);
            configs.push(*c);
        }
    }
    // Further add multi-channel configs if no mono was found. The signal will be downmixed later.
    for c in all_configs.iter() {
        if c.channels() >= 2 {
            log::debug!("Found audio source config: {:?}", &c);
            configs.push(*c);
        }
    }
    if configs.is_empty() {
        return Err(anyhow!("No suitable audio {} config found", direction));
    }
    let sr = SampleRate(sample_rate);
    let select_with_format = |format: cpal::SampleFormat| -> Option<StreamSelection> {
        for c in configs.iter() {
            if c.sample_format() != format {
                continue;
            }
            if sr >= c.min_sample_rate() && sr <= c.max_sample_rate() {
                let mut cfg: StreamConfig = (*c).with_sample_rate(sr).into();
                cfg.buffer_size = BufferSize::Fixed(frame_size as u32);
                return Some(StreamSelection {
                    config: cfg,
                    format,
                });
            }
        }
        None
    };
    for format in [
        cpal::SampleFormat::F32,
        cpal::SampleFormat::I16,
        cpal::SampleFormat::U16,
    ] {
        if let Some(cfg) = select_with_format(format) {
            return Ok(cfg);
        }
    }

    for format in [
        cpal::SampleFormat::F32,
        cpal::SampleFormat::I16,
        cpal::SampleFormat::U16,
    ] {
        if let Some(range) = configs.iter().find(|c| c.sample_format() == format) {
            let mut cfg: StreamConfig = (*range).with_max_sample_rate().into();
            cfg.buffer_size =
                BufferSize::Fixed(frame_size as u32 * cfg.sample_rate.0 / sample_rate);
            log::warn!(
                "Using best matching config {:?} with sample format {:?}",
                cfg,
                format
            );
            return Ok(StreamSelection {
                config: cfg,
                format,
            });
        }
    }
    Err(anyhow!(
        "No audio {} config matches requested sample rate {} Hz",
        direction,
        sample_rate
    ))
}

fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

fn u16_to_f32(sample: u16) -> f32 {
    (sample as f32 / u16::MAX as f32) * 2.0 - 1.0
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn f32_to_u16(sample: f32) -> u16 {
    let scaled = sample.clamp(-1.0, 1.0) * 0.5 + 0.5;
    (scaled * u16::MAX as f32).round() as u16
}

fn push_into_ring(rb: &mut RbProd, data: &[f32], ch: u16, needs_downmix: bool) {
    let expected_frames = data.len() / ch as usize;
    let mut written = 0usize;
    let mut dropped = 0usize;
    if needs_downmix {
        let mut iter = data.chunks(ch as usize).map(df::mean);
        while written < expected_frames {
            let pushed = rb.push_iter(&mut iter);
            if pushed == 0 {
                dropped = expected_frames - written;
                break;
            }
            written += pushed;
        }
    } else {
        let mut offset = 0usize;
        while offset < data.len() {
            let pushed = rb.push_slice(&data[offset..]);
            if pushed == 0 {
                dropped = expected_frames.saturating_sub(offset);
                break;
            }
            offset += pushed;
        }
    }
    if dropped > 0 {
        log::warn!("输入环形缓冲区已满，丢弃 {} 帧音频", dropped);
    }
    rb.sync();
}

fn fill_output_buffer(rb: &mut RbCons, data: &mut [f32], ch: u16, needs_upmix: bool) {
    let frames = data.len() / ch as usize;
    let mut n = 0;
    if needs_upmix {
        while n < frames {
            let mut filled = 0;
            let mut data_it = data.chunks_mut(ch as usize).skip(n);
            for (sample, frame) in rb.pop_iter().zip(&mut data_it) {
                frame.fill(sample);
                n += 1;
                filled += 1;
            }
            if filled == 0 {
                for frame in data.chunks_mut(ch as usize).skip(n) {
                    frame.fill(0.0);
                    n += 1;
                }
                break;
            }
        }
    } else {
        while n < frames {
            let popped = rb.pop_slice(&mut data[n..]);
            if popped == 0 {
                data[n..].fill(0.0);
                n = frames;
                break;
            }
            n += popped;
        }
    }
    debug_assert_eq!(n, frames);
}

impl AudioSink {
    fn new(sample_rate: u32, frame_size: usize, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_output_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.output_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }
        let selection =
            get_stream_config(&device, sample_rate, StreamDirection::Output, frame_size)
                .with_context(|| {
                    format!(
                        "No suitable audio output config found for device {}",
                        device.name().unwrap_or_else(|_| "unknown".into())
                    )
                })?;

        Ok(Self {
            stream: None,
            config: selection.config,
            device,
            sample_format: selection.format,
        })
    }
    fn start(&mut self, mut rb: RbCons) -> Result<()> {
        let ch = self.config.channels;
        let needs_upmix = ch > 1;
        let stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_output_stream(
                &self.config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    fill_output_buffer(&mut rb, data, ch, needs_upmix);
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "Returning data to audio sink with len: {}, rms: {}",
                            data.len() / ch as usize,
                            df::rms(data.iter())
                        );
                    }
                },
                move |err| log::error!("Error during audio output {:?}", err),
                None, // None=blocking, Some(Duration)=timeout
            )?,
            cpal::SampleFormat::I16 => {
                let mut scratch = Vec::<f32>::new();
                self.device.build_output_stream(
                    &self.config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        let needed = data.len();
                        if scratch.len() < needed {
                            scratch.resize(needed, 0.0);
                        } else {
                            scratch[..needed].fill(0.0);
                        }
                        fill_output_buffer(&mut rb, &mut scratch[..needed], ch, needs_upmix);
                        for (dst, src) in data.iter_mut().zip(scratch.iter()) {
                            *dst = f32_to_i16(*src);
                        }
                    },
                    move |err| log::error!("Error during audio output {:?}", err),
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let mut scratch = Vec::<f32>::new();
                self.device.build_output_stream(
                    &self.config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        let needed = data.len();
                        if scratch.len() < needed {
                            scratch.resize(needed, 0.0);
                        } else {
                            scratch[..needed].fill(0.0);
                        }
                        fill_output_buffer(&mut rb, &mut scratch[..needed], ch, needs_upmix);
                        for (dst, src) in data.iter_mut().zip(scratch.iter()) {
                            *dst = f32_to_u16(*src);
                        }
                    },
                    move |err| log::error!("Error during audio output {:?}", err),
                    None,
                )?
            }
            other => panic!("Unsupported output sample format {:?}", other),
        };
        stream.play()?;
        log::info!("Starting playback stream on device {}", self.device.name()?);
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

impl AudioSource {
    fn new(sample_rate: u32, frame_size: usize, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_input_device().expect("no input device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.input_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }
        let selection = get_stream_config(&device, sample_rate, StreamDirection::Input, frame_size)
            .with_context(|| {
                format!(
                    "No suitable audio input config found for device {}",
                    device.name().unwrap_or_else(|_| "unknown".into())
                )
            })?;

        Ok(Self {
            stream: None,
            config: selection.config,
            device,
            sample_format: selection.format,
        })
    }
    fn start(&mut self, mut rb: RbProd) -> Result<()> {
        let ch = self.config.channels;
        let needs_downmix = ch > 1;
        let stream = match self.sample_format {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &self.config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if log::log_enabled!(log::Level::Trace) {
                        log::trace!(
                            "Got data from audio source with len: {}, rms: {}",
                            data.len() / ch as usize,
                            df::rms(data.iter())
                        );
                    }
                    push_into_ring(&mut rb, data, ch, needs_downmix);
                },
                move |err| log::error!("Error during audio output {:?}", err),
                None, // None=blocking, Some(Duration)=timeout
            )?,
            cpal::SampleFormat::I16 => {
                let mut scratch = Vec::<f32>::new();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if scratch.len() < data.len() {
                            scratch.resize(data.len(), 0.0);
                        }
                        for (dst, src) in scratch.iter_mut().zip(data.iter()) {
                            *dst = i16_to_f32(*src);
                        }
                        push_into_ring(&mut rb, &scratch[..data.len()], ch, needs_downmix);
                    },
                    move |err| log::error!("Error during audio output {:?}", err),
                    None,
                )?
            }
            cpal::SampleFormat::U16 => {
                let mut scratch = Vec::<f32>::new();
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if scratch.len() < data.len() {
                            scratch.resize(data.len(), 0.0);
                        }
                        for (dst, src) in scratch.iter_mut().zip(data.iter()) {
                            *dst = u16_to_f32(*src);
                        }
                        push_into_ring(&mut rb, &scratch[..data.len()], ch, needs_downmix);
                    },
                    move |err| log::error!("Error during audio output {:?}", err),
                    None,
                )?
            }
            other => panic!("Unsupported input sample format {:?}", other),
        };
        log::info!("Starting capture stream on device {}", self.device.name()?);
        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

pub(crate) struct AtomicControls {
    has_init: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
}
impl AtomicControls {
    pub fn into_inner(self) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (self.has_init, self.should_stop)
    }
}
pub(crate) struct GuiCom {
    pub s_lsnr: Option<SendLsnr>,
    pub s_spec: Option<(SendSpec, SendSpec)>,
    pub r_opt: Option<RecvControl>,
    pub s_eq_status: Option<SendEqStatus>,
    pub s_env_status: Option<SendEnvStatus>,
}
impl GuiCom {
    pub fn into_inner(
        self,
    ) -> (
        Option<SendLsnr>,
        Option<(SendSpec, SendSpec)>,
        Option<RecvControl>,
        Option<SendEqStatus>,
        Option<SendEnvStatus>,
    ) {
        (
            self.s_lsnr,
            self.s_spec,
            self.r_opt,
            self.s_eq_status,
            self.s_env_status,
        )
    }
}

fn get_worker_fn(
    mut rb_in: RbCons,
    mut rb_out: RbProd,
    input_sr: usize,
    output_sr: usize,
    controls: AtomicControls,
    df_com: Option<GuiCom>,
    recorder: Option<RecordingHandle>,
    df_params: DfParams,
    channels: usize,
) -> impl FnMut() {
    let (has_init, should_stop) = controls.into_inner();
    let (s_lsnr, mut s_spec, mut r_opt, s_eq_status, s_env_status) = if let Some(df_com) = df_com {
        df_com.into_inner()
    } else {
        (None, None, None, None, None)
    };
    let recording = recorder.clone();
    move || {
        let mut df = DfTract::new(df_params.clone(), &RuntimeParams::default_with_ch(channels))
            .expect("Failed to initialize DeepFilterNet runtime");
        debug_assert_eq!(df.ch, 1); // Processing for more channels are not implemented yet
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        df.process(inframe.view(), outframe.view_mut())
            .expect("Failed to run DeepFilterNet");
        let mut dynamic_eq = DynamicEq::new(df.sr as f32, EqPresetKind::default());
        let mut highpass = HighpassFilter::new(df.sr as f32);
        let mut stft_eq = StftStaticEq::new(df.sr);
        let mut exciter = HarmonicExciter::new(df.sr as f32, 7500.0, 1.6, 0.25);
        let mut transient_shaper = TransientShaper::new(df.sr as f32);
        let mut saturation = Saturation::new();
        let mut agc = AutoGainControl::new(df.sr as f32);
        let mut bus_limiter = BusLimiter::new(df.sr);
        // 更长淡入，避免启动瞬间脉冲
        let fade_total = (df.sr as f32 * 0.20) as usize; // 200ms fade-in
        let mut fade_progress = 0usize;
        let mut highpass_enabled = true;
        let mut highpass_cutoff = 60.0f32;
        let mut manual_highpass = highpass_cutoff;
        highpass.set_cutoff(highpass_cutoff);
        let mut stft_enabled = true;
        let mut stft_hf_gain = 2.0f32;
        let mut stft_air_gain = 3.0f32;
        let mut stft_tilt_gain = 0.0f32;
        let mut stft_band_offsets = [0.0f32; 8];
        let mut stft_dirty = true;
        let mut transient_enabled = true;
        let mut saturation_enabled = true;
        let mut agc_enabled = true;
        let mut exciter_enabled = true;
        let mut bypass_enabled = false;
        let mut mute_playback = false;
        let mut _df_mix = 1.0f32;
        let mut headroom_gain = 0.9f32;
        let mut post_trim_gain = 1.0f32;
        // 自适应环境/设备状态（默认关闭，通过 UI 控制开启）
        let mut env_class = EnvClass::Noisy;
        let mut smoothed_energy = -80.0f32;
        let mut smoothed_flatness = 0.0f32;
        let mut smoothed_centroid = 0.0f32;
        // 噪声地板 & SNR 跟踪
        let mut noise_floor_db = -60.0f32;
        let mut snr_db = 10.0f32;
        let mut target_atten = 45.0f32;
        let mut target_min_thresh = -60.0f32;
        let mut target_max_thresh = 10.0f32;
        let mut target_df_mix = 1.0f32;
        let mut target_hp = 80.0f32;
        let mut target_eq_mix = 0.7f32;
        let mut target_exciter_mix = 0.15f32;
        let mut manual_atten = df.atten_lim.unwrap_or(target_atten);
        let mut manual_min_thresh = df.min_db_thresh;
        let mut manual_max_thresh = df.max_db_df_thresh;
        // 冲击/键盘保护计数
        let mut impact_hold = 0usize;
        const IMPACT_HOLD_FRAMES: usize = 60; // 约 0.6s（取决于 hop）
                                              // 柔和模式用于外部 ANC/极静环境，降低降噪力度并补高频
        let mut soft_mode = false;
        let mut last_soft_mode = false;
        // 滞后计数，避免频繁切换
        let mut soft_mode_hold = 0usize;
        const SOFT_MODE_HOLD_FRAMES: usize = 160; // 约 2s（取决于 hop）
        let mut _auto_gain_scale = 1.0f32;
        has_init.store(true, Ordering::Relaxed);
        log::info!("Worker init");
        let block_duration = df.hop_size as f32 / df.sr as f32;
        // 连续低峰值检测阈值（约 3 秒）
        let low_peak_required = ((3.0 / block_duration).ceil() as usize).max(30);
        // 启动静音时长（避免“啪”），单位：Hop block 数
        let mut warmup_blocks = ((df.sr as f32 * 0.3) / df.hop_size as f32).ceil() as usize; // ~300ms
        let (mut input_resampler, n_in) = if input_sr != df.sr {
            let r = FftFixedOut::<f32>::new(input_sr, df.sr, df.hop_size, 1, 1)
                .expect("Failed to init input resampler");
            let n_in = r.input_frames_max();
            let buf = r.input_buffer_allocate(true);
            (Some((r, buf)), n_in)
        } else {
            (None, df.hop_size)
        };
        let (mut output_resampler, n_out) = if output_sr != df.sr {
            let r = FftFixedIn::<f32>::new(df.sr, output_sr, df.hop_size, 1, 1)
                .expect("Failed to init output resampler");
            let n_out = r.output_frames_max();
            let buf = r.output_buffer_allocate(true);
            // let buf = vec![0.; n_out];
            (Some((r, buf)), n_out)
        } else {
            (None, df.hop_size)
        };
        let mut output_resampler_cleared = output_resampler.is_none();
        let input_retry_delay = Duration::from_micros(100);
        let input_timeout = Duration::from_millis(20);
        let mut auto_sys_volume = false;
        let mut last_sys_adjust =
            Instant::now().checked_sub(Duration::from_secs(10)).unwrap_or_else(Instant::now);
        let mut last_sys_restore = last_sys_adjust;
        let mut clip_counter = 0usize;
        let mut low_peak_counter = 0usize;
        // Align with UI default: 环境自适应默认关闭，避免启动瞬间修改参数
        let mut env_auto_enabled = false;
        'processing: while !should_stop.load(Ordering::Relaxed) {
            if rb_in.len() < n_in {
                // Sleep for half a hop size
                sleep(Duration::from_secs_f32(
                    df.hop_size as f32 / df.sr as f32 / 2.,
                ));
                continue;
            }
            if let Some((ref mut r, ref mut buf)) = input_resampler.as_mut() {
                let mut filled = 0usize;
                let start_fill = Instant::now();
                while filled < n_in {
                    let pulled = rb_in.pop_slice(&mut buf[0][filled..n_in]);
                    if pulled == 0 {
                        if should_stop.load(Ordering::Relaxed) {
                            log::debug!("停止时输入数据不足，退出处理循环");
                            break 'processing;
                        }
                        if start_fill.elapsed() > input_timeout {
                            log::warn!(
                                "等待输入数据超时（需要 {}，已获取 {}），丢弃该帧",
                                n_in,
                                filled
                            );
                            continue 'processing;
                        }
                        sleep(input_retry_delay);
                        continue;
                    }
                    filled += pulled;
                }
                if filled != n_in {
                    continue 'processing;
                }
                if let Some(slice) = inframe.as_slice_mut() {
                    if let Err(err) = r.process_into_buffer(buf, &mut [slice], None) {
                        log::error!("输入重采样失败: {:?}", err);
                        continue 'processing;
                    }
                } else {
                    log::error!("输入帧内存布局异常，跳过本帧");
                    continue 'processing;
                }
            } else {
                let mut filled = 0usize;
                let start_fill = Instant::now();
                if let Some(buffer) = inframe.as_slice_mut() {
                    while filled < n_in {
                        let pulled = rb_in.pop_slice(&mut buffer[filled..n_in]);
                        if pulled == 0 {
                            if should_stop.load(Ordering::Relaxed) {
                                log::debug!("停止时输入数据不足，退出处理循环");
                                break 'processing;
                            }
                            if start_fill.elapsed() > input_timeout {
                                log::warn!(
                                    "等待输入数据超时（需要 {}，已获取 {}），丢弃该帧",
                                    n_in,
                                    filled
                                );
                                continue 'processing;
                            }
                            sleep(input_retry_delay);
                            continue;
                        }
                        filled += pulled;
                    }
                    if filled != n_in {
                        continue 'processing;
                    }
                } else {
                    log::error!("输入帧内存布局异常，跳过本帧");
                    continue 'processing;
                }
            };
            // 录音原始信号（设备采样率或重采样后），在任何处理前
            if let Some(ref rec) = recording {
                if let Some(buffer) = inframe.as_slice() {
                    rec.append_noisy(buffer);
                }
            }

            if highpass_enabled {
                if let Some(buffer) = inframe.as_slice_mut() {
                    highpass.process(buffer);
                }
            }
            let mut lsnr = 0.0;
            if bypass_enabled {
                if let (Some(out), Some(inp)) = (outframe.as_slice_mut(), inframe.as_slice()) {
                    out.copy_from_slice(inp);
                }
            } else {
                lsnr = df
                    .process(inframe.view(), outframe.view_mut())
                    .expect("Failed to run DeepFilterNet");
            }
            if env_auto_enabled && !bypass_enabled {
                // 环境噪声特征估计与自适应参数：噪声地板 + SNR 连续映射
                let feats = compute_noise_features(df.get_spec_noisy());
                let (rms_db, update_alpha) = if let Some(buf) = inframe.as_slice() {
                    let rms = df::rms(buf.iter());
                    let db = 20.0 * rms.max(1e-9).log10();
                    let alpha = if db < -50.0 {
                        0.35
                    } else if db < -30.0 {
                        0.18
                    } else if db < -20.0 {
                        0.10
                    } else {
                        0.05
                    };
                    (db, alpha)
                } else {
                    (-60.0, 0.2)
                };
                smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, update_alpha);
                smoothed_flatness =
                    smooth_value(smoothed_flatness, feats.spectral_flatness, update_alpha);
                smoothed_centroid =
                    smooth_value(smoothed_centroid, feats.spectral_centroid, update_alpha);

                // 噪声地板跟踪：下降快，上升慢
                if rms_db < noise_floor_db {
                    noise_floor_db = smooth_value(noise_floor_db, rms_db, 0.4);
                } else {
                    noise_floor_db = smooth_value(noise_floor_db, rms_db, 0.02);
                }
                snr_db = (rms_db - noise_floor_db).clamp(-5.0, 30.0);

                // 柔和模式：低能量、低平坦度、低重心
                let soft_candidate =
                    smoothed_energy < -55.0 && smoothed_flatness < 0.2 && smoothed_centroid < 0.35;
                if soft_candidate {
                    soft_mode_hold = soft_mode_hold.saturating_add(1);
                } else {
                    soft_mode_hold = soft_mode_hold.saturating_sub(soft_mode_hold.min(1));
                }
                if soft_mode_hold > SOFT_MODE_HOLD_FRAMES {
                    soft_mode = true;
                } else if soft_mode_hold < SOFT_MODE_HOLD_FRAMES / 4 {
                    soft_mode = false;
                }
                if soft_mode != last_soft_mode {
                    last_soft_mode = soft_mode;
                    if soft_mode {
                        log::info!("环境自适应: 切换到柔和模式");
                        if let Some(ref sender) = s_env_status {
                            let _ = sender.try_send(EnvStatus::Soft);
                        }
                    } else {
                        log::info!("环境自适应: 切换到正常模式");
                        if let Some(ref sender) = s_env_status {
                            let _ = sender.try_send(EnvStatus::Normal);
                        }
                    }
                }

                // 键盘/冲击检测
                let mut impact = false;
                if let Some(buf) = inframe.as_slice() {
                    let rms = df::rms(buf.iter());
                    let peak = buf.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
                    let crest = if rms > 1e-6 { peak / rms } else { 0.0 };
                    if crest > 12.0 && rms > 1e-3 {
                        impact = true;
                    }
                }
                if impact {
                    impact_hold = (impact_hold + 10).min(IMPACT_HOLD_FRAMES);
                } else if impact_hold > 0 {
                    impact_hold = impact_hold.saturating_sub(1);
                }

                // SNR 连续映射（安静保色，嘈杂保清晰）
                let mapped = map_snr_to_params(snr_db);
                target_atten = mapped.atten.min(manual_atten);
                target_min_thresh = mapped.min_thresh.max(manual_min_thresh);
                target_max_thresh = mapped.max_thresh.min(manual_max_thresh);
                target_df_mix = 1.0;
                target_hp = mapped.hp_cut.max(manual_highpass);
                target_eq_mix = mapped.eq_mix;
                target_exciter_mix = mapped.exciter_mix;

                // 档位仅用于日志
                let target_env =
                    classify_env(smoothed_energy, smoothed_flatness, smoothed_centroid);
                if target_env != env_class {
                    log::info!(
                        "环境自适应: {:?} -> {:?} (energy {:.1} dB, flatness {:.3}, centroid {:.3}, SNR {:.1} dB, floor {:.1} dB)",
                        env_class, target_env, smoothed_energy, smoothed_flatness, smoothed_centroid, snr_db, noise_floor_db
                    );
                    env_class = target_env;
                }

                if soft_mode {
                    target_atten = 24.0;
                    target_min_thresh = -58.0;
                    target_max_thresh = 12.0;
                    target_df_mix = 1.0;
                    target_hp = 60.0;
                    target_eq_mix = 0.6;
                    target_exciter_mix = 0.2;
                }

                if impact_hold > 0 {
                    target_atten = (target_atten + 8.0).min(60.0);
                    target_df_mix = 1.0;
                    target_hp = target_hp.max(110.0);
                    target_eq_mix = (target_eq_mix - 0.1).max(0.5);
                    target_exciter_mix = 0.0;
                }

                _df_mix = 1.0;
                let current_atten = df.atten_lim.unwrap_or(target_atten);
                let new_atten = smooth_value(current_atten, target_atten, 0.25);
                df.set_atten_lim(new_atten);
                df.min_db_thresh = smooth_value(df.min_db_thresh, target_min_thresh, 0.25);
                df.max_db_df_thresh = smooth_value(df.max_db_df_thresh, target_max_thresh, 0.25);
                df.max_db_erb_thresh = smooth_value(df.max_db_erb_thresh, target_max_thresh, 0.25);
                highpass_cutoff = smooth_value(highpass_cutoff, target_hp, 0.25);
                highpass.set_cutoff(highpass_cutoff);
                // 动态 EQ 全湿，避免干/湿并行带来的相位梳状
                dynamic_eq.set_dry_wet(1.0);
                exciter
                    .set_mix(smooth_value(exciter.mix(), target_exciter_mix, 0.25).clamp(0.0, 0.3));
            }
            // 录音降噪输出（仅 DF + 高通），保证 nc 不受后级 STFT/EQ 影响
            if let Some(ref rec) = recording {
                if let Some(buffer) = outframe.as_slice() {
                    rec.append_denoised(buffer);
                }
            }

            if !bypass_enabled {
                if stft_enabled {
                    if stft_dirty {
                        stft_eq.set_curve(
                            stft_hf_gain,
                            stft_air_gain,
                            stft_tilt_gain,
                            &stft_band_offsets,
                            df.sr,
                        );
                        stft_dirty = false;
                    }
                    if let Some(buf) = outframe.as_slice_mut() {
                        stft_eq.process_block(buf);
                    }
                }
            }
            let bypass_post = bypass_enabled
                || (!dynamic_eq.is_enabled()
                    && !transient_enabled
                    && !saturation_enabled
                    && !agc_enabled);

            if !bypass_post {
                // 自适应峰值保护，防止 DF 输出直接过 0dBFS
                if let Some(buf) = outframe.as_slice_mut() {
                    let mut peak = 0.0f32;
                    for v in buf.iter() {
                        peak = peak.max(v.abs());
                    }
                    const TARGET: f32 = 0.9;
                    if peak > TARGET && peak.is_finite() && peak > 0.0 {
                        let mut scale = (TARGET / peak).min(1.0);
                        if headroom_gain < 0.9999 {
                            scale = scale.min(headroom_gain.max(0.0).min(1.0));
                        }
                        if scale < 0.9999 {
                            for v in buf.iter_mut() {
                                *v *= scale;
                            }
                        }
                    }
                }
            }
            let eq_start = Instant::now();
            let (eq_gain_db, eq_enabled_flag) = if bypass_post {
                ([0.0; MAX_EQ_BANDS], false)
            } else {
                if transient_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        transient_shaper.process(buffer);
                    }
                }
                if saturation_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        saturation.process(buffer);
                    }
                }
                // Add gentle harmonic excitation to restore air/brightness
                if exciter_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        exciter.process(buffer);
                    }
                }
                let metrics = {
                    let buffer =
                        outframe.as_slice_mut().expect("Output frame should be contiguous");
                    // 保持全湿，防止并行干声导致的相位波动
                    dynamic_eq.set_dry_wet(1.0);
                    dynamic_eq.process_block(buffer)
                };
                if post_trim_gain < 0.9999 {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        for v in buffer.iter_mut() {
                            *v *= post_trim_gain;
                        }
                    }
                }
                if agc_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        agc.process(buffer);
                    }
                }
                if let Some(buffer) = outframe.as_slice_mut() {
                    bus_limiter.process(buffer);
                }
                (metrics.gain_db, metrics.enabled)
            };
            // 削波检测在最终限幅前执行，驱动系统音量保护
            if auto_sys_volume && !SYS_VOL_MON_ACTIVE.load(Ordering::Relaxed) {
                // 在原始输入上检测电平，只降不升的小步调整
                if let Some(buf) = inframe.as_slice() {
                    let mut peak_in = 0.0f32;
                    for v in buf.iter() {
                        peak_in = peak_in.max(v.abs());
                    }
                    const RED_THRESH: f32 = 0.90; // > -1 dBFS
                    const DOWN_STEP: i8 = -4; // 每次下调约4%
                    const CLIP_FRAMES: usize = 1;
                    const COOLDOWN: Duration = Duration::from_millis(800);
                    const MIN_VOL: u8 = 50;
                    const MAX_VOL: u8 = 90;
                    const BLUE_THRESH: f32 = 0.05; // ~-26 dBFS，认为麦克风物理增益偏低
                    const RESTORE_COOLDOWN: Duration = Duration::from_secs(4);
                    const RESTORE_FLOOR: u8 = 35; // 不把系统音量降到监听不到的安全下限
                    if peak_in > RED_THRESH {
                        clip_counter += 1;
                        low_peak_counter = 0;
                    } else {
                        clip_counter = 0;
                        if peak_in < BLUE_THRESH {
                            low_peak_counter = low_peak_counter.saturating_add(1);
                        } else {
                            low_peak_counter = 0;
                        }
                    }
                    if clip_counter >= CLIP_FRAMES && last_sys_adjust.elapsed() > COOLDOWN {
                        clip_counter = 0;
                        last_sys_adjust = Instant::now();
                        let before = get_input_volume();
                        log::warn!(
                            "检测到输入削波 {:.3}，尝试下调系统输入音量 (当前 {:?})",
                            peak_in,
                            before
                        );
                        adjust_system_input_volume_async(DOWN_STEP, MIN_VOL, MAX_VOL);
                        warmup_blocks = warmup_blocks.saturating_add(3);
                    } else if low_peak_counter >= low_peak_required
                        && last_sys_restore.elapsed() > RESTORE_COOLDOWN
                    {
                        low_peak_counter = 0;
                        if let Some(current) = get_input_volume() {
                            if current < RESTORE_FLOOR {
                                let delta = RESTORE_FLOOR.saturating_sub(current) as i8;
                                log::warn!(
                                    "输入峰值持续过低，尝试恢复系统输入音量到底线 {} (当前 {})",
                                    RESTORE_FLOOR,
                                    current
                                );
                                adjust_system_input_volume_async(delta, RESTORE_FLOOR, MAX_VOL);
                                last_sys_restore = Instant::now();
                                last_sys_adjust = last_sys_restore;
                            }
                        }
                    }
                }
            }
            // 启动淡入，避免播放砰声
            if fade_progress < fade_total {
                if let Some(buffer) = outframe.as_slice_mut() {
                    for v in buffer.iter_mut() {
                        if fade_progress >= fade_total {
                            break;
                        }
                        let g = (fade_progress as f32 / fade_total as f32).min(1.0);
                        *v *= g;
                        fade_progress += 1;
                    }
                }
            }
            let eq_elapsed = eq_start.elapsed();
            let mut eq_cpu = 0.0;
            if block_duration > 0.0 {
                eq_cpu = (eq_elapsed.as_secs_f32() / block_duration).max(0.0) * 100.0;
            }
            if let Some(ref sender) = s_eq_status {
                let status = EqStatus {
                    gain_reduction_db: eq_gain_db,
                    cpu_load: eq_cpu.min(100.0),
                    enabled: eq_enabled_flag,
                    dry_wet: dynamic_eq.dry_wet(),
                    agc_gain_db: if agc_enabled {
                        agc.current_gain_db()
                    } else {
                        0.0
                    },
                };
                if sender.len() < 4 {
                    let _ = sender.try_send(status);
                }
            }
            // 最终限幅一次，避免多级限幅导致音色压缩
            if let Some(buffer) = outframe.as_slice_mut() {
                apply_final_limiter(buffer);
            }
            if auto_sys_volume && !SYS_VOL_MON_ACTIVE.load(Ordering::Relaxed) {
                if let Some(buffer) = outframe.as_slice() {
                    let mut peak = 0.0f32;
                    for v in buffer.iter() {
                        peak = peak.max(v.abs());
                    }
                    const CLIP_THRESH: f32 = 0.95;
                    const CLIP_FRAMES: usize = 2;
                    const COOLDOWN: Duration = Duration::from_secs(1);
                    if peak > CLIP_THRESH {
                        clip_counter += 1;
                    } else {
                        clip_counter = 0;
                    }
                    if clip_counter >= CLIP_FRAMES && last_sys_adjust.elapsed() > COOLDOWN {
                        clip_counter = 0;
                        last_sys_adjust = Instant::now();
                        adjust_system_input_volume_async(-15, 20, 90);
                    }
                }
            }
            // 录音最终输出（限幅后）
            if let Some(ref rec) = recording {
                if let Some(buffer) = outframe.as_slice_mut() {
                    rec.append_processed(buffer);
                }
            }
            if !mute_playback {
                if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
                    // 仅首次重采样时清零缓冲，避免未初始化数据脉冲
                    if !output_resampler_cleared {
                        for frame in buf.iter_mut() {
                            for sample in frame.iter_mut() {
                                *sample = 0.0;
                            }
                        }
                        output_resampler_cleared = true;
                    }
                    if let Err(err) =
                        r.process_into_buffer(&[outframe.as_slice().unwrap()], buf, None)
                    {
                        log::error!("输出重采样失败: {:?}", err);
                    } else if warmup_blocks > 0 {
                        warmup_blocks = warmup_blocks.saturating_sub(1);
                    } else {
                        push_output_block(&should_stop, &mut rb_out, &buf[0][..n_out], n_out);
                    }
                } else if let Some(buf) = outframe.as_slice() {
                    if warmup_blocks > 0 {
                        warmup_blocks = warmup_blocks.saturating_sub(1);
                    } else {
                        let temp = buf[..n_out].to_vec();
                        push_output_block(&should_stop, &mut rb_out, &temp[..], n_out);
                    }
                } else {
                    log::error!("输出帧内存布局异常，跳过输出");
                }
            }
            if let Some(sender) = s_lsnr.as_ref() {
                if let Err(err) = sender.send(lsnr) {
                    log::warn!("Failed to send LSNR value: {}", err);
                }
            }
            if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                push_spec(df.get_spec_noisy(), s_noisy);
                push_spec(df.get_spec_enh(), s_enh);
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok(message) = r_opt.try_recv() {
                    match message {
                        ControlMessage::DeepFilter(control, value) => match control {
                            DfControl::AttenLim => {
                                manual_atten = value;
                                df.set_atten_lim(value);
                            }
                            DfControl::PostFilterBeta => df.set_pf_beta(value),
                            DfControl::MinThreshDb => {
                                manual_min_thresh = value;
                                df.min_db_thresh = value;
                            }
                            DfControl::MaxErbThreshDb => df.max_db_erb_thresh = value,
                            DfControl::MaxDfThreshDb => {
                                manual_max_thresh = value;
                                df.max_db_df_thresh = value;
                                df.max_db_erb_thresh = value;
                            }
                        },
                        ControlMessage::Eq(control) => match control {
                            EqControl::SetEnabled(enabled) => dynamic_eq.set_enabled(enabled),
                        EqControl::SetPreset(preset) => dynamic_eq.apply_preset(preset),
                        // 干/湿混合固定全湿，避免梳状；忽略外部传入比例
                        EqControl::SetDryWet(_) => dynamic_eq.set_dry_wet(1.0),
                            EqControl::SetBandGain(idx, gain) => {
                                dynamic_eq.set_band_gain(idx, gain)
                            }
                            EqControl::SetBandFrequency(idx, freq) => {
                                dynamic_eq.set_band_frequency(idx, freq)
                            }
                            EqControl::SetBandQ(idx, q) => dynamic_eq.set_band_q(idx, q),
                            EqControl::SetBandDetectorQ(idx, value) => {
                                dynamic_eq.set_band_detector_q(idx, value)
                            }
                            EqControl::SetBandThreshold(idx, value) => {
                                dynamic_eq.set_band_threshold(idx, value)
                            }
                            EqControl::SetBandRatio(idx, value) => {
                                dynamic_eq.set_band_ratio(idx, value)
                            }
                            EqControl::SetBandMaxGain(idx, value) => {
                                dynamic_eq.set_band_max_gain(idx, value)
                            }
                            EqControl::SetBandAttack(idx, value) => {
                                dynamic_eq.set_band_attack(idx, value)
                            }
                            EqControl::SetBandRelease(idx, value) => {
                                dynamic_eq.set_band_release(idx, value)
                            }
                            EqControl::SetBandMakeup(idx, value) => {
                                dynamic_eq.set_band_makeup(idx, value)
                            }
                            EqControl::SetBandMode(idx, mode) => {
                                dynamic_eq.set_band_mode(idx, mode)
                            }
                            EqControl::SetBandFilter(idx, filter) => {
                                dynamic_eq.set_band_filter(idx, filter)
                            }
                        },
                        ControlMessage::MutePlayback(muted) => {
                            mute_playback = muted;
                            log::info!("播放静音: {}", if mute_playback { "开启" } else { "关闭" });
                        }
                        ControlMessage::HighpassEnabled(enabled) => {
                            highpass_enabled = enabled;
                            if !enabled {
                                highpass.reset();
                            }
                            log::info!("高通滤波: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::HighpassCutoff(freq) => {
                            manual_highpass = freq;
                            highpass_cutoff = freq;
                            highpass.set_cutoff(freq);
                            log::info!("高通截止频率: {:.0} Hz", freq);
                        }
                        ControlMessage::DfMix(_) => {
                            // 固定全湿，忽略外部比例，避免干/湿并行相位问题
                            _df_mix = 1.0;
                            log::info!("DF 混合比例固定为 100%");
                        }
                        ControlMessage::HeadroomGain(gain) => {
                            headroom_gain = gain.clamp(0.0, 1.0);
                            log::info!("Headroom 增益: {:.2}", headroom_gain);
                        }
                        ControlMessage::PostTrimGain(gain) => {
                            post_trim_gain = gain.clamp(0.0, 1.0);
                            log::info!("Post-Trim 增益: {:.2}", post_trim_gain);
                        }
                        ControlMessage::SaturationEnabled(enabled) => {
                            saturation_enabled = enabled;
                            log::info!("饱和度: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::SaturationDrive(drive) => {
                            saturation.set_drive(drive);
                            log::info!("饱和驱动: {:.2}", drive);
                        }
                        ControlMessage::SaturationMakeup(db) => {
                            saturation.set_makeup(db);
                            log::info!("饱和补偿: {:+.1} dB", db);
                        }
                        ControlMessage::SaturationMix(ratio) => {
                            saturation.set_mix((ratio / 100.0).clamp(0.0, 1.0));
                            log::info!("饱和混合: {:.0}%", ratio);
                        }
                        ControlMessage::TransientEnabled(enabled) => {
                            transient_enabled = enabled;
                            if !enabled {
                                transient_shaper.reset();
                            }
                            log::info!("瞬态增强: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::TransientGain(db) => {
                            transient_shaper.set_attack_gain(db);
                        }
                        ControlMessage::TransientSustain(db) => {
                            transient_shaper.set_sustain_gain(db);
                        }
                        ControlMessage::TransientMix(ratio) => {
                            transient_shaper.set_dry_wet((ratio / 100.0).clamp(0.0, 1.0));
                        }
                        ControlMessage::AgcEnabled(enabled) => {
                            agc_enabled = enabled;
                            agc.reset();
                            log::info!("AGC: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::AgcTargetLevel(db) => {
                            agc.set_target_level(db);
                            log::info!("AGC 目标电平: {:.1} dBFS", db);
                        }
                        ControlMessage::AgcMaxGain(db) => {
                            agc.set_max_gain(db);
                            log::info!("AGC 最大增益: {:.1} dB", db);
                        }
                        ControlMessage::AgcMaxAttenuation(db) => {
                            agc.set_max_attenuation(db);
                            log::info!("AGC 最大衰减: {:.1} dB", db);
                        }
                        ControlMessage::AgcWindowSeconds(sec) => {
                            agc.set_window_seconds(sec);
                            log::info!("AGC 窗长: {:.2} s", sec);
                        }
                        ControlMessage::AgcAttackRelease(attack_ms, release_ms) => {
                            agc.set_attack_release(attack_ms, release_ms);
                            log::info!("AGC 攻击/释放: {:.0} / {:.0} ms", attack_ms, release_ms);
                        }
                        ControlMessage::SysAutoVolumeEnabled(enabled) => {
                            auto_sys_volume = enabled;
                            log::info!(
                                "自动系统音量保护: {}",
                                if enabled { "开启" } else { "关闭" }
                            );
                        }
                        ControlMessage::EnvAutoEnabled(enabled) => {
                            env_auto_enabled = enabled;
                            log::info!("环境自适应: {}", if enabled { "开启" } else { "关闭" });
                            if !enabled {
                                soft_mode = false;
                                last_soft_mode = false;
                                if let Some(ref sender) = s_env_status {
                                    let _ = sender.try_send(EnvStatus::Normal);
                                }
                            }
                        }
                        ControlMessage::ExciterEnabled(enabled) => {
                            exciter_enabled = enabled;
                            log::info!("谐波激励: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::ExciterMix(value) => {
                            exciter.set_mix(value.clamp(0.0, 0.5));
                            log::info!("谐波激励混合: {:.0}%", value * 100.0);
                        }
                        ControlMessage::StftEqEnabled(enabled) => {
                            stft_enabled = enabled;
                            log::info!("STFT 静态 EQ: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::StftEqHfGain(db) => {
                            stft_hf_gain = db.clamp(0.0, 8.0);
                            stft_dirty = true;
                            log::info!("STFT EQ 高频补偿: {:+.1} dB", stft_hf_gain);
                        }
                        ControlMessage::StftEqAirGain(db) => {
                            stft_air_gain = db.clamp(0.0, 10.0);
                            stft_dirty = true;
                            log::info!("STFT EQ 空气感补偿: {:+.1} dB", stft_air_gain);
                        }
                        ControlMessage::StftEqTilt(db) => {
                            stft_tilt_gain = db.clamp(-6.0, 6.0);
                            stft_dirty = true;
                            log::info!("STFT EQ 倾斜补偿: {:+.1} dB", stft_tilt_gain);
                        }
                        ControlMessage::StftEqBand(idx, db) => {
                            if idx < stft_band_offsets.len() {
                                stft_band_offsets[idx] = db.clamp(-6.0, 6.0);
                                stft_dirty = true;
                                log::info!("STFT EQ 静态段 {} 调整: {:+.1} dB", idx + 1, db);
                            }
                        }
                        ControlMessage::BypassEnabled(enabled) => {
                            bypass_enabled = enabled;
                            log::info!("全链路旁路: {}", if enabled { "开启" } else { "关闭" });
                        }
                    }
                }
            }
        }
    }
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    let out = spec.iter().map(|x| x.norm_sqr().max(1e-10).log10() * 10.).collect::<Vec<f32>>();
    if let Err(err) = sender.send(out.into_boxed_slice()) {
        log::warn!("Failed to send spectrogram data: {}", err);
    }
}

fn push_output_block(
    should_stop: &Arc<AtomicBool>,
    rb_out: &mut RbProd,
    data: &[f32],
    expected_frames: usize,
) {
    debug_assert!(data.len() >= expected_frames);
    let timeout = Duration::from_millis(20);
    let retry_delay = Duration::from_micros(100);
    let start_time = Instant::now();
    let mut n = 0usize;
    while n < expected_frames {
        if should_stop.load(Ordering::Relaxed) {
            log::debug!("停止播放输出（检测到停止信号）");
            break;
        }
        if start_time.elapsed() > timeout {
            log::warn!("播放输出超时，跳过 {} 个样本", expected_frames - n);
            break;
        }
        let pushed = rb_out.push_slice(&data[n..expected_frames]);
        if pushed == 0 {
            sleep(retry_delay);
        } else {
            n += pushed;
        }
    }
    rb_out.sync();
}

struct BusLimiter {
    gain: f32,
    attack_coef: f32,
    release_coef: f32,
    ceiling: f32,
    peak_trigger: f32,
    impact_threshold: f32,
    impact_ceiling: f32,
    lookahead: Vec<f32>,
    la_pos: usize,
}

impl BusLimiter {
    fn new(sample_rate: usize) -> Self {
        // 更快攻击捕捉冲击，稍长释放避免泵效
        let attack_ms = 0.3;
        let release_ms = 100.0;
        let sr = sample_rate.max(1) as f32;
        let attack_coef = (-1000.0 / (attack_ms * sr)).exp();
        let release_coef = (-1000.0 / (release_ms * sr)).exp();
        // lookahead 稍长以覆盖跌落/敲击类冲击
        let lookahead_ms = 6.0;
        Self {
            gain: 1.0,
            attack_coef,
            release_coef,
            ceiling: 0.90,
            peak_trigger: 0.85,
            impact_threshold: 1.2,
            impact_ceiling: 0.90,
            lookahead: vec![0.0; (sr * lookahead_ms / 1000.0) as usize],
            la_pos: 0,
        }
    }

    fn process(&mut self, buf: &mut [f32]) {
        if buf.is_empty() {
            return;
        }
        // 预夹击：遇到极端尖峰先整体缩放，防止后级超限
        let mut block_peak = 0.0f32;
        for v in buf.iter() {
            block_peak = block_peak.max(v.abs());
        }
        if block_peak > self.impact_threshold && block_peak.is_finite() {
            let scale = (self.impact_ceiling / block_peak).clamp(0.1, 1.0);
            if scale < 0.999 {
                for v in buf.iter_mut() {
                    *v *= scale;
                }
            }
        }
        // copy into lookahead buffer to peek peaks slightly ahead
        let la_len = self.lookahead.len();
        for v in buf.iter() {
            let pos = self.la_pos % la_len;
            self.lookahead[pos] = *v;
            self.la_pos = (self.la_pos + 1) % la_len;
        }
        let mut peak = 0.0f32;
        for v in self.lookahead.iter() {
            peak = peak.max(v.abs());
        }
        let target = if peak > self.peak_trigger && peak.is_finite() && peak > 0.0 {
            self.ceiling / peak
        } else {
            1.0
        };
        let coef = if target < self.gain {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.gain = coef * self.gain + (1.0 - coef) * target;
        if self.gain < 0.9999 || self.gain > 1.0001 {
            for v in buf.iter_mut() {
                *v *= self.gain;
            }
        }
    }
}

fn apply_final_limiter(data: &mut [f32]) {
    let mut peak = 0.0f32;
    for sample in data.iter() {
        peak = peak.max(sample.abs());
    }
    const CEILING: f32 = 0.92;
    if peak > CEILING && peak.is_finite() {
        let scale = CEILING / peak;
        for sample in data.iter_mut() {
            *sample *= scale;
        }
    }
}

pub struct SysVolMonitorHandle {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SysVolMonitorHandle {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        SYS_VOL_MON_ACTIVE.store(false, Ordering::Relaxed);
    }
}

impl Drop for SysVolMonitorHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        SYS_VOL_MON_ACTIVE.store(false, Ordering::Relaxed);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn start_sys_volume_monitor(_selected_input: Option<String>) -> Option<SysVolMonitorHandle> {
    log::info!("自动系统音量后台监测仅在 macOS 可用");
    None
}

#[cfg(target_os = "macos")]
pub fn start_sys_volume_monitor(selected_input: Option<String>) -> Option<SysVolMonitorHandle> {
    use cpal::traits::DeviceTrait;
    use cpal::SampleFormat;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = stop.clone();
    let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<(), String>>(1);
    let handle = thread::spawn(move || {
        let host = cpal::default_host();
        let device = selected_input
            .and_then(|name| {
                host.input_devices().ok().and_then(|mut devs| {
                    devs.find(|d| d.name().ok().as_deref() == Some(name.as_str()))
                })
            })
            .or_else(|| host.default_input_device());

        let device = match device {
            Some(dev) => dev,
            None => {
                let _ = ready_tx.send(Err("未找到输入设备".to_string()));
                return;
            }
        };

        let cfg = match device.default_input_config() {
            Ok(c) => c,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("获取输入配置失败 {}", err)));
                return;
            }
        };
        let sample_format = cfg.sample_format();
        let stream_config: StreamConfig = cfg.into();
        let sr = stream_config.sample_rate.0 as f32;
        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(8);

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    let _ = tx.try_send(data.to_vec());
                },
                move |err| log::warn!("系统音量监听流错误: {}", err),
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let frame: Vec<f32> =
                        data.iter().map(|&v| v as f32 / i16::MAX as f32).collect();
                    let _ = tx.try_send(frame);
                },
                move |err| log::warn!("系统音量监听流错误: {}", err),
                None,
            ),
            SampleFormat::U16 => device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    let frame: Vec<f32> =
                        data.iter().map(|&v| (v as f32 - 32768.0) / 32768.0).collect();
                    let _ = tx.try_send(frame);
                },
                move |err| log::warn!("系统音量监听流错误: {}", err),
                None,
            ),
            other => {
                let _ = ready_tx.send(Err(format!("不支持的采样格式: {:?}", other)));
                return;
            }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(err) => {
                let _ = ready_tx.send(Err(format!("创建输入流失败 {}", err)));
                return;
            }
        };

        if let Err(err) = stream.play() {
            let _ = ready_tx.send(Err(format!("输入流播放失败 {}", err)));
            return;
        }

        let _ = ready_tx.send(Ok(()));
        SYS_VOL_MON_ACTIVE.store(true, Ordering::Relaxed);
        let mut last_adjust =
            Instant::now().checked_sub(Duration::from_secs(5)).unwrap_or_else(Instant::now);
        let mut last_restore = last_adjust;
        let mut clip_counter = 0usize;
        let mut low_peak_counter = 0usize;
        let mut low_peak_required = 60usize;
        let mut block_duration = 0.0f32;

        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }
            let chunk = match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(c) => c,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            if block_duration == 0.0 && sr > 0.0 {
                block_duration = chunk.len() as f32 / sr.max(1.0);
                if block_duration > 0.0 {
                    low_peak_required = ((3.0 / block_duration).ceil() as usize).max(30);
                }
            }
            let mut peak = 0.0f32;
            for &v in chunk.iter() {
                if v.is_finite() {
                    peak = peak.max(v.abs());
                }
            }

            const RED_THRESH: f32 = 0.90;
            const BLUE_THRESH: f32 = 0.05;
            const DOWN_STEP: i8 = -4;
            const CLIP_FRAMES: usize = 1;
            const DOWN_COOLDOWN: Duration = Duration::from_millis(800);
            const RESTORE_COOLDOWN: Duration = Duration::from_secs(4);
            const MIN_VOL: u8 = 50;
            const MAX_VOL: u8 = 90;
            const RESTORE_FLOOR: u8 = 35;

            if peak > RED_THRESH {
                clip_counter += 1;
                low_peak_counter = 0;
            } else {
                clip_counter = 0;
                if peak < BLUE_THRESH {
                    low_peak_counter = low_peak_counter.saturating_add(1);
                } else {
                    low_peak_counter = 0;
                }
            }

            if clip_counter >= CLIP_FRAMES && last_adjust.elapsed() > DOWN_COOLDOWN {
                clip_counter = 0;
                last_adjust = Instant::now();
                let before = get_input_volume();
                log::warn!(
                    "后台监测检测到输入削波 {:.3}，尝试下调系统输入音量 (当前 {:?})",
                    peak,
                    before
                );
                adjust_system_input_volume_async(DOWN_STEP, MIN_VOL, MAX_VOL);
            } else if low_peak_counter >= low_peak_required
                && last_restore.elapsed() > RESTORE_COOLDOWN
            {
                low_peak_counter = 0;
                if let Some(current) = get_input_volume() {
                    if current < RESTORE_FLOOR {
                        let delta = RESTORE_FLOOR.saturating_sub(current) as i8;
                        log::warn!(
                            "后台监测输入峰值持续过低，尝试恢复系统输入音量到底线 {} (当前 {})",
                            RESTORE_FLOOR,
                            current
                        );
                        adjust_system_input_volume_async(delta, RESTORE_FLOOR, MAX_VOL);
                        last_restore = Instant::now();
                        last_adjust = last_restore;
                    }
                }
            }
        }
        SYS_VOL_MON_ACTIVE.store(false, Ordering::Relaxed);
        drop(stream);
    });

    match ready_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => Some(SysVolMonitorHandle {
            stop,
            handle: Some(handle),
        }),
        Ok(Err(err)) => {
            log::warn!("后台系统音量监测未启动：{}", err);
            let _ = handle.join();
            None
        }
        Err(_) => {
            log::warn!("后台系统音量监测启动超时");
            let _ = handle.join();
            None
        }
    }
}

#[cfg(target_os = "macos")]
fn get_input_volume() -> Option<u8> {
    let output = Command::new("osascript")
        .arg("-e")
        .arg("input volume of (get volume settings)")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout);
    s.trim().parse::<u8>().ok()
}

#[cfg(target_os = "macos")]
fn set_input_volume(vol: u8) -> Result<(), String> {
    let cmd = format!("set volume input volume {}", vol);
    let status = Command::new("osascript")
        .arg("-e")
        .arg(cmd)
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("osascript exited with status {}", status))
    }
}

#[cfg(target_os = "macos")]
fn adjust_system_input_volume_async(delta_percent: i8, min: u8, max: u8) {
    std::thread::spawn(move || {
        if let Some(current) = get_input_volume() {
            let delta = delta_percent.clamp(-100, 100);
            let new = current.saturating_add_signed(delta).clamp(min, max);
            match set_input_volume(new) {
                Ok(_) => log::warn!("系统输入音量调整: {} -> {}", current, new),
                Err(err) => log::warn!("系统输入音量调整失败: {}", err),
            }
        } else {
            log::warn!("无法读取系统输入音量");
        }
    });
}

#[cfg(not(target_os = "macos"))]
fn adjust_system_input_volume_async(_delta_percent: i8, _min: u8, _max: u8) {
    // No-op on non-macOS platforms
}

fn smooth_value(current: f32, target: f32, alpha: f32) -> f32 {
    current + (target - current) * alpha.clamp(0.0, 1.0)
}

fn compute_noise_features(spec: ArrayView2<Complex32>) -> NoiseFeatures {
    let (_, freq_len) = spec.dim();
    let freq_len_f32 = freq_len.max(1) as f32;
    let row = spec.row(0);
    let eps = 1e-12f32;
    let mut sum_power = 0.0;
    let mut sum_log_power = 0.0;
    let mut weighted_sum = 0.0;
    for (i, &c) in row.iter().enumerate() {
        let p = c.norm_sqr().max(eps);
        sum_power += p;
        sum_log_power += p.ln();
        weighted_sum += p * i as f32;
    }
    let mean_power = sum_power / freq_len_f32;
    let energy_db = 10.0 * mean_power.max(eps).log10();
    let geometric_mean = (sum_log_power / freq_len_f32).exp();
    let spectral_flatness = geometric_mean / mean_power.max(eps);
    let spectral_centroid = if sum_power > 0.0 {
        (weighted_sum / sum_power) / freq_len_f32
    } else {
        0.0
    };
    NoiseFeatures {
        energy_db,
        spectral_flatness,
        spectral_centroid,
    }
}

fn classify_env(energy_db: f32, flatness: f32, centroid: f32) -> EnvClass {
    // 更敏感的噪声判定，优先进入 Noisy，重度降噪
    if energy_db > -60.0 {
        EnvClass::Noisy
    } else if flatness > 0.25 || centroid > 0.25 {
        EnvClass::Office
    } else {
        EnvClass::Quiet
    }
}
#[derive(Clone, Copy)]
struct SnrParams {
    atten: f32,
    min_thresh: f32,
    max_thresh: f32,
    df_mix: f32,
    hp_cut: f32,
    eq_mix: f32,
    exciter_mix: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn map_snr_to_params(snr_db: f32) -> SnrParams {
    let points = [
        (
            -5.0,
            SnrParams {
                atten: 60.0,
                min_thresh: -55.0,
                max_thresh: 8.0,
                df_mix: 1.0,
                hp_cut: 110.0,
                eq_mix: 0.5,
                exciter_mix: 0.05,
            },
        ),
        (
            5.0,
            SnrParams {
                atten: 50.0,
                min_thresh: -53.0,
                max_thresh: 10.0,
                df_mix: 1.0,
                hp_cut: 90.0,
                eq_mix: 0.6,
                exciter_mix: 0.10,
            },
        ),
        (
            15.0,
            SnrParams {
                atten: 32.0,
                min_thresh: -56.0,
                max_thresh: 12.0,
                df_mix: 0.9,
                hp_cut: 70.0,
                eq_mix: 0.7,
                exciter_mix: 0.18,
            },
        ),
        (
            25.0,
            SnrParams {
                atten: 22.0,
                min_thresh: -58.0,
                max_thresh: 14.0,
                df_mix: 0.8,
                hp_cut: 55.0,
                eq_mix: 0.8,
                exciter_mix: 0.22,
            },
        ),
    ];
    let snr = snr_db.clamp(points[0].0, points[3].0);
    let mut i = 0;
    while i + 1 < points.len() && snr > points[i + 1].0 {
        i += 1;
    }
    let (s0, p0) = points[i];
    let (s1, p1) = points[i + 1];
    let t = if (s1 - s0).abs() > 1e-6 {
        (snr - s0) / (s1 - s0)
    } else {
        0.0
    };
    SnrParams {
        atten: lerp(p0.atten, p1.atten, t),
        min_thresh: lerp(p0.min_thresh, p1.min_thresh, t),
        max_thresh: lerp(p0.max_thresh, p1.max_thresh, t),
        df_mix: lerp(p0.df_mix, p1.df_mix, t),
        hp_cut: lerp(p0.hp_cut, p1.hp_cut, t),
        eq_mix: lerp(p0.eq_mix, p1.eq_mix, t),
        exciter_mix: lerp(p0.exciter_mix, p1.exciter_mix, t),
    }
}

pub fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = record.module_path().unwrap_or("").to_string();
    let level_style = buf.default_level_style(log::Level::Info);

    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        level_style.value(record.level()),
        module,
        record.args()
    )
}

#[allow(dead_code)]
pub struct DeepFilterCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    source: AudioSource,
    sink: AudioSink,
    recording: RecordingHandle,
}

impl Default for DeepFilterCapture {
    fn default() -> Self {
        DeepFilterCapture::new(None, None, None, None, None, None, None, None, None)
            .expect("Error during DeepFilterCapture initialization")
    }
}
impl DeepFilterCapture {
    pub fn new(
        model_path: Option<PathBuf>,
        s_lsnr: Option<SendLsnr>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
        s_eq_status: Option<SendEqStatus>,
        s_env_status: Option<SendEnvStatus>,
        input_device: Option<String>,
        output_device: Option<String>,
    ) -> Result<Self> {
        let ch = 1;
        if model_path.is_some() {
            set_model_path(model_path.clone());
        }
        let metadata = resolve_model_metadata(model_path.or_else(get_model_path))?;
        let df_params = metadata.params.clone();
        let sr = metadata.sr;
        let frame_size = metadata.frame_size;
        let freq_size = metadata.freq_size;
        let in_rb = HeapRb::<f32>::new(frame_size * 400);
        let out_rb = HeapRb::<f32>::new(frame_size * 100);
        let (in_prod, in_cons) = in_rb.split();
        let (out_prod, out_cons) = out_rb.split();
        let in_prod = in_prod.into_postponed();
        let mut out_prod = out_prod.into_postponed();
        {
            // 预填充静音，避免启动瞬间砰声
            let zeros = vec![0.0f32; frame_size * 4];
            let mut pushed = 0;
            while pushed < zeros.len() {
                let n = out_prod.push_slice(&zeros[pushed..]);
                if n == 0 {
                    break;
                }
                pushed += n;
            }
            out_prod.sync();
        }

        let mut source = AudioSource::new(sr as u32, frame_size, input_device)?;
        let mut sink = AudioSink::new(sr as u32, frame_size, output_device)?;
        let should_stop = Arc::new(AtomicBool::new(false));
        let has_init = Arc::new(AtomicBool::new(false));
        let s_spec = match (s_noisy, s_enh) {
            (Some(n), Some(e)) => Some((n, e)),
            _ => None,
        };
        let controls = AtomicControls {
            has_init: has_init.clone(),
            should_stop: should_stop.clone(),
        };
        let recording = Arc::new(RecordingState::new(sr));
        let df_com = GuiCom {
            s_lsnr,
            s_spec,
            r_opt,
            s_eq_status,
            s_env_status,
        };
        let worker_handle = Some(thread::spawn(get_worker_fn(
            in_cons,
            out_prod,
            source.sr() as usize,
            sink.sr() as usize,
            controls,
            Some(df_com),
            Some(recording.clone()),
            df_params,
            ch,
        )));
        while !has_init.load(Ordering::Relaxed) {
            sleep(Duration::from_secs_f32(0.01));
        }
        log::info!("DeepFilter Capture init");
        source.start(in_prod)?;
        sink.start(out_cons)?;

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            source,
            sink,
            recording,
        })
    }

    pub fn recording(&self) -> RecordingHandle {
        self.recording.clone()
    }

    pub fn should_stop(&mut self) -> Result<()> {
        self.should_stop.swap(true, Ordering::Relaxed);
        sleep(Duration::from_millis(150));
        self.sink.pause()?;
        self.source.pause()?;
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining DF Worker");
            match h.join() {
                Ok(_) => log::info!("DF Worker stopped successfully"),
                Err(err) => log::error!("DF worker join failed: {:?}", err),
            }
        }
        Ok(())
    }
}

#[allow(unused)]
#[allow(unknown_lints)] // assigning_clones is clippy nightly only
#[allow(clippy::assigning_clones)]
pub fn main() -> Result<()> {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .format(log_format)
            .init();
    });

    let (lsnr_prod, mut lsnr_cons) = unbounded();
    let mut model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
    if model_path.is_none() {
        model_path = get_model_path();
    }
    if let Some(p) = model_path.as_ref() {
        log::info!("Running with model '{:?}'", p);
    }
    let _c = DeepFilterCapture::new(
        model_path,
        Some(lsnr_prod),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    loop {
        sleep(Duration::from_millis(200));
        while let Ok(lsnr) = lsnr_cons.try_recv() {
            print!("\rCurrent SNR: {:>5.1} dB{esc}[1;", lsnr, esc = 27 as char);
        }
        stdout().flush().unwrap();
    }
}
