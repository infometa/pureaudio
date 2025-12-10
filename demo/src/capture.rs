use std::cell::RefCell;
use std::collections::VecDeque;
use std::env;
use std::fmt::Display;
use std::io::{self, stdout, Write};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use crate::audio::agc::AutoGainControl;
use crate::audio::aec::EchoCanceller;
use crate::audio::eq::{DynamicEq, EqControl, EqPresetKind, EqProcessMetrics, MAX_EQ_BANDS};
use crate::audio::exciter::HarmonicExciter;
use crate::audio::highpass::HighpassFilter;
use crate::audio::saturation::Saturation;
use crate::audio::timbre_restore::TimbreRestore;
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
use crate::audio::silero::{SileroVad, SileroVadConfig};

#[allow(dead_code)]
const OUTPUT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/output");
// CARGO_MANIFEST_DIR 已在 demo/，无需重复 demo 前缀
const TIMBRE_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/timbre_restore.onnx");
const TIMBRE_CONTEXT: usize = 256;
const TIMBRE_HIDDEN: usize = 384;
const TIMBRE_LAYERS: usize = 2;
const SILERO_VAD_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
// 内部处理采样率/帧长固定为 48 kHz / 10 ms，只在 IO 边界做重采样，保持 AEC/DF/VAD 对齐
const PROCESS_SR: usize = 48_000;
const PROCESS_HOP: usize = PROCESS_SR / 100;
const AEC_DEFAULT_DELAY_MS: i32 = 60;

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
#[allow(dead_code)]
pub type RecvLsnr = Receiver<f32>;
#[allow(dead_code)]
pub type SendSpec = Sender<Box<[f32]>>;
#[allow(dead_code)]
pub type RecvSpec = Receiver<Box<[f32]>>;
#[allow(dead_code)]
pub type SendControl = Sender<ControlMessage>;
#[allow(dead_code)]
pub type RecvControl = Receiver<ControlMessage>;
#[allow(dead_code)]
pub type SendEqStatus = Sender<EqStatus>;
#[allow(dead_code)]
pub type RecvEqStatus = Receiver<EqStatus>;
#[allow(dead_code)]
pub type SendEnvStatus = Sender<EnvStatus>;
#[allow(dead_code)]
pub type RecvEnvStatus = Receiver<EnvStatus>;

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
static INPUT_DROPPED_FRAMES: AtomicU64 = AtomicU64::new(0);
static OUTPUT_UNDERRUNS: AtomicU64 = AtomicU64::new(0);

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

#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct RecordingState {
    sr: usize,
    max_samples: usize,
    noisy: Mutex<Vec<f32>>,
    denoised: Mutex<Vec<f32>>,
    processed: Mutex<Vec<f32>>,
    timbre: Mutex<Vec<f32>>,
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
            timbre: Mutex::new(Vec::with_capacity(reserve)),
        }
    }

    #[allow(dead_code)]
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

    pub fn append_timbre(&self, samples: &[f32]) {
        if let Ok(mut buf) = self.timbre.lock() {
            self.append_with_limit(&mut buf, samples, "timbre");
        }
    }

    fn append_with_limit(&self, buf: &mut Vec<f32>, samples: &[f32], _tag: &str) {
        if buf.len() >= self.max_samples {
            // 长时间运行时录音缓冲达到上限，静默丢弃，避免日志刷屏
            return;
        }
        let available = self.max_samples - buf.len();
        let to_copy = available.min(samples.len());
        buf.extend_from_slice(&samples[..to_copy]);
        // 达到容量后继续静默丢弃，避免日志刷屏
    }

    #[allow(dead_code)]
    pub fn take_samples(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let noisy = self.noisy.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        let denoised =
            self.denoised.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        let processed =
            self.processed.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        let timbre = self.timbre.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
        (noisy, denoised, timbre, processed)
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    VadEnabled(bool),
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
    TimbreEnabled(bool),
    AecEnabled(bool),
    AecDelayMs(i32),
    AecAggressive(bool),
    SpecEnabled(bool),
    Rt60Enabled(bool),
    FinalLimiterEnabled(bool),
}

#[allow(dead_code)]
pub fn model_dimensions(
    model_path: Option<PathBuf>,
    _channels: usize,
) -> Result<(usize, usize, usize)> {
    let path = model_path.or_else(get_model_path);
    let meta = resolve_model_metadata(path).map_err(|e| anyhow!("加载模型元数据失败: {e}"))?;
    Ok((meta.sr, meta.frame_size, meta.freq_size))
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
        INPUT_DROPPED_FRAMES.fetch_add(dropped as u64, Ordering::Relaxed);
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
                // 环形缓冲欠载：用上一帧值保持，不再衰减到 0，避免音量忽高忽低
                let prev: Vec<f32> = if n > 0 {
                    data[(n * ch as usize - ch as usize)..(n * ch as usize)].to_vec()
                } else {
                    vec![0.0; ch as usize]
                };
                for frame in data.chunks_mut(ch as usize).skip(n) {
                    frame.clone_from_slice(&prev);
                    n += 1;
                }
                OUTPUT_UNDERRUNS.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    } else {
        while n < frames {
            let popped = rb.pop_slice(&mut data[n..]);
            if popped == 0 {
                let prev = if n > 0 {
                    data[(n - 1)..n].to_vec()
                } else {
                    vec![0.0; 1]
                };
                let remain = frames.saturating_sub(n);
                for i in 0..remain {
                    let idx = n + i;
                    let start = idx;
                    let end = start + 1;
                    data[start..end].clone_from_slice(&prev[..]);
                }
                OUTPUT_UNDERRUNS.fetch_add(1, Ordering::Relaxed);
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
        let Some(mut device) = host.default_output_device() else {
            return Err(anyhow!("未找到默认输出设备"));
        };
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
    #[allow(dead_code)]
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
        let Some(mut device) = host.default_input_device() else {
            return Err(anyhow!("未找到默认输入设备"));
        };
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
    #[allow(dead_code)]
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
    input_capacity_frames: usize,
) -> impl FnMut() {
    let (has_init, should_stop) = controls.into_inner();
    let (s_lsnr, mut s_spec, mut r_opt, s_eq_status, s_env_status) = if let Some(df_com) = df_com {
        df_com.into_inner()
    } else {
        (None, None, None, None, None)
    };
    let recording = recorder.clone();
    move || {
        let mut df = match DfTract::new(df_params.clone(), &RuntimeParams::default_with_ch(channels)) {
            Ok(df) => df,
            Err(err) => {
                log::error!("初始化 DeepFilterNet 失败: {:?}", err);
                return;
            }
        };
        debug_assert_eq!(df.ch, 1); // Processing for more channels are not implemented yet
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        if let Err(err) = df.process(inframe.view(), outframe.view_mut()) {
            log::error!("初始化 DF 首帧失败: {:?}", err);
            return;
        }
        let mut dynamic_eq = DynamicEq::new(df.sr as f32, EqPresetKind::default());
        let mut highpass = HighpassFilter::new(df.sr as f32);
        let mut exciter = HarmonicExciter::new(df.sr as f32, 5000.0, 2.0, 0.30);
        let mut transient_shaper = TransientShaper::new(df.sr as f32);
        // 记录用户配置的瞬态参数，便于在冲击抑制时暂时改写
        let mut transient_attack_db = 3.5f32;
        transient_shaper.set_attack_gain(transient_attack_db);
        let mut saturation = Saturation::new();
        let mut agc = AutoGainControl::new(df.sr as f32, df.hop_size);
        let mut aec = EchoCanceller::new(df.sr as f32, df.hop_size, AEC_DEFAULT_DELAY_MS);
        let mut timbre_restore: Option<TimbreRestore> = None;
        let mut timbre_load_failed = false;
        // VAD 懒加载，避免未开启时的重采样开销
        let mut vad: Option<SileroVad> = None;
        let vad_target_sr = 16000usize;
        let vad_frame_len = vad_target_sr / 1000 * 30; // 30ms @16k = 480
        let vad_source_frame = vad_frame_len;
        let mut vad_resampler: Option<(FftFixedIn<f32>, Vec<Vec<f32>>)> = None;
        let mut vad_buf_raw: VecDeque<f32> = VecDeque::with_capacity(vad_source_frame.max(1) * 3);
        let mut vad_frame_buf: Vec<f32> = vec![0.0f32; vad_source_frame.max(1)];
        let mut vad_drop_count: usize = 0;
        let mut vad_drop_last_log = Instant::now();
        let mut vad_resample_error_count: usize = 0;
        let mut vad_resample_error_last_log = Instant::now();
        let mut vad_oversample_warn_last = Instant::now();
        // 短淡入，避免启动瞬间脉冲，缩短至 20ms 保护首音
        let fade_total = (df.sr as f32 * 0.02) as usize; // 20ms fade-in
        let mut fade_progress = 0usize;
        let mut timbre_overload_frames = 0usize;
        let mut timbre_stride = 1usize;
        let mut timbre_skip_idx = 0usize;
        let mut timbre_last_good = Instant::now();
        let mut highpass_enabled = false;
        let mut highpass_cutoff = 50.0f32;
        let mut manual_highpass = highpass_cutoff;
        highpass.set_cutoff(highpass_cutoff);
        let mut transient_enabled = false;
        let mut saturation_enabled = false;
        let mut agc_enabled = false;
        let mut aec_enabled = false;
        let mut aec_delay_ms: i32;
        let mut _aec_aggressive = true;
        let mut exciter_enabled = false;
        let mut _timbre_enabled = false;
        let mut bypass_enabled = false;
        let mut mute_playback = false;
        let mut _df_mix = 1.0f32;
        // 耳机场景：预留更多头间距，减少 clipping
        let mut headroom_gain = 0.92f32;
        let mut post_trim_gain = 1.0f32;
        let mut spec_enabled = true;
        let mut rt60_enabled = true;
        let mut final_limiter_enabled = false;
        // 自适应环境/设备状态（默认关闭，通过 UI 控制开启）
        let mut env_class = EnvClass::Noisy;
        let mut smoothed_energy = -80.0f32;
        let mut smoothed_flatness = 0.0f32;
        let mut smoothed_centroid = 0.0f32;
        let mut smoothed_rt60 = 0.35f32;
        let mut last_env_log = Instant::now();
        // 环境特征节流
        let mut feature_counter = 0usize;
        let mut cached_feats = NoiseFeatures::default();
        let mut vad_enabled = false;
        let mut vad_state = false;
        let mut vad_voice_count = 0usize;
        let mut vad_noise_count = 0usize;
        // 噪声门控已禁用（交给 WebRTC AGC/DF 处理），固定全通
        let mut _gate_gain = 1.0f32;
        // 噪声地板 & SNR 跟踪
        let mut noise_floor_db = -60.0f32;
        let mut snr_db = 10.0f32;
        let mut target_atten;
        let mut target_min_thresh;
        let mut target_max_thresh;
        let mut target_hp;
        let mut target_exciter_mix;
        let mut target_transient_sustain;
        let mut manual_min_thresh = df.min_db_thresh;
        let mut manual_max_thresh = df.max_db_df_thresh;
        let mut office_factor = 0.5f32;
        // 冲击/键盘保护计数
        let mut impact_hold = 0usize;
        // 呼吸/衣物摩擦等近场气流噪声的抑制计数
        let mut breath_hold = 0usize;
        const IMPACT_HOLD_FRAMES: usize = 120; // 更长保持，强压键盘/点击
                                              // 柔和模式用于外部 ANC/极静环境，降低降噪力度并补高频
        let mut soft_mode = false;
        let mut last_soft_mode = false;
        // 滞后计数，避免频繁切换
        let mut soft_mode_hold = 0usize;
        const SOFT_MODE_HOLD_FRAMES: usize = 160; // 约 2s（取决于 hop）
        // 启动保护期，前 1s 禁止 gate/瞬态强抑制，避免开口被吞
        // 启动保护期：预热后直接进入正常处理
        let startup_guard_until = Instant::now();
        let mut _auto_gain_scale = 1.0f32;
        has_init.store(true, Ordering::Relaxed);
        log::info!("Worker init");
        let block_duration = df.hop_size as f32 / df.sr as f32;
        let rt60_window_frames = ((0.7 / block_duration).ceil() as usize).max(14);
        let mut rt60_history: VecDeque<f32> = VecDeque::with_capacity(rt60_window_frames);
        let mut proc_time_avg_ms = 0.0f32;
        let mut proc_time_peak_ms = 0.0f32;
        let mut perf_last_log = Instant::now();
        // 连续低峰值检测阈值（约 3 秒）
        let low_peak_required = ((3.0 / block_duration).ceil() as usize).max(30);
        // 预热：在不触碰 IO 的情况下跑几帧静音，暖机模型/重采样状态，避免首句被吞
        if let (Some(inbuf), Some(outbuf)) = (inframe.as_slice_mut(), outframe.as_slice_mut()) {
            inbuf.fill(0.0);
            outbuf.fill(0.0);
            let warmup = 6usize;
            for _ in 0..warmup {
                if highpass_enabled {
                    highpass.process(inbuf);
                }
                // 使用原始数组视图，避免同时持有 &mut 并再借用
                let mut out_view = ArrayViewMut2::from_shape((df.ch, df.hop_size), outbuf)
                    .expect("outframe shape");
                let in_view = ArrayView2::from_shape((df.ch, df.hop_size), inbuf).expect("inframe shape");
                let _ = df.process(in_view, out_view.view_mut());
                if transient_enabled {
                    transient_shaper.process(outbuf);
                }
                if saturation_enabled {
                    saturation.process(outbuf);
                }
                if exciter_enabled {
                    exciter.process(outbuf);
                }
                if agc_enabled {
                    agc.process(outbuf);
                }
                if final_limiter_enabled {
                    apply_final_limiter(outbuf);
                }
                outbuf.fill(0.0);
                inbuf.fill(0.0);
            }
            log::info!("预热完成（{} 帧静音）", warmup);
        }
        let (mut input_resampler, n_in) = if input_sr != df.sr {
            match FftFixedOut::<f32>::new(input_sr, df.sr, df.hop_size, 1, 1) {
                Ok(r) => {
                    let n_in = r.input_frames_max();
                    let buf = r.input_buffer_allocate(true);
                    log::info!(
                        "输入重采样: 设备 {} Hz -> 内部 {} Hz，块长 {}",
                        input_sr,
                        df.sr,
                        n_in
                    );
                    (Some((r, buf)), n_in)
                }
                Err(err) => {
                    log::error!("输入重采样初始化失败: {:?}", err);
                    return;
                }
            }
        } else {
            (None, df.hop_size)
        };
        let (mut output_resampler, n_out) = if output_sr != df.sr {
            match FftFixedIn::<f32>::new(df.sr, output_sr, df.hop_size, 1, 1) {
                Ok(r) => {
                    let n_out = r.output_frames_max();
                    let buf = r.output_buffer_allocate(true);
                    log::info!(
                        "输出重采样: 内部 {} Hz -> 设备 {} Hz，块长 {}",
                        df.sr,
                        output_sr,
                        n_out
                    );
                    (Some((r, buf)), n_out)
                }
                Err(err) => {
                    log::error!("输出重采样初始化失败: {:?}", err);
                    return;
                }
            }
        } else {
            (None, df.hop_size)
        };
        let mut resample_latency_ms = 0.0f32;
        if input_sr != df.sr {
            resample_latency_ms += (n_in as f32 / input_sr as f32) * 1000.0;
        }
        if output_sr != df.sr {
            resample_latency_ms += (n_out as f32 / output_sr as f32) * 1000.0;
        }
        // AEC 初始延迟根据设备/重采样自适应估算，而非写死
        let auto_aec_delay =
            ((block_duration * 1000.0) + resample_latency_ms + 5.0).round().clamp(0.0, 200.0);
        aec_delay_ms = auto_aec_delay as i32;
        aec.set_delay_ms(aec_delay_ms);
        let mut pipeline_delay_ms =
            block_duration * 1000.0 + aec_delay_ms as f32 + resample_latency_ms;
        log::info!(
            "估算链路延迟 {:.2} ms (DF hop {:.2} ms, AEC 延迟 {} ms, 重采样 {:.2} ms)",
            pipeline_delay_ms,
            block_duration * 1000.0,
            aec_delay_ms,
            resample_latency_ms
        );
        // 频谱推送节流计数
        let mut spec_push_counter: usize = 0;
        let mut output_resampler_cleared = output_resampler.is_none();
        let input_retry_delay = Duration::from_micros(100);
        let input_timeout = Duration::from_millis(20);
        let mut auto_sys_volume = false;
        let mut last_sys_adjust =
            Instant::now().checked_sub(Duration::from_secs(10)).unwrap_or_else(Instant::now);
        let mut last_sys_restore = last_sys_adjust;
        let mut clip_counter = 0usize;
        let mut low_peak_counter = 0usize;
        // Align with UI默认：环境自适应默认关闭
        let mut env_auto_enabled = false;
        'processing: while !should_stop.load(Ordering::Relaxed) {
            if rb_in.len() < n_in {
                // 更快轮询，减少初始等待导致的起音丢失
                sleep(Duration::from_millis(1));
                continue;
            }
            let backlog = rb_in.len();
            if _timbre_enabled && backlog.saturating_mul(2) > input_capacity_frames {
                // 输入积压过高时跳过音色修复，先确保音频不中断
                if timbre_overload_frames == 0 {
                    log::warn!(
                        "输入缓冲积压 ({} / {} 帧)，暂时跳过音色修复以避免丢帧",
                        backlog,
                        input_capacity_frames
                    );
                }
                timbre_overload_frames = timbre_overload_frames.max(32);
                timbre_stride = (timbre_stride + 1).min(4);
                timbre_skip_idx = 0;
            } else if timbre_overload_frames > 0 {
                // 在积压缓解后逐步恢复
                if backlog * 4 <= input_capacity_frames {
                    timbre_overload_frames = timbre_overload_frames.saturating_sub(1);
                }
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
                                "等待输入数据超时（需要 {}，已获取 {}），用静音补齐",
                                n_in,
                                filled
                            );
                            for s in buf[0][filled..n_in].iter_mut() {
                                *s = 0.0;
                            }
                            break;
                        }
                        sleep(input_retry_delay);
                        continue;
                    }
                    filled += pulled;
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
                                    "等待输入数据超时（需要 {}，已获取 {}），用静音补齐",
                                    n_in,
                                    filled
                                );
                                for s in buffer[filled..n_in].iter_mut() {
                                    *s = 0.0;
                                }
                                break;
                            }
                            sleep(input_retry_delay);
                            continue;
                        }
                        filled += pulled;
                    }
                } else {
                    log::error!("输入帧内存布局异常，跳过本帧");
                    continue 'processing;
                }
            };
            // 包含输入填充在内的全链路计时
            let frame_start = Instant::now();
            // 录音原始信号（设备采样率或重采样后），在任何处理前
            if let Some(ref rec) = recording {
                if let Some(buffer) = inframe.as_slice() {
                    rec.append_noisy(buffer);
                }
            }

            // 输入增益：软件层前级，避免过载或过低（默认 0 dB）
            if let Some(buffer) = inframe.as_slice_mut() {
                let input_gain = 1.0f32; // 如需调节可做成控制消息
                if input_gain < 0.9999 || input_gain > 1.0001 {
                    for v in buffer.iter_mut() {
                        *v *= input_gain;
                    }
                }
                sanitize_samples("输入信号", buffer);
                if aec_enabled {
                    aec.process_capture(buffer);
                    if !aec.is_active() {
                        log::warn!("AEC3 未激活（检查帧长/初始化），当前旁路");
                        aec_enabled = false;
                    }
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
                lsnr = match df.process(inframe.view(), outframe.view_mut()) {
                    Ok(v) => v,
                    Err(err) => {
                        log::error!("DeepFilterNet 处理失败: {:?}", err);
                        continue;
                    }
                };
                if let Some(buffer) = outframe.as_slice_mut() {
                    sanitize_samples("降噪输出", buffer);
                }
            }
            if env_auto_enabled && !bypass_enabled {
                // 环境噪声特征估计与自适应参数：噪声地板 + SNR 连续映射
                let (rms_db, update_alpha) = if let Some(buf) = inframe.as_slice() {
                    let rms = df::rms(buf.iter());
                    let db = 20.0 * rms.max(1e-9).log10();
                    // 提高自适应平滑系数，缩短响应但避免突变
                    let alpha = if db < -50.0 {
                        0.55
                    } else if db < -30.0 {
                        0.45
                    } else if db < -20.0 {
                        0.35
                    } else {
                        0.25
                    };
                    (db, alpha)
                } else {
                    (-60.0, 0.3)
                };
                // 环境特征节流：每 2 帧计算一次，其余使用缓存
                feature_counter = feature_counter.wrapping_add(1);
                const FEATURE_INTERVAL: usize = 2;
                let feats = if feature_counter % FEATURE_INTERVAL == 0 {
                    let new_feats = compute_noise_features(df.get_spec_noisy());
                    cached_feats = new_feats;
                    new_feats
                } else {
                    cached_feats
                };
                smoothed_energy = smooth_value(smoothed_energy, feats.energy_db, update_alpha);
                smoothed_flatness =
                    smooth_value(smoothed_flatness, feats.spectral_flatness, update_alpha);
                smoothed_centroid =
                    smooth_value(smoothed_centroid, feats.spectral_centroid, update_alpha);

                // Silero VAD 判定（30ms @16k），仅在开启时运行，重采样懒加载
                if vad_enabled {
                    if let Some(buf) = inframe.as_slice() {
                        let cap = vad_source_frame.saturating_mul(3).max(1);
                        // 根据采样率决定是否重采样
                        let mut push_sample = |v: f32| {
                            if vad_buf_raw.len() >= cap {
                                vad_buf_raw.pop_front();
                                vad_drop_count = vad_drop_count.saturating_add(1);
                            }
                            vad_buf_raw.push_back(v);
                        };
                        if df.sr == vad_target_sr {
                            for &v in buf {
                                push_sample(v);
                            }
                        } else {
                            if vad_resampler.is_none() {
                                match FftFixedIn::<f32>::new(df.sr, vad_target_sr, df.hop_size, 1, 1) {
                                    Ok(r) => {
                                        let buf_rs = r.input_buffer_allocate(true);
                                        vad_resampler = Some((r, buf_rs));
                                    }
                                    Err(err) => {
                                        log::warn!("VAD 重采样初始化失败，VAD 将旁路: {}", err);
                                        vad_enabled = false;
                                    }
                                }
                            }
                            if let Some((rs, rs_buf)) = vad_resampler.as_mut() {
                                if rs_buf.len() >= 1 && rs_buf[0].len() >= buf.len() {
                                    rs_buf[0][..buf.len()].copy_from_slice(buf);
                                    match rs.process(rs_buf, None) {
                                        Ok(out) => {
                                            if let Some(out_ch) = out.get(0) {
                                                let samples_to_process =
                                                    out_ch.len().min(vad_source_frame);
                                                for &v in out_ch.iter().take(samples_to_process) {
                                                    push_sample(v);
                                                }
                                                if out_ch.len() > samples_to_process
                                                    && vad_oversample_warn_last.elapsed()
                                                        > Duration::from_secs(5)
                                                {
                                                    log::warn!(
                                                        "VAD 重采样输出过多 ({} > {}), 已截断",
                                                        out_ch.len(),
                                                        samples_to_process
                                                    );
                                                    vad_oversample_warn_last = Instant::now();
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            vad_resample_error_count =
                                                vad_resample_error_count.saturating_add(1);
                                            if vad_resample_error_last_log.elapsed()
                                                > Duration::from_secs(5)
                                            {
                                                log::warn!(
                                                    "VAD 重采样失败: {:?}，该帧数据丢失 (累计 {})",
                                                    err,
                                                    vad_resample_error_count
                                                );
                                                vad_resample_error_last_log = Instant::now();
                                            }
                                            if vad_resample_error_count > 10 {
                                                log::error!(
                                                    "VAD 重采样连续失败 {} 次，禁用 VAD",
                                                    vad_resample_error_count
                                                );
                                                vad_enabled = false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if vad_drop_count > 0 && vad_drop_last_log.elapsed() > Duration::from_secs(5) {
                            log::warn!(
                                "VAD 缓冲溢出，已丢弃 {} 样本（cap={}）。请检查处理负载或提升缓冲。",
                                vad_drop_count,
                                cap
                            );
                            vad_drop_count = 0;
                            vad_drop_last_log = Instant::now();
                        }
                    }
                }
                let guard_active = Instant::now() < startup_guard_until;
                let mut vad_voice = false;
                if vad_enabled && vad_source_frame > 0 && vad_buf_raw.len() >= vad_source_frame {
                    if vad.is_none() {
                        if PathBuf::from(SILERO_VAD_MODEL).exists() {
                            vad = SileroVad::new(SILERO_VAD_MODEL, SileroVadConfig::default()).ok();
                            if vad.is_none() {
                                log::warn!("Silero VAD 初始化失败，VAD 旁路");
                            }
                        } else {
                            log::warn!("Silero VAD 模型缺失: {}", SILERO_VAD_MODEL);
                            vad_enabled = false;
                        }
                    }
                    // 每帧最多处理一帧 VAD，只有填满完整帧才送入模型，复用缓冲避免分配
                    if let Some(ref mut v) = vad {
                        let mut filled = 0usize;
                        for i in 0..vad_source_frame {
                            if let Some(s) = vad_buf_raw.pop_front() {
                                vad_frame_buf[i] = s;
                                filled += 1;
                            } else {
                                break;
                            }
                        }
                        if filled == vad_source_frame {
                            if let Ok(_) = v.process(&vad_frame_buf[..vad_source_frame]) {
                                vad_voice = v.is_speaking();
                            }
                        } else {
                            // 未填满，放回队列等待下一帧
                            for j in (0..filled).rev() {
                                vad_buf_raw.push_front(vad_frame_buf[j]);
                            }
                        }
                    }
                }
                let heuristic_voice = rms_db > -55.0
                    && smoothed_flatness < 0.40
                    && smoothed_centroid < 0.65;
                let energy_gap = rms_db - noise_floor_db;
                let energy_gate = energy_gap > 10.0 && rms_db > -60.0;
                // 语音判定：VAD & 特征 & SNR，同时要求能量高于噪声地板
                let mut is_voice = vad_voice && snr_db > 8.0 && energy_gate;
                let energy_gap_threshold = if guard_active { 8.0 } else { 14.0 };
                if !is_voice && heuristic_voice && energy_gap > energy_gap_threshold {
                    is_voice = true;
                }
                // 启动保护期：强制快速进入语音，避免首音被吞
                if guard_active {
                    if is_voice {
                        vad_state = true;
                        vad_voice_count = 3;
                        vad_noise_count = 0;
                    } else {
                        // 首秒内只要能量略高也视为语音
                        if energy_gap > 6.0 {
                            vad_state = true;
                        }
                    }
                } else {
                    // 滞后：累积计数防抖，语音判定更快，噪声判定更慢
                    if is_voice {
                        vad_voice_count = vad_voice_count.saturating_add(1).min(50);
                        vad_noise_count = vad_noise_count.saturating_sub(vad_noise_count.min(1));
                    } else {
                        vad_noise_count = vad_noise_count.saturating_add(1).min(50);
                        vad_voice_count = vad_voice_count.saturating_sub(vad_voice_count.min(1));
                    }
                    if vad_voice_count >= 1 {
                        vad_state = true;
                    } else if vad_noise_count >= 4 {
                        vad_state = false;
                    }
                }
                is_voice = vad_state;
                // 近讲优先：如果离麦较远（能量差不足），强制当噪声
                if !guard_active && is_voice && energy_gap < 12.0 && rms_db < -42.0 {
                    is_voice = false;
                }

                // 噪声地板跟踪：仅在非语音段更新，下降快，上升慢；启动期更快适配环境
                if !is_voice && rt60_enabled {
                    let nf_fast = if guard_active { 0.7 } else { 0.45 };
                    let nf_slow = if guard_active { 0.10 } else { 0.03 };
                    if rms_db < noise_floor_db {
                        noise_floor_db = smooth_value(noise_floor_db, rms_db, nf_fast);
                    } else {
                        noise_floor_db = smooth_value(noise_floor_db, rms_db, nf_slow);
                    }
                    if rt60_history.len() >= rt60_window_frames {
                        rt60_history.pop_front();
                    }
                    rt60_history.push_back(rms_db);
                    if let Some(rt) = estimate_rt60_from_energy(&rt60_history, block_duration) {
                        // 避免估计异常拉满，强制上限 0.8s
                        smoothed_rt60 = smooth_value(smoothed_rt60, rt.clamp(0.2, 0.8), 0.25);
                    }
                }
                snr_db = (rms_db - noise_floor_db).clamp(-5.0, 30.0);

                // 柔和模式：低能量、低平坦度、低重心
                // 仅在高 SNR、低噪声平坦度时进入柔和模式；低 SNR 或高混响时禁用
                let soft_candidate = smoothed_energy < -55.0
                    && smoothed_flatness < 0.2
                    && smoothed_centroid < 0.35
                    && snr_db > 8.0
                    && smoothed_rt60 < 0.6;
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
                if soft_mode && (snr_db < 12.0 || smoothed_rt60 > 0.6 || !is_voice) {
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
                    if crest > 5.5 && rms > 1e-4 {
                        impact = true;
                    } else {
                        // 高频能量突增（键盘/点击常见），辅助触发
                    let mut hf_energy = 0.0f32;
                    let mut hf_count = 0usize;
                    // 取每隔 4 个样本近似高频
                    for (idx, v) in buf.iter().enumerate() {
                        if idx % 4 == 0 {
                            hf_energy += v * v;
                            hf_count += 1;
                        }
                    }
                    let hf_rms = if hf_count > 0 {
                        (hf_energy / hf_count as f32).sqrt()
                    } else {
                        0.0
                    };
                    if hf_rms > 8e-4 && crest > 3.5 {
                        impact = true;
                    }
                }
                }
                if impact {
                    impact_hold = (impact_hold + 20).min(IMPACT_HOLD_FRAMES);
                } else if impact_hold > 0 {
                    impact_hold = impact_hold.saturating_sub(1);
                }

                // 急促呼吸/衣物摩擦：低重心、低平坦度但能量突增
                if !is_voice
                    && rms_db > noise_floor_db + 8.0
                    && smoothed_centroid < 0.55
                    && smoothed_flatness < 0.45
                {
                    breath_hold = (breath_hold + 12).min(IMPACT_HOLD_FRAMES / 2);
                } else if breath_hold > 0 {
                    breath_hold = breath_hold.saturating_sub(1);
                }

                // 自适应目标初始化（允许保留当前 DF/HP 状态，避免跳变）
                // “办公区(嘈杂) ↔ 会议室(高混响)” 双锚点线性插值
                let office_anchor = SnrParams {
                    atten: 70.0,
                    min_thresh: -8.0,
                    max_thresh: 10.0,
                    hp_cut: 80.0,
                    exciter_mix: 0.08,
                };
                let conf_anchor = SnrParams {
                    atten: 40.0,
                    min_thresh: -18.0,
                    max_thresh: 14.0,
                    hp_cut: 50.0,
                    exciter_mix: 0.15,
                };
                // SNR 低则强行偏向办公区；否则根据 RT60 判断混响权重
                let target_office = if snr_db < 10.0 {
                    1.0
                } else {
                    1.0 - ((smoothed_rt60 - 0.2) / 0.3).clamp(0.0, 1.0)
                };
                office_factor = smooth_value(office_factor, target_office, 0.05);
                target_atten = lerp(conf_anchor.atten, office_anchor.atten, office_factor);
                target_min_thresh =
                    lerp(conf_anchor.min_thresh, office_anchor.min_thresh, office_factor)
                        .max(manual_min_thresh);
                target_max_thresh =
                    lerp(conf_anchor.max_thresh, office_anchor.max_thresh, office_factor)
                        .min(manual_max_thresh);
                target_hp = lerp(conf_anchor.hp_cut, office_anchor.hp_cut, office_factor)
                    .max(manual_highpass);
                target_exciter_mix =
                    lerp(conf_anchor.exciter_mix, office_anchor.exciter_mix, office_factor);
                target_transient_sustain =
                    lerp(-6.0, -2.0, office_factor); // 会议室切尾更狠，办公区轻抑制

                // 档位仅用于日志
                let target_env =
                    classify_env(smoothed_energy, smoothed_flatness, smoothed_centroid);
                if target_env != env_class {
                    log::info!(
                        "环境自适应: {:?} -> {:?} (energy {:.1} dB, flatness {:.3}, centroid {:.3}, SNR {:.1} dB, floor {:.1} dB, RT60 {:.2} s)",
                        env_class, target_env, smoothed_energy, smoothed_flatness, smoothed_centroid, snr_db, noise_floor_db, smoothed_rt60
                    );
                    env_class = target_env;
                }

                // 节流日志，便于观测自适应是否生效（Warn 级别在默认日志下可见）
                if last_env_log.elapsed() >= Duration::from_millis(900) {
                    let impact_note = if impact_hold > 0 {
                        "检测到冲击/点击，已触发强抑制（无 duck 以保护语音起音）"
                    } else {
                        "未检测到冲击"
                    };
                    log::warn!(
                        "自适应: SNR {:.1} dB, RT60 {:.2} s；{}；VAD 语音={}；调整 衰减 {:.1} dB，高通 {:.0} Hz，阈值 {:.1}/{:.1} dB，激励 {:.2}；软模式 {}，冲击保持 {}",
                        snr_db,
                        smoothed_rt60,
                        impact_note,
                        is_voice,
                        target_atten,
                        target_hp,
                        target_min_thresh,
                        target_max_thresh,
                        target_exciter_mix,
                        soft_mode,
                        impact_hold
                    );
                    last_env_log = Instant::now();
                }

                if soft_mode {
                    target_atten = 30.0;
                    target_min_thresh = -58.0;
                    target_max_thresh = 12.0;
                    target_hp = 60.0;
                    target_exciter_mix = 0.2;
                }

                // 启动保护期内使用保守参数，避免首句被重度压制
                if guard_active {
                    target_atten = target_atten.min(40.0);
                    target_min_thresh = target_min_thresh.max(-60.0);
                    target_max_thresh = target_max_thresh.min(12.0);
                    target_hp = target_hp.min(100.0);
                    target_exciter_mix = target_exciter_mix.max(0.08);
                } else {
                    // 语音段放松：降低额外抑制，减小高通，保护起音
                    if is_voice && impact_hold == 0 {
                        target_atten = target_atten.min(48.0);
                        target_min_thresh = target_min_thresh.max(-58.0);
                        target_max_thresh = target_max_thresh.min(10.0);
                        target_hp = target_hp.min(110.0);
                        target_exciter_mix = target_exciter_mix.max(0.05);
                        soft_mode = false;
                        soft_mode_hold = 0;
                    } else if !is_voice {
                        // 非语音段：重度抑制，快速压制键盘/底噪
                        target_atten = (target_atten + 22.0).min(100.0);
                        target_hp = target_hp.max(220.0);
                        target_min_thresh = target_min_thresh.max(-54.0);
                        target_max_thresh = target_max_thresh.min(6.0);
                        target_exciter_mix = 0.0;
                        soft_mode = false;
                        soft_mode_hold = 0;
                    }
                }

                if impact_hold > 0 {
                    // 键盘/点击/关门：瞬时提高抑制和高通，关闭激励
                    target_atten = (target_atten + 12.0).min(75.0);
                    target_hp = target_hp.max(180.0);
                    target_exciter_mix = 0.0;
                    // 冲击时瞬态压制，避免突出
                    transient_shaper.set_attack_gain(-6.0);
                } else if breath_hold > 0 {
                    // 急促呼吸/摩擦：提高抑制和高通，去除高频激励
                    target_atten = (target_atten + 18.0).min(95.0);
                    target_hp = target_hp.max(200.0);
                    target_min_thresh = target_min_thresh.max(-52.0);
                    target_max_thresh = target_max_thresh.min(6.0);
                    target_exciter_mix = 0.0;
                    transient_shaper.set_attack_gain(-4.0);
                    target_transient_sustain = target_transient_sustain.min(-4.0);
                } else {
                    // 恢复用户设定的瞬态增益
                    transient_shaper.set_attack_gain(transient_attack_db);
                }

                // 噪声门控交由 DF/AGC 处理，保持全通
                _gate_gain = 1.0;

                // 全湿，避免干湿并行导致相位/冲击泄露
                _df_mix = 1.0;
                let current_atten = df.atten_lim.unwrap_or(target_atten);
                // 更快的参数平滑，兼顾响应与平顺
                let alpha_fast = 0.5;
                let alpha_hp = 0.15; // 高通调节更平滑，避免可闻跳变
                let new_atten = smooth_value(current_atten, target_atten, alpha_fast);
                df.set_atten_lim(new_atten);
                df.min_db_thresh = smooth_value(df.min_db_thresh, target_min_thresh, alpha_fast);
                df.max_db_df_thresh = smooth_value(df.max_db_df_thresh, target_max_thresh, alpha_fast);
                df.max_db_erb_thresh = smooth_value(df.max_db_erb_thresh, target_max_thresh, alpha_fast);
                // 限制高通截止频率的单步变化，避免可闻跳变
                let hp_target = smooth_value(highpass_cutoff, target_hp, alpha_hp);
                let max_step = 15.0; // Hz per frame
                let delta = (hp_target - highpass_cutoff).clamp(-max_step, max_step);
                highpass_cutoff += delta;
                highpass.set_cutoff(highpass_cutoff);
                // 动态 EQ 全湿，避免干/湿并行带来的相位梳状
                dynamic_eq.set_dry_wet(1.0);
                exciter
                    .set_mix(smooth_value(exciter.mix(), target_exciter_mix, 0.25).clamp(0.0, 0.3));
                transient_shaper.set_sustain_gain(target_transient_sustain);
            }
            // 录音降噪输出（仅 DF + 高通），保证 nc 不受后级 STFT/EQ 影响
            if let Some(ref rec) = recording {
                if let Some(buffer) = outframe.as_slice() {
                    rec.append_denoised(buffer);
                }
            }

            let mut skip_timbre = false;
            if _timbre_enabled && !bypass_enabled && timbre_overload_frames == 0 {
                if timbre_stride > 1 {
                    timbre_skip_idx = (timbre_skip_idx + 1) % timbre_stride;
                    if timbre_skip_idx != 0 {
                        skip_timbre = true;
                    }
                }
                if timbre_restore.is_none() && !timbre_load_failed {
                    if !PathBuf::from(TIMBRE_MODEL).exists() {
                        timbre_load_failed = true;
                        log::warn!("音色修复模型缺失: {}", TIMBRE_MODEL);
                    } else {
                    match TimbreRestore::new(
                        TIMBRE_MODEL,
                        TIMBRE_CONTEXT,
                        TIMBRE_HIDDEN,
                        TIMBRE_LAYERS,
                    ) {
                        Ok(p) => {
                            log::info!("音色修复模型已加载用于实时处理");
                            timbre_restore = Some(p);
                        }
                        Err(err) => {
                            timbre_load_failed = true;
                            log::warn!("音色修复模型加载失败: {}", err);
                        }
                    }
                    }
                }
                if !skip_timbre {
                    if let Some(ref mut tr) = timbre_restore {
                        if let Some(buffer) = outframe.as_slice_mut() {
                            let t0 = Instant::now();
                            if let Err(err) = tr.process_frame(buffer) {
                                log::warn!("音色修复处理失败，已重置状态: {}", err);
                                tr.reset();
                            } else if let Some(ref rec) = recording {
                                rec.append_timbre(buffer);
                            }
                            let elapsed = t0.elapsed().as_secs_f32();
                            if elapsed > block_duration * 0.8 {
                                timbre_overload_frames = timbre_overload_frames.max(16);
                                timbre_stride = (timbre_stride + 1).min(4);
                                timbre_skip_idx = 0;
                                log::warn!(
                                    "音色修复耗时 {:.1} ms，提升节流至每 {} 帧处理一次",
                                    elapsed * 1000.0,
                                    timbre_stride
                                );
                            } else {
                                timbre_last_good = Instant::now();
                            }
                        }
                    }
                }
            } else if _timbre_enabled && timbre_overload_frames > 0 {
                timbre_overload_frames = timbre_overload_frames.saturating_sub(1);
                timbre_skip_idx = 0;
            }
            if _timbre_enabled
                && timbre_stride > 1
                && timbre_overload_frames == 0
                && timbre_last_good.elapsed() > Duration::from_millis(400)
            {
                timbre_stride -= 1;
                timbre_skip_idx = 0;
                log::info!("音色修复恢复至每 {} 帧处理一次", timbre_stride);
                timbre_last_good = Instant::now();
            }

            // STFT 静态 EQ 已移除，不再处理
            let bypass_post = bypass_enabled
                || (!dynamic_eq.is_enabled()
                    && !transient_enabled
                    && !saturation_enabled
                    && !agc_enabled);
            let eq_start = Instant::now();
            let (eq_gain_db, eq_enabled_flag) = if bypass_post {
                ([0.0; MAX_EQ_BANDS], false)
            } else {
                // 瞬态塑形放在动态 EQ 前，保护起音
                if transient_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        transient_shaper.process(buffer);
                    }
                }
                // 动态 EQ
                let metrics = if let Some(buffer) = outframe.as_slice_mut() {
                    dynamic_eq.set_dry_wet(1.0);
                    dynamic_eq.process_block(buffer)
                } else {
                    log::error!("输出帧内存布局异常，跳过动态 EQ");
                    EqProcessMetrics::default()
                };
                // 饱和/谐波（放在 AGC 前）
                if saturation_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        saturation.process(buffer);
                    }
                }
                if exciter_enabled {
                    if let Some(buffer) = outframe.as_slice_mut() {
                        exciter.process(buffer);
                    }
                }
                // 输出增益在 AGC 前统一设置，然后交给 AGC 控制最终电平
                if let Some(buffer) = outframe.as_slice_mut() {
                    let mut out_gain = post_trim_gain * headroom_gain;
                    if out_gain > 1.0 {
                        log::warn!(
                            "输出增益 {:.2} 超过 0 dB，已限制为 1.0（请调低 Post-trim 或 Headroom）",
                            out_gain
                        );
                        out_gain = 1.0;
                    }
                    if (out_gain - 1.0).abs() > 1e-6 {
                        for v in buffer.iter_mut() {
                            *v *= out_gain;
                        }
                    }
                    if agc_enabled {
                        agc.process(buffer);
                    }
                } else {
                    log::error!("输出帧内存布局异常，跳过后级处理");
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
            // 启动淡入，避免播放砰声；20ms 指数淡入，保护首音
            if fade_progress < fade_total {
                if let Some(buffer) = outframe.as_slice_mut() {
                    for v in buffer.iter_mut() {
                        if fade_progress >= fade_total {
                            break;
                        }
                        // 指数淡入，前几帧仍保留可听能量
                        let fp = fade_progress as f32;
                        let g = 1.0 - (-(fp) / (fade_total as f32 / 3.0)).exp().min(1.0);
                        *v *= g;
                        fade_progress += 1;
                    }
                }
            }
            // gate 已禁用
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
            if let Some(buffer) = outframe.as_slice_mut() {
                // 一次遍历检测异常峰值并记录峰值
                let mut peak = 0.0f32;
                for v in buffer.iter() {
                    peak = peak.max(v.abs());
                }
                if peak > 2.0 {
                    log::warn!(
                        "检测到异常峰值 {:.2}，将限幅保护（可能某处理节点异常增益）",
                        peak
                    );
                    for v in buffer.iter_mut() {
                        *v = v.clamp(-1.2, 1.2);
                    }
                    peak = 1.2;
                }
                if aec_enabled {
                    aec.process_render(buffer);
                    if !aec.is_active() {
                        log::warn!("AEC3 未激活（检查帧长/初始化），当前旁路");
                        aec_enabled = false;
                    }
                }
                // 最终限幅一次，避免多级限幅导致音色压缩
                if final_limiter_enabled {
                    apply_final_limiter(buffer);
                }
                if peak > 0.99 && perf_last_log.elapsed() > Duration::from_secs(2) {
                    log::warn!("输出峰值 {:.3}，接近裁剪，请下调增益/饱和/激励", peak);
                }
                // 处理耗时监测
                let elapsed_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
                let smooth = 0.08f32;
                proc_time_avg_ms = proc_time_avg_ms * (1.0 - smooth) + elapsed_ms * smooth;
                proc_time_peak_ms = proc_time_peak_ms.max(elapsed_ms);
                // 预算=DF hop + 重采样延迟（设备与模型采样率不一致时），下限 30ms 以适配 24k↔16k
                let budget_ms = (block_duration * 1000.0 + resample_latency_ms).max(30.0);
                if perf_last_log.elapsed() > Duration::from_secs(5) {
                    log::info!(
                        "帧耗时 avg/peak {:.2}/{:.2} ms（预算 {:.2} ms，重采样 {:.2} ms）",
                        proc_time_avg_ms,
                        proc_time_peak_ms,
                        budget_ms,
                        resample_latency_ms
                    );
                    proc_time_peak_ms *= 0.5; // 简单衰减记录
                    perf_last_log = Instant::now();
                }
                // 留出 50% 容错，避免设备采样率不可调导致的常驻告警
                if elapsed_ms > budget_ms * 1.5 && perf_last_log.elapsed() > Duration::from_millis(500) {
                    log::warn!(
                        "单帧耗时 {:.2} ms 超预算 {:.2} ms，可能导致掉帧",
                        elapsed_ms,
                        budget_ms
                    );
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
                    if !output_resampler_cleared {
                        for frame in buf.iter_mut() {
                            for sample in frame.iter_mut() {
                                *sample = 0.0;
                            }
                        }
                        output_resampler_cleared = true;
                    }
                    if let Some(slice) = outframe.as_slice() {
                        if let Err(err) = r.process_into_buffer(&[slice], buf, None) {
                            log::error!("输出重采样失败: {:?}", err);
                        } else {
                            push_output_block(&should_stop, &mut rb_out, &buf[0][..n_out], n_out);
                        }
                    } else {
                        log::error!("输出帧内存布局异常，跳过输出");
                    }
                } else if let Some(buf) = outframe.as_slice() {
                    push_output_block(&should_stop, &mut rb_out, &buf[..n_out], n_out);
                } else {
                    log::error!("输出帧内存布局异常，跳过输出");
                }
            }
            if let Some(sender) = s_lsnr.as_ref() {
                if let Err(err) = sender.send(lsnr) {
                    log::warn!("Failed to send LSNR value: {}", err);
                }
            }
            // 频谱推送节流：默认每 3 帧一次
            spec_push_counter = spec_push_counter.wrapping_add(1);
            const SPEC_PUSH_INTERVAL: usize = 3;
            if spec_enabled && spec_push_counter % SPEC_PUSH_INTERVAL == 0 {
                if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                    push_spec(df.get_spec_noisy(), s_noisy);
                    push_spec(df.get_spec_enh(), s_enh);
                }
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok(message) = r_opt.try_recv() {
                    match message {
                        ControlMessage::DeepFilter(control, value) => match control {
                            DfControl::AttenLim => {
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
                            EqControl::SetBandGain(idx, gain) => dynamic_eq.set_band_gain(idx, gain),
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
                            EqControl::SetBandMode(idx, mode) => dynamic_eq.set_band_mode(idx, mode),
                            EqControl::SetBandFilter(idx, filter) => {
                                dynamic_eq.set_band_filter(idx, filter)
                            }
                        },
                        ControlMessage::MutePlayback(muted) => {
                            mute_playback = muted;
                            log::info!("播放静音: {}", if mute_playback { "开启" } else { "关闭" });
                        }
                        ControlMessage::BypassEnabled(enabled) => {
                            bypass_enabled = enabled;
                            log::info!("旁路: {}", if enabled { "开启" } else { "关闭" });
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
                        ControlMessage::VadEnabled(enabled) => {
                            vad_enabled = enabled;
                            log::warn!("Silero VAD: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::TransientGain(db) => {
                            transient_attack_db = db;
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
                        ControlMessage::AecEnabled(enabled) => {
                            aec_enabled = enabled;
                            aec.set_enabled(enabled);
                            log::info!("AEC3: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::AecAggressive(enabled) => {
                            _aec_aggressive = enabled;
                            aec.set_aggressive(enabled);
                            log::info!("AEC3 强力模式: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::AecDelayMs(v) => {
                            aec_delay_ms = v;
                            aec.set_delay_ms(v);
                            pipeline_delay_ms =
                                block_duration * 1000.0 + aec_delay_ms as f32 + resample_latency_ms;
                            log::info!(
                                "AEC3 延迟补偿: {} ms，总链路估算 {:.2} ms",
                                v,
                                pipeline_delay_ms
                            );
                        }
                        ControlMessage::SysAutoVolumeEnabled(_) => {
                            auto_sys_volume = false;
                            log::warn!("系统音量保护已禁用（使用 WebRTC AGC），忽略 UI 设置");
                        }
                        ControlMessage::EnvAutoEnabled(enabled) => {
                            env_auto_enabled = enabled;
                            log::warn!("环境自适应: {}", if enabled { "开启" } else { "关闭" });
                            if !enabled {
                                soft_mode = false;
                                last_soft_mode = false;
                                smoothed_rt60 = 0.35;
                                rt60_history.clear();
                                if let Some(ref sender) = s_env_status {
                                    let _ = sender.try_send(EnvStatus::Normal);
                                }
                            }
                        }
                        ControlMessage::SpecEnabled(enabled) => {
                            spec_enabled = enabled;
                            log::info!("频谱/RT60 推送: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::Rt60Enabled(enabled) => {
                            rt60_enabled = enabled;
                            if !enabled {
                                rt60_history.clear();
                            }
                            log::info!("RT60 估计: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::FinalLimiterEnabled(enabled) => {
                            final_limiter_enabled = enabled;
                            log::info!("最终限幅器: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::ExciterEnabled(enabled) => {
                            exciter_enabled = enabled;
                            log::info!("谐波激励: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::ExciterMix(value) => {
                            exciter.set_mix(value.clamp(0.0, 0.5));
                            log::info!("谐波激励混合: {:.0}%", value * 100.0);
                        }
                        ControlMessage::TimbreEnabled(enabled) => {
                            _timbre_enabled = enabled;
                            if enabled {
                                timbre_load_failed = false;
                                if let Some(ref mut tr) = timbre_restore {
                                    tr.reset();
                                }
                            } else if let Some(ref mut tr) = timbre_restore {
                                tr.reset();
                            }
                            log::info!("音色修复: {}", if enabled { "开启" } else { "关闭" });
                        }
                    }
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
                        ControlMessage::VadEnabled(enabled) => {
                            vad_enabled = enabled;
                            log::warn!("Silero VAD: {}", if enabled { "开启" } else { "关闭" });
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
                        ControlMessage::AecEnabled(enabled) => {
                            aec_enabled = enabled;
                            aec.set_enabled(enabled);
                            log::info!("AEC3: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::AecAggressive(enabled) => {
                            _aec_aggressive = enabled;
                            aec.set_aggressive(enabled);
                            log::info!("AEC3 强力模式: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::AecDelayMs(v) => {
                            aec_delay_ms = v;
                            aec.set_delay_ms(v);
                            pipeline_delay_ms =
                                block_duration * 1000.0 + aec_delay_ms as f32 + resample_latency_ms;
                            log::info!(
                                "AEC3 延迟补偿: {} ms，总链路估算 {:.2} ms",
                                v,
                                pipeline_delay_ms
                            );
                        }
                        ControlMessage::SysAutoVolumeEnabled(_) => {
                            auto_sys_volume = false;
                            log::warn!("系统音量保护已禁用（使用 WebRTC AGC），忽略 UI 设置");
                        }
                        ControlMessage::EnvAutoEnabled(enabled) => {
                            env_auto_enabled = enabled;
                            log::warn!("环境自适应: {}", if enabled { "开启" } else { "关闭" });
                            if !enabled {
                                soft_mode = false;
                                last_soft_mode = false;
                                smoothed_rt60 = 0.35;
                                rt60_history.clear();
                                if let Some(ref sender) = s_env_status {
                                    let _ = sender.try_send(EnvStatus::Normal);
                                }
                            }
                        }
                        ControlMessage::SpecEnabled(enabled) => {
                            spec_enabled = enabled;
                            log::info!("频谱/RT60 推送: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::Rt60Enabled(enabled) => {
                            rt60_enabled = enabled;
                            if !enabled {
                                rt60_history.clear();
                            }
                            log::info!("RT60 估计: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::FinalLimiterEnabled(enabled) => {
                            final_limiter_enabled = enabled;
                            log::info!("最终限幅器: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::ExciterEnabled(enabled) => {
                            exciter_enabled = enabled;
                            log::info!("谐波激励: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::ExciterMix(value) => {
                            exciter.set_mix(value.clamp(0.0, 0.5));
                            log::info!("谐波激励混合: {:.0}%", value * 100.0);
                        }
                        ControlMessage::BypassEnabled(enabled) => {
                            bypass_enabled = enabled;
                            log::info!("全链路旁路: {}", if enabled { "开启" } else { "关闭" });
                        }
                        ControlMessage::TimbreEnabled(enabled) => {
                            _timbre_enabled = enabled;
                            if enabled {
                                timbre_load_failed = false;
                                if let Some(ref mut tr) = timbre_restore {
                                    tr.reset();
                                }
                            } else if let Some(ref mut tr) = timbre_restore {
                                tr.reset();
                            }
                            log::info!("音色修复: {}", if enabled { "开启" } else { "关闭" });
                        }
                    }
                }
            }
        }
    }
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    thread_local! {
        static SPEC_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }
    SPEC_BUF.with(|buf_cell| {
        let mut buf = buf_cell.borrow_mut();
        let needed = spec.len();
        if buf.len() < needed {
            buf.resize(needed, 0.0);
        }
        for (dst, src) in buf.iter_mut().zip(spec.iter()) {
            *dst = src.norm_sqr().max(1e-10).log10() * 10.0;
        }
        // 仍需为发送拥有所有权，保留一次拷贝
        let out = buf[..needed].to_vec().into_boxed_slice();
        if let Err(err) = sender.send(out) {
            log::warn!("Failed to send spectrogram data: {}", err);
        }
    });
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    // 简单逐样本限幅，避免整帧缩放带来的“忽高忽低”抽吸
    const CEILING: f32 = 0.98;
    for sample in data.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        } else if *sample > CEILING {
            *sample = CEILING;
        } else if *sample < -CEILING {
            *sample = -CEILING;
        }
    }
}

/// 简单峰值防护，避免链路后级触顶。
pub struct SysVolMonitorHandle {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SysVolMonitorHandle {
    #[allow(dead_code)]
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
#[allow(dead_code)]
pub fn start_sys_volume_monitor(_selected_input: Option<String>) -> Option<SysVolMonitorHandle> {
    log::info!("自动系统音量后台监测仅在 macOS 可用");
    None
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
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

fn sanitize_samples(tag: &str, samples: &mut [f32]) -> bool {
    const DENORMAL_THRESHOLD: f32 = 1e-38;
    let mut dirty = false;
    for s in samples.iter_mut() {
        if !s.is_finite() || s.abs() < DENORMAL_THRESHOLD {
            *s = 0.0;
            dirty = true;
        }
    }
    if dirty {
        log::debug!("{tag} 检测到非法/次正规样本，已归零防止性能下降");
    }
    dirty
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

fn estimate_rt60_from_energy(history: &VecDeque<f32>, block_duration: f32) -> Option<f32> {
    let len = history.len();
    if len < 10 {
        return None;
    }
    let duration = block_duration * (len.saturating_sub(1) as f32);
    if duration < 0.2 {
        return None;
    }
    let head = (len as f32 * 0.35).max(4.0) as usize;
    let tail = (len as f32 * 0.25).max(3.0) as usize;
    let start_mean = history.iter().take(head).sum::<f32>() / head as f32;
    let end_mean = history.iter().rev().take(tail).sum::<f32>() / tail as f32;
    let decay_db = start_mean - end_mean;
    if decay_db < 6.0 {
        return None;
    }
    let slope = (end_mean - start_mean) / duration; // dB/s，衰减应为负值
    if slope >= -10.0 || slope.abs() < 1e-6 {
        return None;
    }
    let rt60 = (-60.0 / slope).clamp(0.2, 1.2);
    Some(rt60)
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
    hp_cut: f32,
    exciter_mix: f32,
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
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
        if sr != PROCESS_SR || frame_size != PROCESS_HOP {
            return Err(anyhow!(
                "模型采样率/帧长为 {} Hz / {}，但内部处理要求 48 kHz / {}，请使用 48k 模型以保证仅在 IO 边界重采样",
                sr,
                frame_size,
                PROCESS_HOP
            ));
        }
        let freq_size = metadata.freq_size;
        let input_capacity_frames = frame_size * 800; // 扩大输入缓冲，容忍瞬时负载
        let in_rb = HeapRb::<f32>::new(input_capacity_frames);
        // 扩大输出缓冲，容忍处理抖动，避免欠载导致的音量波动
        let out_rb = HeapRb::<f32>::new(frame_size * 800);
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
        // 打印实际设备采样率，方便确认是否需要边界重采样
        let in_name = source.device.name().unwrap_or_else(|_| "unknown input".into());
        let out_name = sink.device.name().unwrap_or_else(|_| "unknown output".into());
        log::info!(
            "Input device '{}' @ {} Hz (internal processing {} Hz)",
            in_name,
            source.sr(),
            PROCESS_SR
        );
        log::info!(
            "Output device '{}' @ {} Hz (internal processing {} Hz)",
            out_name,
            sink.sr(),
            PROCESS_SR
        );
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
            input_capacity_frames,
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

    #[allow(dead_code)]
    pub fn recording(&self) -> RecordingHandle {
        self.recording.clone()
    }

    #[allow(dead_code)]
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
