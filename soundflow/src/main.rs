//! Command‑line entry point for the soundflow pipeline.
//!
//! This binary reads a mono WAV file, processes it block by block through the
//! adaptive noise reduction pipeline, and writes the result to an output WAV
//! file. The processing includes DeepFilterNet inference, environment
//! classification, high‑pass filtering, automatic gain control and limiting.
//!
//! Usage:
//!
//! ```
//! cargo run --release -- <input.wav> <output.wav> [model.df]
//! ```
//!
//! If `model.df` is omitted, the default DeepFilterNet model embedded in the
//! `df` crate is used.

use std::env;
use std::path::Path;

use hound;
use soundflow::processor::Processor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize env_logger for optional logging. If the user sets
    // `RUST_LOG=info`, the processor can emit informational messages.
    let _ = env_logger::try_init();
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input.wav> <output.wav> [model.df]\n\nThis tool processes a mono 48 kHz WAV file with DeepFilterNet and writes the output.",
            args.get(0).unwrap_or(&"soundflow".to_string())
        );
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let model_path_opt = args.get(3).map(|s| Path::new(s).as_ref());
    // Create the processor with an optional model path.
    let mut processor = Processor::new(model_path_opt)?;
    let hop = processor.hop_size();
    let sample_rate = processor.sample_rate() as u32;
    // Open input WAV file. The file must be mono and match the processor sample rate.
    let mut reader = hound::WavReader::open(input_path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err("Input WAV must be mono".into());
    }
    if spec.sample_rate != sample_rate {
        return Err(format!(
            "Input sample rate {} does not match model sample rate {}",
            spec.sample_rate, sample_rate
        )
        .into());
    }
    // Read samples as f32 in the range [-1, 1]. Support 16‑bit and 24‑bit integers.
    let mut samples = Vec::new();
    match spec.bits_per_sample {
        16 => {
            for s in reader.samples::<i16>() {
                let v = s? as f32 / i16::MAX as f32;
                samples.push(v);
            }
        }
        24 | 32 => {
            for s in reader.samples::<i32>() {
                let v = s? as f32 / i32::MAX as f32;
                samples.push(v);
            }
        }
        _ => {
            return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into());
        }
    }
    // Process in blocks of hop size. If the input length is not a multiple of hop,
    // the remaining samples are appended unprocessed.
    let mut processed = Vec::with_capacity(samples.len());
    let mut idx = 0;
    while idx + hop <= samples.len() {
        let block = &samples[idx..idx + hop];
        let out_block = processor.process_block(block);
        processed.extend(out_block);
        idx += hop;
    }
    // Append any leftover samples without processing to preserve alignment.
    processed.extend_from_slice(&samples[idx..]);
    // Write output WAV as 16‑bit integer PCM.
    let spec_out = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec_out)?;
    for v in processed {
        let val = (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(val)?;
    }
    writer.finalize()?;
    Ok(())
}