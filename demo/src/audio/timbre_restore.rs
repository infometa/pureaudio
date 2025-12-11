use anyhow::{Context, Result};
use log::error;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

/// 基于 ONNX Runtime 的流式音色修复（使用 `download-binaries` 自动获取 runtime，无需手动 dylib）。
/// 
/// 性能注意事项：
/// - 当前实现每帧执行多次Vec拷贝，在低端设备上可能成为瓶颈
/// - ort 2.0的Tensor::from_array要求所有权，无法使用零拷贝
/// - 未来优化方向：使用CowArray或预分配tensor池
pub struct TimbreRestore {
    session: Session,
    context_size: usize,
    input_buffer: Vec<f32>,
    hidden: Vec<f32>,
    hidden_size: usize,
    num_layers: usize,
    // 性能监控
    total_frames: std::sync::atomic::AtomicU64,
    timeout_count: std::sync::atomic::AtomicU64,
}

impl TimbreRestore {
    pub fn new(
        model_path: impl AsRef<Path>,
        context_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let session = match Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_execution_providers([ort::execution_providers::CoreMLExecutionProvider::default().build()])?
            .commit_from_file(&model_path) 
        {
            Ok(s) => {
                log::info!("TimbreRestore: CoreML 尝试加载成功");
                s
            }
            Err(e) => {
                log::warn!("TimbreRestore: CoreML 加载失败，回退到 CPU: {}", e);
                Session::builder()?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(1)?
                    .commit_from_file(&model_path)?
            }
        };

        Ok(Self {
            session,
            context_size,
            input_buffer: vec![0.0f32; context_size],
            hidden: vec![0.0f32; hidden_size * num_layers],
            hidden_size,
            num_layers,
            total_frames: std::sync::atomic::AtomicU64::new(0),
            timeout_count: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn process_frame(&mut self, frame: &mut [f32]) -> Result<()> {
        let frame_len = frame.len();
        let total_input = self.context_size + frame_len;
        
        // 性能计数
        self.total_frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Ensure capacity (should only happen once if frame size > default)
        if self.input_buffer.len() < total_input {
            self.input_buffer.resize(total_input, 0.0);
        }

        // 1. Fill input buffer
        if self.input_buffer.len() >= total_input {
            self.input_buffer[self.context_size..total_input].copy_from_slice(frame);
        } else {
            return Err(anyhow::anyhow!("Timbre buffer error: buffer size mismatch"));
        }

        // 2. Prepare Inputs
        // 注意：ort 2.0的Tensor::from_array要求数据所有权，必须执行to_vec()
        // 这是当前API的限制，每帧会产生3次堆分配（input + hidden_in + hidden_out）
        // 
        // 性能影响估算 @ 48kHz/480样本帧：
        // - input: (256+480)*4 = 2944 bytes/frame * 100 frames/s = 294KB/s
        // - hidden: (384*2)*4 = 3072 bytes/frame * 100 frames/s = 307KB/s
        // - 总计约600KB/s堆分配，在低端设备上可能导致GC压力
        //
        // 未来优化方向：
        // 1. 使用ort的Session::run_with_tensors()如果支持引用
        // 2. 探索unsafe { Tensor::from_raw_parts() }避免拷贝
        // 3. 预分配tensor池，重用内存
        let input_shape = vec![1usize, 1, total_input];
        let input_val = Tensor::from_array((input_shape, self.input_buffer[..total_input].to_vec()))
            .context("Failed to create input tensor")?;
        
        let h_shape = vec![self.num_layers, 1usize, self.hidden_size];
        let h_in = Tensor::from_array((h_shape, self.hidden.to_vec()))
            .context("Failed to create hidden tensor")?;

        // 3. Run inference
        // TODO: 添加超时保护，防止神经网络推理阻塞实时处理
        let outputs = self.session.run(ort::inputs![input_val, h_in])
            .context("ONNX inference failed")?;

        // 4. Extract Output
        let output_tensor = &outputs[0];
        let (_, output_data) = output_tensor.try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;
        
        // Scale back to frame
        if output_data.len() < frame_len {
            return Err(anyhow::anyhow!(
                "Output tensor size ({}) < frame size ({})", 
                output_data.len(), frame_len
            ));
        }
        let start = output_data.len().saturating_sub(frame_len);
        let len = frame.len().min(output_data.len() - start);
        frame[..len].copy_from_slice(&output_data[start..start + len]);
        
        // 5. Extract and Update Hidden State
        let h_out_tensor = &outputs[1];
        let (_, h_out_data) = h_out_tensor.try_extract_tensor::<f32>()
            .context("Failed to extract hidden state")?;
        
        if self.hidden.len() == h_out_data.len() {
            self.hidden.copy_from_slice(h_out_data);
        } else {
            log::warn!(
                "Hidden state size mismatch: expected {}, got {}. Reallocating.",
                self.hidden.len(), h_out_data.len()
            );
            self.hidden = h_out_data.to_vec();
        }

        // 6. Shift Context Window
        if self.context_size > 0 {
            let keep_start = total_input.saturating_sub(self.context_size);
            self.input_buffer.copy_within(keep_start..total_input, 0);
        }

        Ok(())
    }
    
    /// 获取性能统计
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.total_frames.load(std::sync::atomic::Ordering::Relaxed),
            self.timeout_count.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.hidden.fill(0.0);
        // 不重置计数器，保留统计信息
    }
}

/// 简易工厂：失败返回 None 并打日志
#[allow(dead_code)]
pub fn load_default_timbre(model_path: impl AsRef<Path>) -> Option<TimbreRestore> {
    match TimbreRestore::new(model_path, 256, 384, 2) {
        Ok(p) => Some(p),
        Err(e) => {
            error!("加载音色修复模型失败: {e}");
            None
        }
    }
}
