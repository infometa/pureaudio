use anyhow::Result;
use log::error;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;

/// 基于 ONNX Runtime 的流式音色修复（使用 `download-binaries` 自动获取 runtime，无需手动 dylib）。
pub struct TimbreRestore {
    session: Session,
    context_size: usize,
    context_buffer: Vec<f32>,
    hidden: Vec<f32>,
    hidden_size: usize,
    num_layers: usize,
}

impl TimbreRestore {
    pub fn new(
        model_path: impl AsRef<Path>,
        context_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            context_size,
            context_buffer: vec![0.0f32; context_size],
            hidden: vec![0.0f32; hidden_size * num_layers],
            hidden_size,
            num_layers,
        })
    }

    pub fn process_frame(&mut self, frame: &mut [f32]) -> Result<()> {
        let frame_len = frame.len();
        // 输入：上下文 + 当前帧
        let mut input_full = Vec::with_capacity(self.context_size + frame_len);
        input_full.extend_from_slice(&self.context_buffer);
        input_full.extend_from_slice(frame);

        // 模型只有 1 个输入：[1, 1, T]
        let input_value =
            Tensor::from_array((vec![1usize, 1, input_full.len()], input_full.clone()))?;
        let h_shape = vec![self.num_layers, 1usize, self.hidden_size];
        let h_in = Tensor::from_array((h_shape, self.hidden.clone()))?;

        let outputs = self.session.run(ort::inputs![input_value, h_in])?;

        let (_, output) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, h_out) = outputs[1].try_extract_tensor::<f32>()?;
        self.hidden = h_out.to_vec();

        // 仅取最后 frame_len 部分作为当前帧输出
        let total_len = output.len();
        let start = total_len.saturating_sub(frame_len);
        for (i, &v) in output.iter().skip(start).enumerate() {
            if i < frame.len() {
                frame[i] = v;
            }
        }

        // 更新上下文缓冲
        if self.context_size > 0 {
            let keep_start = input_full.len().saturating_sub(self.context_size);
            self.context_buffer.copy_from_slice(&input_full[keep_start..]);
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.context_buffer.fill(0.0);
        self.hidden.fill(0.0);
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
