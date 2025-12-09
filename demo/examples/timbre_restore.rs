//! Rust 集成示例
//! 
//! 将训练好的 ONNX 模型集成到 DeepFilter 项目中
//! 
//! 添加到 demo/Cargo.toml:
//! ```toml
//! [dependencies]
//! ort = { version = "2.0", features = ["load-dynamic"] }
//! ```

use std::path::Path;
use std::sync::Arc;

// 使用 ort (ONNX Runtime)
use ort::{Environment, Session, SessionBuilder, Value, GraphOptimizationLevel};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Array1, IxDyn};

/// 音色修复处理器
pub struct TimbreRestore {
    session: Session,
    frame_size: usize,
    sample_rate: usize,
    // 用于流式处理的缓冲（卷积上下文）
    context_size: usize,
    context_buffer: Vec<f32>,
    // RNN 隐状态 [num_layers, 1, hidden_size]
    hidden_size: usize,
    num_layers: usize,
    hidden: Vec<f32>,
}

impl TimbreRestore {
    /// 创建新的音色修复处理器
    /// 
    /// # Arguments
    /// * `model_path` - ONNX 模型路径
    /// * `frame_size` - 处理帧大小 (建议与 DF 一致，如 480)
    /// * `sample_rate` - 采样率 (48000)
    /// * `hidden_size` - GRU 隐状态维度（需与训练配置一致，Balanced: 384）
    /// * `num_layers` - GRU 层数（需与训练配置一致，Balanced: 2）
    /// * `context_size` - 卷积因果上下文长度（采样点，默认 256）
    pub fn new(
        model_path: impl AsRef<Path>,
        frame_size: usize,
        sample_rate: usize,
        hidden_size: usize,
        num_layers: usize,
        context_size: usize,
    ) -> anyhow::Result<Self> {
        // 初始化 ONNX Runtime 环境
        let environment = Environment::builder()
            .with_name("timbre_restore")
            .build()?
            .into_arc();
        
        // 创建 Session
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // 单线程，避免与 DF 竞争
            .with_model_from_file(model_path)?;
        
        Ok(Self {
            session,
            frame_size,
            sample_rate,
            context_size,
            context_buffer: vec![0.0f32; context_size],
            hidden_size,
            num_layers,
            hidden: vec![0.0f32; num_layers * 1 * hidden_size],
        })
    }
    
    /// 处理单帧音频（实时流式处理）
    /// 
    /// # Arguments
    /// * `frame` - 输入帧 (原地修改)
    pub fn process_frame(&mut self, frame: &mut [f32]) -> anyhow::Result<()> {
        let frame_len = frame.len();
        // 构建带上下文的输入: [context + frame]
        let mut input_full = Vec::with_capacity(self.context_size + frame_len);
        input_full.extend_from_slice(&self.context_buffer);
        input_full.extend_from_slice(frame);

        // ONNX 输入: audio [1, 1, T_full], h_in [num_layers, 1, hidden]
        // 保存输入尾部作为下一帧的卷积上下文（使用模型输入而非模型输出）
        let tail_start = input_full.len().saturating_sub(self.context_size);
        let context_tail = if self.context_size > 0 {
            input_full[tail_start..].to_vec()
        } else {
            Vec::new()
        };
        let input_array = Array2::from_shape_vec((1, input_full.len()), input_full)?;
        let h_shape = (self.num_layers, 1, self.hidden_size);
        let h_in = ndarray::Array::from_shape_vec(h_shape, self.hidden.clone())?;

        let input_value = Value::from_array(input_array.view().insert_axis(ndarray::Axis(1)))?;
        let h_value = Value::from_array(h_in)?;

        let outputs = self.session.run(vec![input_value, h_value])?;

        // 提取输出
        let output: ndarray::ArrayViewD<f32> = outputs[0].try_extract_tensor()?;
        // h_out
        let h_out: ndarray::ArrayViewD<f32> = outputs[1].try_extract_tensor()?;
        // 更新隐藏状态
        self.hidden = h_out.to_owned().iter().cloned().collect();

        // 只取输出最后 frame_len 部分
        let total_len = output.len();
        let start = total_len.saturating_sub(frame_len);
        for (i, &v) in output.iter().skip(start).enumerate() {
            frame[i] = v;
        }

        // 更新上下文缓冲（使用模型输入尾部，保证因果性）
        if self.context_size > 0 && context_tail.len() == self.context_size {
            self.context_buffer.copy_from_slice(&context_tail);
        }
        
        Ok(())
    }
    
    /// 批量处理音频
    /// 
    /// # Arguments
    /// * `audio` - 完整音频 (原地修改)
    pub fn process_batch(&mut self, audio: &mut [f32]) -> anyhow::Result<()> {
        let audio_len = audio.len();
        
        // 重置状态与上下文
        self.hidden.fill(0.0);
        self.context_buffer.fill(0.0);

        // 逐帧处理，保证与流式一致
        let mut offset = 0;
        while offset < audio_len {
            let end = (offset + self.frame_size).min(audio_len);
            let mut frame = audio[offset..end].to_vec();
            self.process_frame(&mut frame)?;
            audio[offset..end].copy_from_slice(&frame);
            offset = end;
        }
        
        Ok(())
    }
}

// ========================================
// 集成到 capture.rs 的示例
// ========================================

/*
在 capture.rs 中的集成方式:

1. 在 DeepFilterCapture 中添加:
   ```rust
   let mut timbre_restore = TimbreRestore::new(
       "timbre_restore.onnx",
       frame_size,
       sr,
       384,      // hidden_size (Balanced 模型)
       2,        // num_layers (Balanced 模型)
       256,      // context_size (卷积感受野余量)
   ).ok();
   ```

2. 在处理循环中，DF 输出后添加:
   ```rust
   // DeepFilterNet 处理
   lsnr = df.process(inframe.view(), outframe.view_mut())?;
   
   // 音色修复
   if let Some(ref mut tr) = timbre_restore {
       if let Some(buf) = outframe.as_slice_mut() {
           if let Err(e) = tr.process_frame(buf) {
               log::warn!("Timbre restore failed: {}", e);
           }
       }
   }
   
   // 后续处理 (EQ, AGC 等)
   ```

3. 确保 ONNX 模型路径正确配置
*/

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timbre_restore() {
        // 测试（需要模型文件）
        if !Path::new("timbre_restore.onnx").exists() {
            println!("Skipping test: model not found");
            return;
        }
        
        let mut tr = TimbreRestore::new("timbre_restore.onnx", 480, 48000, 384, 2, 256).unwrap();
        
        // 测试帧处理
        let mut frame = vec![0.0f32; 480];
        for i in 0..480 {
            frame[i] = (i as f32 / 480.0 * std::f32::consts::PI * 4.0).sin() * 0.5;
        }
        
        tr.process_frame(&mut frame).unwrap();
        
        // 验证输出在合理范围
        assert!(frame.iter().all(|&x| x.abs() <= 1.0));
    }
}
