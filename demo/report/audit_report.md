# 音频代码深度审计与实施报告

## 1. 概述
本次审计针对 `demo` 目录下的音频处理核心代码（包括 `capture.rs`, `aec.rs` 及 `audio/` 子模块）。重新审计确认：项目存在严重的**实时性安全（Real-time Safety）**违规，即在音频回调热路径中进行了**动态内存分配、锁竞争和阻塞式 IO**。此外，AEC 实现存在逻辑闭环导致回声泄漏。

以下是详细的问题清单及对应的代码修复实施计划。

---

## 2. 核心问题与实施计划 (Problem & Implementation)

### 2.1 [Critical] TimbreRestore 模块严重内存分配
**问题描述**：
`src/audio/timbre_restore.rs` 中的 `process_frame` 方法在每一帧（10ms）都执行多次堆内存分配：
1. `Vec::with_capacity` 创建新输入缓冲。
2. `Tensor::from_array` 和 `try_extract_tensor` 触发数据拷贝。
3. `self.hidden = h_out.to_vec()` 深度拷贝 Hidden State。
**后果**：极端严重的 GC 压力和耗时抖动，导致音频爆音。

**实施计划**：
修改 `TimbreRestore` 结构体，使用预分配的 `input_buffer` 和 `hidden_buffer`，移除所有 `Vec::new` 和 `to_vec`。

```rust
// File: src/audio/timbre_restore.rs

pub struct TimbreRestore {
    session: Session,
    context_size: usize,
    // [Fix] 预分配的大 buffer： [context ... frame]
    input_buffer: Vec<f32>, 
    hidden: Vec<f32>,   
    hidden_size: usize,
    num_layers: usize,
}

impl TimbreRestore {
    // ... new() 中预分配 vec![0.0; context_size + max_frame_len] ...

    pub fn process_frame(&mut self, frame: &mut [f32]) -> Result<()> {
        let frame_len = frame.len();
        let total_len = self.context_size + frame_len;
        
        // 1. 填充输入 (Zero Alloc)
        if self.input_buffer.len() < total_len {
             self.input_buffer.resize(total_len, 0.0);
        }
        // 将当前帧拷贝到 buffer 尾部
        self.input_buffer[self.context_size..total_len].copy_from_slice(frame);

        // 2. 构造 View (Zero Copy)
        let input_tensor = Tensor::from_array((vec![1, 1, total_len], &self.input_buffer[..total_len]))?;
        let h_in = Tensor::from_array((vec![self.num_layers, 1, self.hidden_size], self.hidden.as_slice()))?; 

        // 3. 推理
        let outputs = self.session.run(ort::inputs![input_tensor, h_in])?;

        // 4. 提取与更新 (Zero Alloc)
        let output_data: &[f32] = outputs[0].try_extract_tensor_data()?; 
        let start_idx = output_data.len().saturating_sub(frame_len);
        frame.copy_from_slice(&output_data[start_idx..]); // 回填音频

        let h_out_data: &[f32] = outputs[1].try_extract_tensor_data()?;
        self.hidden.copy_from_slice(h_out_data); // 原地更新 Hidden

        // 5. 更新 Context (Ring shift)
        if self.context_size > 0 {
            let new_start = total_len.saturating_sub(self.context_size);
            self.input_buffer.copy_within(new_start..total_len, 0);
        }
        Ok(())
    }
}
```

---

### 2.2 [High] AEC 双讲逻辑错误导致回声泄漏
**问题描述**：
`src/capture.rs` 中使用 VAD/能量判断双讲状态，因为回声本身能量大且具有语音特征，被误判为双讲，从而调用 `aec.set_double_talk(true)` 强制降低 AEC 抑制等级。同时，`src/audio/aec.rs` 默认开启 `Aggressive` 模式，导致单讲时误切近端语音。
**后果**：回声消不干净，且自己说话断断续续。

**实施计划**：
1.  **Capture 层**：移除所有外部对 `aec.set_double_talk` 的调用。
2.  **AEC 层**：将默认抑制等级调整为 `Moderate`，移除 `to_vec` 分配。

```rust
// File: src/capture.rs (修改 Audio Worker Loop)

// [DELETE] 删除以下逻辑
// let dt_active = dt_bypass_frames > 0;
// aec.set_double_talk(dt_active); 

// [KEEP] 仅保留纯处理调用
if aec_enabled {
    aec.process_render(buffer);
}
```

```rust
// File: src/audio/aec.rs

// [Fix 1] 默认参数调整
// let aggressive = true;  ->  let aggressive = false;
// 在 apply_config 中增加 Moderate 默认回退：
// ...
// } else {
//    EchoCancellationSuppressionLevel::Moderate // [NEW DEFAULT]
// };

// [Fix 2] 移除内存分配
pub fn process_render(&mut self, buf: &[f32]) {
    // ...
    // [DELETE] let mut temp = chunk.to_vec();
    // [ADD] Use self.scratch
    if chunk.len() == self.frame_size {
        self.scratch.copy_from_slice(chunk);
    } else {
        self.scratch[..chunk.len()].copy_from_slice(chunk);
        self.scratch[chunk.len()..].fill(0.0); // Padding
    }
    // process(&mut self.scratch)
}
```

---

### 2.3 [Medium] 阻塞式日志与隐形锁
**问题描述**：
所有效果器（`TransientShaper`, `Highpass`, `Saturation`）的 `sanitize_samples` 函数中包含 `log::warn!`。一旦触发 NaN 检测，日志 IO 会立即阻塞音频线程，导致音频卡顿。
**后果**：不可预测的音频掉帧（Drop-out）。

**实施计划**：
使用原子计数器替代直接日志打印。

```rust
// File: src/audio/transient_shaper.rs (及其他模块)

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TransientShaper {
    // ...
    error_count: AtomicUsize, // [NEW]
}

// 修改 sanitize_samples
fn sanitize_samples(&self, samples: &mut [f32]) -> bool {
    let mut found = false;
    for s in samples {
        if !s.is_finite() {
            *s = 0.0;
            found = true;
        }
    }
    if found {
        // [Fix] 仅原子递增，不打印日志
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }
    found
}
```

---

### 2.4 [High] DynamicEQ 预设切换内存分配
**问题描述**：
`src/audio/eq/dynamic_eq.rs` 中的 `rebuild_bands` 会在切换预设时销毁并新建 `Vec<DynamicBand>`。
**实施计划**：
改用固定大小数组（`[DynamicBand; MAX_EQ_BANDS]`）或在初始化时一次性分配最大容量的 Vec，切换预设时仅更新 Band 内部参数，不进行内存重分配。

## 3. 总结
请严格按照上述 "2.1 TimbreRestore" -> "2.2 AEC" -> "2.3 Log" 的顺序进行修复。
1.  **TimbreRestore** 是性能瓶颈核心。
2.  **AEC 修复** 直接解决用户反馈的双讲/回声业务问题。
3.  **日志优化** 提升系统的鲁棒性。
