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

---

## 4. AEC 生产级路线图与代码修改建议（vNext）

> 说明：本节基于当前 Demo 代码现状（`demo/src/capture.rs` + `demo/src/audio/aec.rs` + `demo/src/audio/adaptive.rs`），给出从“可用 Demo”走向“商业级 AEC”所需的路线图与对应的**代码级改动建议**。这里只给方向/伪代码，不直接修改源码。

### 4.1 目标与验收标准

**目标**：在不同设备（内置麦克风、USB 声卡、会议音箱、耳机/蓝牙）与不同环境（安静/办公室/嘈杂、短/长混响 RT60）下，稳定提供生产级回声消除与双讲保护。

**核心指标（建议纳入 CI 回归）**：
1. **ERLE（单讲回声抑制）**：> 35 dB（宽带），且收敛时间 < 1.5 s。  
2. **双讲近端保护**：双讲时近端衰减 < 3 dB，远端残留回声 < -30 dBFS。  
3. **音质副作用**：无明显抽吸/泵音、无咔嗒切换、无低频发薄。  
4. **稳定性**：长时间运行无发散；峰值/削波场景能自保护不崩坏。

### 4.2 路线图（阶段化落地）

#### 阶段 A：稳定 AEC 集成（1–2 周，必须先做）
**问题对齐**：  
- AEC 每帧重配导致收敛抖动和 CPU 浪费。  
- 设备推荐配置（是否启用 AEC/输出音量）未真正生效。  
- UI 延迟控制与 delay_agnostic 语义不一致。  
- Hangover 参考为静音，无法消混响尾音。  

**改动建议**：
1. **配置节流/状态机修正**（`demo/src/audio/aec.rs`）  
   - 增加内部 `config_dirty` 或缓存上一次 suppression/delay/aggressive 状态。  
   - 只有在 suppression 档位变化或过渡期结束时才调用 `proc.set_config()`；稳定单讲不应每帧重配。  
   - 修正 `set_double_talk` 的更新条件，避免 `dt_exit_frames==0 && !double_talk` 时每次都触发 `apply_config()`。
2. **设备自适应真正落地**（`demo/src/capture.rs` + `demo/src/audio/adaptive.rs`）  
   - `recommend_config.enable_aec=false` 时，初始化阶段直接把 `aec_user_enabled=false`，并在运行中禁止自动打开。  
   - `recommend_config.output_volume` 应影响实际播放/参考能量（例如在混入 far-end 前做一次全局 gain，或调整 `headroom_gain`）。  
3. **延迟语义统一**（`demo/src/audio/aec.rs` + `demo/src/capture.rs`）  
   - 若继续使用 `enable_delay_agnostic=true`，则 `set_delay_ms`/UI 延迟滑条只作为“初始提示”，且**不得频繁重配**。  
   - 如果要支持“固定延迟”模式：提供开关 `delay_agnostic=false` 且 `stream_delay_ms=Some(delay_ms)`，并在 UI 明确两种模式。  
4. **Hangover 参考修正**（`demo/src/capture.rs`）  
   - Hangover 期间应保留最近一帧真实 render 参考（或其衰减版本），而不是 0 填充；否则对尾音消除无效。  
5. **高通单一化**（`demo/src/capture.rs` / `demo/src/audio/aec.rs`）  
   - 只保留一处高通（外部或 WebRTC 内部二选一），避免低频相关性被削弱。

#### 阶段 B：提升双讲/参考链路鲁棒性（2–4 周）
**目标**：在音乐伴奏、强回声、远端很响/近端很弱等复杂场景下避免误判，减少残留与吞音。

**改动建议**：
1. **多特征双讲检测**（`demo/src/capture.rs`）  
   - 现有 `is_true_double_talk` 仅 VAD+能量差，建议加入：  
     - far-end 能量门控（render 很弱时不触发 DT）；  
     - render–capture 相干度/互相关峰值（高相关→更像回声）；  
     - AEC 内部 ERLE/残余估计（如果库暴露）。  
   - 进入双讲敏感、退出保守；`dt_holdoff_frames` 根据 `rt60` 自适应（RT60 越长 holdoff 越长）。  
2. **render 参考来自“最终送到扬声器的信号”**  
   - 生产中 render 需包含系统/应用音量、EQ/DRC/limiter 后的真实输出；否则滤波器匹配偏差大。  
   - Demo 的 `AutoPlayBuffer` 只是内部测试链路；真实通话需接入 OS/SDK far-end PCM。

#### 阶段 C：残余回声抑制与非线性处理（4–8 周）
**目标**：达到商业级“干净度”，尤其是小音箱/手机等带非线性失真的设备。

**改动建议**：
1. **Residual Echo Suppressor（RES/NLP）**  
   - 在 `aec.process_capture` 之后新增频域残余回声抑制模块（带舒适噪声/抑制平滑），对 AEC3 剩余回声做二次压制。  
   - 建议新建 `demo/src/audio/residual_echo.rs`，在 worker 里 AEC 后、DF 前插入。  
2. **Hybrid/Neural AEC**  
   - 对强非线性/长混响设备，加入轻量神经回声抑制（可选开关，延迟 ≤10ms）。  
   - 与传统 AEC3 并联或串联（先 AEC3 再 Neural RES），并对设备类型自适应开启。

#### 阶段 D：生产级评测与持续调参（持续）
**必建体系**：
1. **多设备/多房间数据集**：不同距离、角度、音量、单双讲、音乐+语音混合。  
2. **自动回归脚本**：输出 ERLE、双讲保护衰减、AECMOS/PESQ、收敛时间、抽吸/失真评分。  
3. **阈值治理**：所有策略阈值（能量门控/holdoff/RES 强度）必须在回归里验证，不允许凭直觉改动。

### 4.3 关键代码修改建议（按文件）

#### 4.3.1 `demo/src/audio/aec.rs`
1. **apply_config 节流**  
   - 增加字段缓存：`last_suppression: Option<EchoCancellationSuppressionLevel>`、`last_aggressive: bool`、`last_delay_ms: i32`。  
   - 仅当新计算的 suppression 或 aggressive/delay 与 last 不同时才 `set_config`。  
2. **修正 set_double_talk 的重配条件**  
   - 过渡期倒计时每帧递减可以保留，但只有在 `state_changed==true` 或 `dt_exit_frames` 从 1→0 时更新配置。  
3. **延迟模式开关**  
   - 暴露 `set_delay_agnostic(bool)`：  
     - `true`：`stream_delay_ms=None`（现状），UI 仅作初始提示；  
     - `false`：`stream_delay_ms=Some(delay_ms)` 并禁止频繁更新。  

#### 4.3.2 `demo/src/capture.rs`
1. **设备推荐生效**  
   - 初始化阶段读取 `RecommendedConfig`：  
     - `enable_aec=false` → `aec_user_enabled=false` + UI 灰化；  
     - `output_volume` → 作用于 far-end 混入/参考能量（防止过响导致发散）。  
2. **双讲调用节流**  
   - 计算 `is_double_talk` 后仅在状态变化或 holdoff 刷新时调用 `aec.set_double_talk`，避免每帧触发 AEC 内部状态变更。  
3. **Hangover 参考保留**  
   - 在 `render_active` 进入 hangover 时，保留上一帧 `aec_ref_buf`（可指数衰减），不要清零。  
4. **高通选择**  
   - 若保留外部高通，则在 AEC Config 中关闭内部高通（或反之）。  

#### 4.3.3 `demo/src/audio/adaptive.rs`
1. **设备识别增强**  
   - 仅靠名称不可靠，建议读取系统设备 ID/通道数/是否带 DSP-AEC 标志；区分耳机/蓝牙/会议音箱/车载等。  
2. **未知设备的运行时校准**  
   - 启动 2–3 秒做一次 far-end 扫频/粉噪能量测量，估计回声路径增益与初始延迟，生成临时 profile。

---

> 若按上述路线图推进，阶段 A 解决“稳定与语义一致性”，阶段 B/C 把效果从“Demo 可用”推到“商业级干净度”，阶段 D 保证持续可维护与可回归。
