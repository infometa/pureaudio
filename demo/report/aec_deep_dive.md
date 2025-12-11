# AEC 深度审计报告：发现架构级致命缺陷

## 1. 核心结论
经过对 `capture.rs` 信号路由的深度分析，我找到了导致 "AEC 效果差" 和 "自己说话断断续续" 的**根本原因**：

**架构级错误：近端语音（麦克风信号）被错误地混入了 AEC 的参考信号（Reference）中。**

这导致 AEC 算法认为“用户的声音”也是需要消除的“回声”的一部分，从而拼命尝试消除用户说的话。这不仅仅是参数调整能解决的，必须修改代码逻辑。

## 2. 问题详解

### 错误的代码逻辑 (`capture.rs`)

1.  **L1249**: `df.process(inframe, outframe)` -> `outframe` 填充了降噪后的麦克风信号（近端语音）。
2.  **L1966-1977**: 如果有 `auto_play_buffer`（模拟远端信号），将其混音叠加到 `outframe` 中。
    *   此时 `outframe` = **近端语音 + 远端信号**。
3.  **L2028**: `aec.process_render(buffer)`，其中 `buffer` 就是这个 `outframe`。
    *   **后果**：你告诉 AEC：“扬声器正在播放（近端语音 + 远端信号）”。
    *   AEC 的工作是：从麦克风输入中，减去“扬声器播放的内容”。
    *   因此，AEC 会尝试：**从麦克风输入中减去（近端语音）**。
    *   **结果：你自己说的话被当成回声消除了！**

### 正确的逻辑

AEC 的参考信号（Render）必须**纯净的远端信号**（即仅限 `auto_play_buffer` 的内容），绝对不能包含当前的麦克风采集信号。

即使为了从扬声器听到自己的声音（耳返/Side-tone）而把麦克风信号混合输出，这个混合动作也必须发生在 **AEC 参考信号采样点之后**，或者在送给 AEC Reference 时专门剔除麦克风分量。

## 3. 修复方案

### 修改 `src/capture.rs`

我们需要将“混音输出”和“提取参考信号”分离。

```rust
// [修改前]
// 混音 -> outframe
// aec.process_render(outframe)

// [修改后建议]

// 1. 定义一个专门的 buffer 用于存放纯净远端参考
let mut aec_reference = [0.0f32; 480]; // 假设帧长匹配
let mut has_far_end = false;

// 2. 处理自动播放逻辑
if let Some(ref pcm) = auto_play_buffer {
    // ... 
    // 将 pcm 数据拷贝到 aec_reference (用于 AEC)
    // 同时混合到 outframe (用于播放)
    for i in 0..copy_len {
        let far_end_sample = pcm[auto_play_pos + i];
        
        // 填充参考信号
        if i < aec_reference.len() {
             aec_reference[i] = far_end_sample;
        }

        // 混音到输出 (可选，看是否需要听到远端)
        buffer[i] += far_end_sample * 0.5; 
    }
    has_far_end = true;
}

// 3. 仅当有远端信号时，才运行 AEC Render
if aec_enabled && has_far_end {
    // 这里的 reference 只有远端声音，没有麦克风声音
    aec.process_render(&aec_reference);
} else if aec_enabled {
    // 如果没有远端信号，通常应该传入静音或者不调用 process_render
    // 为了维持 AEC 内部状态更新，可以传全 0
    aec_reference.fill(0.0);
    aec.process_render(&aec_reference);
}
```

## 4. 其他遗留问题
之前的审计结论依然有效，且与此问题叠加导致效果更差：
1.  **外部双讲干扰**：`aec.set_double_talk` 必须移除。
2.  **TimbreRestore 内存分配**：必须修复。

但请注意：**只要参考信号路由错误（混入近端语音）不解决，任何参数调整都无法解决“自己说话断续”的问题。**

## 5. 开发建议
请按以下顺序修改代码：
1.  **修正信号路由**（最优先）：确保 `aec.process_render` 只吃到 `auto_play_buffer` 的数据，不要吃 `outframe`（麦克风数据）。
2.  **移除干扰逻辑**：删除 `aec.set_double_talk`。
3.  **优化内存**：修复 `TimbreRestore`。
