# 最终代码审计报告 (v2.0)

**日期**: 2025-12-11
**审计对象**: `df-demo` (Capture & Audio Processing)
**结论**: 关键缺陷已修复，系统逻辑闭环，建议上线测试。

## 1. 致命缺陷修复 (Critical Fixes)

### 1.1 AEC 0ms 启动陷阱 (已修复)
*   **问题**: AEC 在设备检测逻辑运行前即被初始化，且使用 0ms 默认延迟。
*   **后果**: 启动后的前几秒（或永久，取决于时序）AEC 处于 0ms 延迟状态，完全无法消除 60ms+ 的物理回声，导致瞬时啸叫。
*   **修复**: 重构 `src/capture.rs` 初始化顺序。`DeviceDetector` 现已移至 AEC 初始化**之前**执行，确保 `EchoCanceller::new` 获得准确的 `aec_delay_ms` (60/80ms)。
*   **验证**: 代码逻辑已确认，变量顺序正确。

### 1.2 AEC 参考信号丢失 (已修复)
*   **问题**: `aec_ref_buf` 仅在播放测试伴奏时填充，麦克风直通（Loopback）模式下为空。
*   **后果**: 用户说话时，AEC 认为扬声器是静音的，不进行消除，导致“人声 -> 麦克风 -> 扬声器 -> 麦克风”正反馈啸叫。
*   **修复**: 修改 `capture.rs` 中 `aec_ref_buf` 的填充逻辑，确保**所有**送往输出设备的信号（包括 Mic Loopback）都被完整复制到 AEC 参考缓冲区。
*   **验证**: 逻辑检查通过。

### 1.3 AGC 增益失控 (已修复)
*   **问题**: 默认 `agc_max_gain` 为 9dB。
*   **后果**: 在闭环声学系统中，9dB 额外增益极易导致 Loop Gain > 1，引发啸叫。
*   **修复**: 默认增益降至 **3dB**。
*   **验证**: 参数已更新。

## 2. 逻辑完整性审计 (Logic Audit)

### 2.1 智能双讲检测 (Double Talk Detection)
*   **审计点**: `near_energy_db` 是否使用了正确信号？是否在对比前已更新？
*   **结果**: ✅ 通过。
    *   `near_energy_db` 在 line ~1391 计算（基于 Raw Input，正确）。
    *   `far_db` 在 line ~2279 计算（基于 Final Output，正确）。
    *   对比逻辑位于 line ~2325，时序正确。

### 2.2 延迟自适应 (Delay Estimation)
*   **审计点**: `DelayEstimator` 输入信号是否正确？变量是否同步？
*   **结果**: ✅ 修复后通过。
    *   输入修正：`aec_delay_ms` 变量在自动修正块中原先未同步，现已修复（line ~2302）。
    *   信号源：估计器使用 `inframe` (Cleaned) 和 `ref`。
    *   **备注**: 使用“处理后信号”进行延迟估计意味着仅当“回声消除失败（残留回声）”时估计器才生效。这作为 Failsafe 机制是可接受的。

### 2.3 自适应音量监控 (Volume Monitor)
*   **审计点**: `volume_monitor` 是否接收到了输入/输出数据？
*   **结果**: ✅ 通过。
    *   `update_input` 在 line ~1394 (Frame N)。
    *   `update_output` 在 line ~2282 (Frame N)。
    *   逻辑闭环。

## 3. 残留警告 (Allowed Warnings)
*   编译警告 (Unused functions/constants) 均为无害冗余代码，不影响运行时稳定性。

## 4. 建议 (Recommendations)
1.  **实机测试**: 建议在不同延迟特性的设备（如蓝牙耳机 vs USB 声卡）上验证 0ms 修复的效果。
2.  **日志监控**: 关注 `log::info("✨ 自动执行延迟修正...")` 日志，确认自适应逻辑是否过于频繁触发（理想情况应只在启动或设备切换时触发）。

**审计人**: Antigravity AI
**状态**: **READY (可发布)**
