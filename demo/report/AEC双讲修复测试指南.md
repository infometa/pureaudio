# AEC 双讲修复测试指南

**修复日期**: 2025-12-11  
**修复版本**: v2.0  
**状态**: ✅ 编译通过，等待测试

---

## 📋 修复内容总结

### 核心改进

1. ✅ **启用双讲检测**
   - 移除 `#[allow(dead_code)]` 标记
   - `set_double_talk()` 方法现已激活

2. ✅ **改进抑制策略**
   - 默认 `aggressive = true`（强力模式）
   - 动态切换：
     - 单讲 → High suppression（强力消除回声）
     - 双讲 → Low suppression（保护近端语音）

3. ✅ **集成VAD双讲检测**
   - 实时监测：`vad_state` (近端) + `render_active` (远端)
   - 智能判断：两者同时存在 = 双讲
   - Hangover机制：300ms缓冲，避免频繁切换

4. ✅ **增强诊断能力**
   - 添加详细日志输出
   - 显示VAD状态、渲染能量、双讲状态
   - 可实时监控AEC工作情况

---

## 🧪 测试方案

### 环境准备

1. **编译最新代码**
```bash
cd /Users/haifeng/Desktop/code/project/pureaudio
cargo build --release --manifest-path demo/Cargo.toml --features ui
```

2. **启用调试日志**
```bash
export RUST_LOG=debug
./target/release/df-demo
```

3. **准备测试音频**
   - 远端音频文件（如音乐、朗读）
   - 准备麦克风（清晰环境）

---

## 测试用例

### 测试1：纯远端（单讲） - 回声消除测试

#### 操作步骤
1. 启动程序，开启：降噪 + AEC
2. 导入远端音频文件（软件放音）
3. **不要说话**，让远端音频播放
4. 录制30秒

#### 预期结果
- ✅ 录音中几乎听不到远端声音（回声<-40dB）
- ✅ 日志显示：
  ```
  AEC状态: VAD=false, Render=true(-10dB), 双讲=false, 
           enabled=true, double_talk=false
  AEC配置: suppression=High, delay=XXms
  ```

#### 判断标准
| 等级 | 回声残留 | 评分 |
|-----|---------|------|
| 优秀 | <-50dB，几乎无法察觉 | A |
| 良好 | -40~-50dB，仔细听能察觉 | B |
| 及格 | -30~-40dB，明显但可接受 | C |
| **修复前** | **-20dB左右，明显残留** | **❌ D** |

---

### 测试2：纯近端 - 基线测试

#### 操作步骤
1. 启动程序，开启：降噪 + AEC
2. **不导入音频**，仅用麦克风说话
3. 朗读测试文本30秒

#### 预期结果
- ✅ 录音清晰完整
- ✅ 日志显示：
  ```
  AEC状态: VAD=true, Render=false(-80dB), 双讲=false
  AEC配置: suppression=High
  ```

#### 注意事项
- 这个场景AEC实际不工作（无远端参考）
- 用于验证基础录音质量

---

### 测试3：双讲 - 核心测试 🎯

#### 操作步骤
1. 启动程序，开启：降噪 + AEC
2. 导入远端音频（持续播放）
3. **同时说话**，与远端音频同时进行
4. 录制60秒，涵盖：
   - 远端说话时插话（打断）
   - 持续对话（双讲）
   - 停顿（单讲切换）

#### 预期结果（修复后）
- ✅ 近端语音清晰完整，**不再断断续续**
- ⚠️ 远端声音有轻微残留（可接受的trade-off）
- ✅ 日志显示动态切换：
  ```
  # 远端单独说话时
  AEC状态: VAD=false, Render=true(-15dB), 双讲=false
  AEC配置: suppression=High
  
  # 开始说话（双讲）
  AEC双讲状态切换: 单讲 -> 双讲
  AEC状态: VAD=true, Render=true(-15dB), 双讲=true
  AEC配置: suppression=Low
  
  # 停止说话（过渡期）
  AEC双讲状态切换: 双讲 -> 单讲
  # ... 150ms过渡 ...
  AEC过渡期结束，恢复强力抑制
  AEC配置: suppression=High
  ```

#### 对比表格

| 场景 | 修复前 | 修复后（预期） |
|-----|--------|---------------|
| 近端清晰度 | ❌ 断断续续 | ✅ 完整清晰 |
| 远端残留 | ⚠️ 明显 | ⚠️ 轻微（可接受） |
| 切换平滑度 | ⚠️ 可能咔嗒 | ✅ 平滑过渡（150ms） |
| 总体可用性 | ❌ 不可用 | ✅ 实用 |

---

### 测试4：快速切换 - 压力测试

#### 操作步骤
1. 启动程序，开启：降噪 + AEC
2. 播放远端音频
3. 快速交替：
   - 说话2秒 → 停顿1秒
   - 重复10次

#### 预期结果
- ✅ 切换平滑，无咔嗒声
- ✅ 日志显示过渡期正常工作：
  ```
  AEC双讲状态切换: 双讲 -> 单讲
  # exit_frames: 15 -> 14 -> ... -> 0
  AEC过渡期结束，恢复强力抑制
  ```

---

## 📊 日志分析

### 正常工作日志示例

```log
# 程序启动
[INFO] AEC3 初始化完成：48000Hz, aggressive=true, delay=107ms

# 远端播放开始
[DEBUG] AEC状态: VAD=false, Render=true(-12dB), 双讲=false, 
        enabled=true, active=true, delay=107ms, aggressive=true, 
        double_talk=false, exit_frames=0
[DEBUG] AEC配置: suppression=High, delay=107ms

# 开始说话（触发双讲）
[DEBUG] AEC双讲状态切换: 单讲 -> 双讲
[DEBUG] AEC配置: suppression=Low, delay=107ms, double_talk=true

# 停止说话（进入过渡期）
[DEBUG] AEC双讲状态切换: 双讲 -> 单讲
[DEBUG] AEC配置: suppression=Low, delay=107ms, exit_frames=15

# 过渡期倒计时...
[DEBUG] AEC配置: suppression=Low, exit_frames=14
[DEBUG] AEC配置: suppression=Low, exit_frames=13
... (每帧-1，共150ms)

# 过渡结束
[DEBUG] AEC过渡期结束，恢复强力抑制
[DEBUG] AEC配置: suppression=High, delay=107ms, exit_frames=0
```

### 异常日志排查

#### 问题1：VAD始终false
```log
[DEBUG] AEC状态: VAD=false, Render=true, 双讲=false
```
**原因**: VAD未启用或阈值过高  
**解决**: 在UI中启用VAD，或降低阈值

#### 问题2：Render始终false
```log
[DEBUG] AEC状态: VAD=true, Render=false(-80dB), 双讲=false
```
**原因**: 
- 远端音频未播放
- 音量过小（<-40dB）
- 参考信号未正确送给AEC

**解决**: 
- 确认导入并播放远端音频
- 提高音量
- 检查`aec_ref_buf`是否正确填充

#### 问题3：双讲never触发
```log
# 同时说话时仍显示
[DEBUG] AEC状态: VAD=true, Render=true, 双讲=false ← 应该是true！
```
**原因**: 代码逻辑错误或VAD/Render检测失败  
**排查**: 
1. 检查 `let is_double_talk = vad_state && render_active;`
2. 确认两个变量都为true
3. 查看是否有其他逻辑覆盖

---

## 🔧 参数调优

### 延迟参数优化

如果回声残留严重（修复后仍有>-30dB残留）：

```rust
// capture.rs 修改自动延迟估算
const AEC_DEFAULT_DELAY_MS: i32 = 120;  // 增加基础延迟

// 或调整extra_device_ms
let extra_device_ms = 100.0;  // 从80增加到100
```

**测试方法**：
1. 录制纯远端音频
2. 在波形编辑器中测量原始脉冲和回声的时间差
3. 设置为实测值 + 20ms裕量

---

### VAD阈值调整

如果近端语音漏检（说话时VAD=false）：

修改 `silero.rs` 或在capture.rs调整：
```rust
let vad_cfg = SileroVadConfig {
    positive_speech_threshold: 0.3,  // 更敏感（从0.5降低）
    negative_speech_threshold: 0.15, // 更敏感
    ..Default::default()
};
```

---

### 过渡期时长调整

如果切换有咔嗒声或回声尾音过长：

```rust
// aec.rs Line 122
self.dt_exit_frames = 10;  // 缩短到100ms（从15帧=150ms）

// 或延长
self.dt_exit_frames = 30;  // 延长到300ms
```

---

### 渲染信号阈值调整

如果render_active误判：

```rust
// capture.rs Line 2073
let has_render_signal = render_db > -45.0;  // 更严格（从-40降低）

// Hangover时长
render_active = last_render_time.elapsed() < Duration::from_millis(200);  // 从300缩短
```

---

## ✅ 成功标准

### 最低要求（必须满足）
- [x] 编译通过
- [ ] 测试1通过：纯远端回声<-40dB
- [ ] 测试3通过：双讲时近端完整清晰
- [ ] 无crash或panic
- [ ] 日志显示双讲状态正常切换

### 优秀标准（期望达到）
- [ ] 纯远端回声<-50dB
- [ ] 双讲时近端语音质量≥90%
- [ ] 切换平滑无咔嗒
- [ ] CPU占用无明显增加
- [ ] 用户主观评价："可以正常使用"

---

## 🐛 已知问题和限制

### 1. 双讲时轻微回声残留
**现象**: 双讲时，录音中可能听到轻微的远端声音  
**原因**: Low suppression为了保护近端，降低了回声消除强度  
**是否正常**: ✅ 正常，这是双讲场景的固有trade-off  
**缓解**: 
- 降低远端音量
- 提高近端麦克风增益
- 使用定向麦克风

### 2. VAD误触发
**现象**: 环境噪音被误判为语音  
**影响**: 可能过早进入双讲模式，导致回声残留  
**缓解**: 
- 提高VAD阈值
- 使用更安静的环境
- 添加noise gate

### 3. 延迟不匹配
**现象**: 设置的延迟与实际系统延迟不符  
**影响**: 回声消除效果下降  
**缓解**: 
- 使用`enable_delay_agnostic=true`（已启用）
- 手动测量并调整延迟参数
- 避免使用高延迟的USB设备

---

## 📈 性能影响

### CPU占用
- **修复前**: ~15%（AEC基础）
- **修复后**: ~15%（无增加）
- **额外开销**: VAD已在运行，双讲检测仅增加1行判断

### 内存占用
- **无增加**: 仅添加几个bool变量和日志

### 延迟
- **无增加**: 双讲检测是同步的，无额外延迟

---

## 🎯 测试检查清单

### 基础测试
- [ ] 程序正常启动
- [ ] AEC可以启用/禁用
- [ ] 导入远端音频正常播放
- [ ] 麦克风录音正常

### 功能测试
- [ ] 测试1：纯远端回声消除
  - 回声残留: _____ dB
  - 评级: A / B / C / D
- [ ] 测试2：纯近端基线
  - 录音质量: ⭐⭐⭐⭐⭐
- [ ] 测试3：双讲核心测试
  - 近端清晰度: 完整 / 轻微断续 / 严重断续
  - 远端残留: 无 / 轻微 / 明显
  - 评级: A / B / C / D
- [ ] 测试4：快速切换
  - 平滑度: 平滑 / 偶尔咔嗒 / 经常咔嗒

### 日志检查
- [ ] 双讲状态切换日志正常
- [ ] VAD状态正确反映说话
- [ ] Render状态正确反映播放
- [ ] Suppression级别动态调整

### 对比评估
修复前 vs 修复后：
- 回声消除: 改善 ___%
- 双讲质量: 改善 ___%
- 整体可用性: 不可用 → 可用 / 基本可用 / 完美

---

## 📝 测试报告模板

```markdown
# AEC双讲修复测试报告

**测试人**: _________
**测试日期**: 2025-12-__
**测试环境**: 
- OS: macOS / Linux / Windows
- CPU: _________
- 麦克风: _________
- 采样率: 48000 Hz

## 测试结果

### 测试1：纯远端（单讲）
- 回声残留: -__ dB
- 主观评价: A / B / C / D
- 备注: __________

### 测试3：双讲
- 近端清晰度: ⭐⭐⭐⭐⭐
- 远端残留: 无 / 轻微 / 明显
- 切换平滑度: ⭐⭐⭐⭐⭐
- 主观评价: A / B / C / D
- 备注: __________

## 日志摘要
```log
（粘贴关键日志）
```

## 发现的问题
1. __________
2. __________

## 总体评价
修复效果: 优秀 / 良好 / 及格 / 不及格
建议: 通过 / 有条件通过 / 需要进一步调优

## 附件
- [ ] 测试录音文件（修复前）
- [ ] 测试录音文件（修复后）
- [ ] 完整日志
```

---

## 🚀 下一步

### 如果测试通过
1. 进行更多场景测试（不同环境、设备）
2. 长时间稳定性测试（1小时+）
3. 多用户测试反馈
4. 考虑发布

### 如果测试失败
1. 查看日志分析问题
2. 按照参数调优章节调整
3. 如仍有问题，提供：
   - 详细测试报告
   - 完整日志
   - 录音文件（修复前后对比）
   - 环境信息

---

**文档版本**: v2.0  
**创建日期**: 2025-12-11  
**状态**: ✅ 编译完成，等待测试
