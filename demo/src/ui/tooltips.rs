#![allow(dead_code)]

pub const NOISE_SUPPRESSION: &str =
    "控制降噪强度。\n• 0 dB: 不降噪\n• 50 dB: 适度降噪（推荐）\n• 100 dB: 最强降噪，可能让声音发闷";

pub const POST_FILTER: &str = "增强降噪效果，但可能影响音质。\n• 0: 关闭后滤波（推荐）\n• 0.5: 适度增强\n• 1.0: 最强，可能过度抑制";

pub const MIN_THRESHOLD: &str = "低于此阈值的频段被认为是噪声。\n• -15 dB: 保守（可能漏噪声）\n• -5 dB: 适中\n• 5 dB: 激进（可能误判人声）";

pub const MAX_ERB_THRESHOLD: &str = "ERB 频段的噪声判断上限。一般保持默认即可，越高越保守。";

pub const MAX_DF_THRESHOLD: &str = "DeepFilter 频段的噪声判断上限。建议保持默认值，特殊场景再调。";

pub const EQ_ENABLED: &str =
    "启用或禁用动态 EQ 处理。\n• 开启: 改善音质，恢复高频\n• 关闭: 仅执行降噪";

pub const EQ_PRESET_GENERAL: &str = "通用语音增强：平衡的参数，适合大多数场景。";
pub const EQ_PRESET_PODCAST: &str = "播客优化：更强齿音控制(5.5:1)，高频更亮，久听不累。";
pub const EQ_PRESET_MEETING: &str = "会议清晰化：最强齿音控制(5.8:1)，强调可懂度。";

pub const EQ_MIX: &str = "控制 EQ 处理的混合比例。\n• 0%: 完全旁路 EQ\n• 50%: 原始与 EQ 各一半\n• 100%: 完全应用 EQ（推荐）";

pub const EQ_ADVANCED_TOGGLE: &str =
    "展开后可手动调节每个频段的增益，适合有音频经验的用户精细调节。";

pub const NOISE_ADVANCED_TOGGLE: &str =
    "展开高级降噪阈值，精细控制噪声判断范围。默认值已适配大多数场景，调整请谨慎。";

pub const HIGHPASS_FILTER: &str =
    "高通滤波器 (40~90Hz)\n移除极低频噪声（空调轰鸣/桌面振动），避免切掉男声胸腔，默认 60Hz。";

pub const TRANSIENT_SHAPER: &str =
    "瞬态增强\n提升辅音等瞬态细节，让语音更有冲击力，补偿降噪造成的瞬态损失。";

pub const TRANSIENT_GAIN: &str = "瞬态增益\n控制瞬态提升幅度。推荐 3-5 dB，范围 0-12 dB。";

pub const TRANSIENT_MIX: &str =
    "瞬态混合\n设置干/湿比例。0% = 原始信号，100% = 完全处理。推荐 60-80%。";

pub const TRANSIENT_SUSTAIN: &str =
    "释放增益\n对非瞬态部分的增益，可设为负值以削短尾音（会议室去混响）。范围 -12~+6 dB。";

pub const AGC: &str = "自动增益控制\n平衡整体响度到约 -16 dBFS，避免忽大忽小，并带有软限制保护。";
pub const AGC_TARGET: &str =
    "目标电平\nAGC 会把输出拉向此响度。推荐 -12~-6 dBFS，主播模式可升至 -3 dBFS。";
pub const AGC_MAX_GAIN: &str = "最大增益\n限制轻声音最多放大多少。范围 6~15 dB。";
pub const AGC_MAX_ATTEN: &str = "最大衰减\n限制响声音最多衰减多少。范围 3~12 dB。";

pub const SATURATION: &str =
    "饱和/谐波增强\n在降噪后补充模拟味道和厚度，默认开启，Drive≈1.2，Mix=100%。";
pub const SATURATION_DRIVE: &str = "驱动 (Drive)\n决定进入软削波的强度，推荐 1.1~1.4。";
pub const SATURATION_MAKEUP: &str =
    "补偿增益\n饱和会压缩峰值，此参数可微调整体增益，范围 -6~+3 dB。";
pub const SATURATION_MIX: &str = "混合比例\n控制干/湿。0% 全干，100% 全处理。";

pub const DF_MIX: &str =
    "降噪混合\n控制 DF 输出与原始（预处理后）信号的混合比例。100% 全湿；降低可保留自然度。";

pub const EQ_BAND_LOW: &str =
    "调节低频（约 100Hz）的增益偏移。负值减少隆隆感，正值增加厚度。范围 -12~+12 dB";
pub const EQ_BAND_MUD: &str =
    "调节 260Hz 左右的浑浊感。负值减少箱音鼻音，正值增加温暖。范围 -12~+12 dB";
pub const EQ_BAND_PRESENCE: &str =
    "调节 3kHz 清晰度。正值增强辅音和可懂度，负值柔化刺耳。范围 -12~+12 dB";
pub const EQ_BAND_SIBILANCE: &str =
    "调节 6~8kHz 齿音。负值减轻刺耳的 S/T 音，正值增加细节。范围 -12~+12 dB";
pub const EQ_BAND_AIR: &str = "调节 12kHz 空气感。正值带来通透度，负值减少嘶嘶声。范围 -12~+12 dB";

pub const EQ_RESET_BANDS: &str = "将所有频段增益重置为 0 dB，恢复预设默认参数。";

pub const START_BUTTON: &str = "开始实时音频处理，捕获麦克风后进行降噪与 EQ 输出。";
pub const STOP_BUTTON: &str = "停止处理并保存录音，会生成源录音/降噪后/最终输出三份文件。";
pub const EXIT_BUTTON: &str = "退出应用。未保存的录音将丢失。";

pub const MUTE_PLAYBACK: &str =
    "播放静音开关\n\n开启后：\n✓ 麦克风继续录音\n✓ 降噪与 EQ 继续处理\n✓ 录音和频谱照常更新\n✗ 不再播放到扬声器\n\n适用于使用扬声器播放测试音频、防止程序输出干扰的场景。";

pub const EQ_PARAM_GAIN: &str =
    "增益 (静态)\n直接提升或削弱该频段的音量。\n• 正值: 增强\n• 负值: 削减\n• 范围: -12 ~ +12 dB";

pub const EQ_PARAM_FREQUENCY: &str =
    "中心频率\n决定本段着重处理的频率位置。\n• 低频: 50-200 Hz\n• 中频: 500-2000 Hz\n• 高频: 4k-12k Hz\n• 范围: 20 ~ 20000 Hz";

pub const EQ_PARAM_THRESHOLD: &str =
    "阈值 (触发点)\n只有当该频段声音超过此电平时才触发动态处理。阈值越高越容易触发，越低越难触发。";

pub const EQ_PARAM_RATIO: &str =
    "比率 (力度)\n控制超过阈值后压缩/扩展的力度。\n• 4:1 表示超出的部分只保留 1/4\n• 数字越大处理越强。";

pub const EQ_PARAM_Q: &str =
    "Q 值 (带宽)\n决定处理范围宽窄。\n• 小 Q: 自然、范围广\n• 大 Q: 精准、范围窄\n• 范围: 0.1 ~ 5.0";

pub const EQ_PARAM_DETECTOR_Q: &str =
    "检测器 Q\n决定检测器监听的范围，一般与 Q 值相近。较大的检测 Q 让检测更精准。";

pub const EQ_PARAM_MAX_GAIN: &str =
    "最大动态增益\n限制动态 EQ 最多能提升/削弱的 dB 数，避免过度处理。";

pub const EQ_PARAM_ATTACK: &str =
    "起音时间 (Attack)\n触发后多快开始处理。\n• 快: 1-20ms 捕捉瞬态\n• 中: 30-60ms 平衡\n• 慢: 80-100ms 更柔和";

pub const EQ_PARAM_RELEASE: &str =
    "释放时间 (Release)\n处理结束后多快恢复。\n• 快: 50-120ms\n• 中: 150-250ms\n• 慢: 300-500ms";

pub const EQ_PARAM_MAKEUP: &str = "补偿增益\n在动态处理后额外补偿的固定增益，用于平衡整体响度。";

pub const EQ_PARAM_MODE: &str =
    "动态模式\n• 下行压缩: 声音太响时压低 (控制浑浊/齿音)\n• 上行扩展: 声音太弱时拉高 (增强清晰/空气感)";

pub const EQ_PARAM_FILTER: &str =
    "滤波器类型\n• 峰值: 只影响中心附近频率 (中频处理)\n• 低搁架: 影响低频整段 (低频清理)\n• 高搁架: 影响高频整段 (空气感)";
