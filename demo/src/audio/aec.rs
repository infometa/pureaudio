use log::{error, warn};
use std::time::Instant;
use webrtc_audio_processing::{
    Config, EchoCancellation, EchoCancellationSuppressionLevel, InitializationConfig, Processor,
};

/// 简易 AEC3 包装（基于 WebRTC Audio Processing），默认关闭，可按需启用。
pub struct EchoCanceller {
    processor: Option<Processor>,
    frame_size: usize,
    scratch: Vec<f32>,
    enabled: bool,
    active: bool,
    delay_ms: i32,
    delay_agnostic: bool,
    internal_highpass: bool,
    aggressive_base: bool,
    double_talk: bool,
    dt_exit_frames: u16,
    // 优化1.3：自适应过渡时间
    dt_start_time: Option<Instant>,  // 双讲开始时间
    dt_duration_ms: u32,              // 双讲持续时间（毫秒）
    // 配置节流：缓存上一次真正下发给 WebRTC 的参数
    last_suppression: Option<EchoCancellationSuppressionLevel>,
    last_delay_agnostic: bool,
    last_internal_highpass: bool,
    last_aggressive_base: bool,
    last_delay_ms: i32,
}

impl EchoCanceller {
    pub fn new(sample_rate: f32, hop_size: usize, delay_ms: i32) -> Self {
        let frame_size = (sample_rate / 100.0).round() as usize; // 10ms 帧
        let frame_ok = frame_size > 0 && hop_size % frame_size == 0;
        let init = InitializationConfig {
            num_capture_channels: 1,
            num_render_channels: 1,
            ..InitializationConfig::default()
        };
        // 默认使用强力模式以消除回声残留，双讲时会自动切换到温和模式保护近端
        let aggressive = true;
        let processor = if frame_ok {
            match Processor::new(&init) {
                Ok(p) => {
                    let mut aec = Self {
                        processor: Some(p),
                        frame_size,
                        scratch: vec![0.0; frame_size],
                        enabled: frame_ok,
                        active: frame_ok,
                        delay_ms,
                        delay_agnostic: true,
                        internal_highpass: false, // 外部已有高通，默认关闭 WebRTC 内置高通避免双高通
                        aggressive_base: aggressive,
                        double_talk: false,
                        dt_exit_frames: 0,
                        dt_start_time: None,
                        dt_duration_ms: 0,
                        last_suppression: None,
                        last_delay_agnostic: true,
                        last_internal_highpass: false,
                        last_aggressive_base: aggressive,
                        last_delay_ms: delay_ms.max(0),
                    };
                    aec.apply_config();
                    aec.processor
                }
                Err(e) => {
                    error!("初始化 AEC3 失败，旁路: {e}");
                    None
                }
            }
        } else {
            warn!(
                "AEC3 旁路：DF hop_size={} 与 10ms 帧长({} samples)不匹配",
                hop_size, frame_size
            );
            None
        };
        let active = frame_ok && processor.is_some();
        Self {
            processor,
            frame_size,
            scratch: vec![0.0; frame_size],
            enabled: frame_ok,
            active,
            delay_ms,
            delay_agnostic: true,
            internal_highpass: false,
            aggressive_base: aggressive,
            double_talk: false,
            dt_exit_frames: 0,
            dt_start_time: None,
            dt_duration_ms: 0,
            last_suppression: None,
            last_delay_agnostic: true,
            last_internal_highpass: false,
            last_aggressive_base: aggressive,
            last_delay_ms: delay_ms.max(0),
        }
    }

    fn apply_config(&mut self) {
        let Some(proc) = self.processor.as_mut() else { return; };
        
        // 优化1.2：三档渐进式抑制策略
        // 
        // 状态机：
        // 1. 双讲中 → Low suppression（最弱抑制，全力保护近端）
        // 2. 过渡期 → Moderate suppression（中等抑制，平滑过渡）
        // 3. 单讲 → High suppression（最强抑制，全力消除回声）
        // 
        // 优势：
        // - 避免 High ↔ Low 直接切换的咔嗒声
        // - 过渡期提供平滑的抑制级别变化
        // - 减少误判造成的音质瑕疵
        let suppression = if self.double_talk {
            // 双讲中：最弱抑制，保护近端语音
            EchoCancellationSuppressionLevel::Low
        } else if self.dt_exit_frames > 0 {
            // 过渡期：中等抑制，平滑过渡
            EchoCancellationSuppressionLevel::Moderate
        } else if self.aggressive_base {
            // 单讲：最强抑制，消除回声
            EchoCancellationSuppressionLevel::High
        } else {
            // 备用：中等抑制
            EchoCancellationSuppressionLevel::Moderate
        };
        let delay_ms = self.delay_ms.max(0);

        // === 配置节流 ===
        // 只有在 suppression / delay 模式 / 高通开关 / aggressive / delay_ms 真正变化时才下发配置，
        // 避免每帧 set_config 扰动 AEC3 内部滤波器并浪费 CPU。
        if self.last_suppression == Some(suppression)
            && self.last_delay_agnostic == self.delay_agnostic
            && self.last_internal_highpass == self.internal_highpass
            && self.last_aggressive_base == self.aggressive_base
            && self.last_delay_ms == delay_ms
        {
            return;
        }

        // 配置变化时记录日志
        log::debug!(
            "AEC配置更新: suppression={:?}, delay={}ms, delay_agnostic={}, internal_hp={}, double_talk={}, exit_frames={}, dt_duration={}ms",
            suppression,
            delay_ms,
            self.delay_agnostic,
            self.internal_highpass,
            self.double_talk,
            self.dt_exit_frames,
            self.dt_duration_ms
        );

        // 提供初始延迟提示能显著缩短 AEC3 初始收敛时间；
        // delay_agnostic=true 时该值仅作为“初始对齐提示”，后续仍由 AEC3 自适应。
        let delay_hint = if delay_ms > 0 { Some(delay_ms) } else { None };
        let ec = EchoCancellation {
            suppression_level: suppression,
            stream_delay_ms: delay_hint,
            enable_delay_agnostic: self.delay_agnostic,  // 自适应/固定延迟模式切换
            enable_extended_filter: true, // 启用扩展滤波，增强对复杂回声路径的处理
        };
        let cfg = Config { 
            echo_cancellation: Some(ec), 
            enable_high_pass_filter: self.internal_highpass,  // 避免双高通
            ..Config::default() 
        };
        proc.set_config(cfg);

        self.last_suppression = Some(suppression);
        self.last_delay_agnostic = self.delay_agnostic;
        self.last_internal_highpass = self.internal_highpass;
        self.last_aggressive_base = self.aggressive_base;
        self.last_delay_ms = delay_ms;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.processor.is_some();
        self.active = self.enabled;
        if enabled && self.processor.is_none() {
            warn!("AEC3 启用失败：处理器不可用（帧长不匹配或初始化失败）");
        }
    }

    pub fn set_delay_ms(&mut self, delay_ms: i32) {
        self.delay_ms = delay_ms.max(0);
        // delay_agnostic 模式下 delay_ms 仅作为初始提示，不频繁重配
        if !self.delay_agnostic {
            self.apply_config();
        }
    }

    pub fn set_aggressive(&mut self, aggressive: bool) {
        self.aggressive_base = aggressive;
        self.apply_config();
    }

    /// 切换延迟模式：true=自适应延迟（默认），false=固定延迟
    pub fn set_delay_agnostic(&mut self, enabled: bool) {
        self.delay_agnostic = enabled;
        self.apply_config();
    }

    /// 控制 WebRTC 内置高通开关（外部已高通时建议关闭）
    pub fn set_internal_highpass(&mut self, enabled: bool) {
        self.internal_highpass = enabled;
        self.apply_config();
    }

    /// 双讲检测时调用：降低抑制档位，保护近端语音
    /// 
    /// 参数:
    /// - active: true = 检测到双讲（近端+远端同时说话），使用Low suppression保护近端
    ///          false = 单讲或静音，使用High suppression消除回声
    pub fn set_double_talk(&mut self, active: bool) {
        let state_changed = self.double_talk != active;
        let mut config_dirty = false;

        if state_changed {
            log::debug!(
                "AEC双讲状态切换: {} -> {}",
                if self.double_talk { "双讲" } else { "单讲" },
                if active { "双讲" } else { "单讲" }
            );

            if active {
                // 进入双讲：记录开始时间并清空过渡计数
                self.dt_start_time = Some(Instant::now());
                self.dt_exit_frames = 0;
            } else if self.double_talk {
                // 退出双讲：计算持续时间并自适应设置过渡期帧数
                if let Some(start) = self.dt_start_time {
                    self.dt_duration_ms = start.elapsed().as_millis() as u32;
                    self.dt_exit_frames = if self.dt_duration_ms < 200 {
                        15
                    } else if self.dt_duration_ms < 1000 {
                        25
                    } else {
                        35
                    };
                    log::debug!(
                        "AEC双讲退出: 持续{}ms → 过渡期{}帧({}ms)",
                        self.dt_duration_ms,
                        self.dt_exit_frames,
                        self.dt_exit_frames * 10
                    );
                } else {
                    self.dt_exit_frames = 15;
                }
                self.dt_start_time = None;
            }

            self.double_talk = active;
            config_dirty = true;
        }

        // 过渡期倒计时：只有在过渡真正结束那一刻才更新配置
        if !self.double_talk && self.dt_exit_frames > 0 {
            let prev = self.dt_exit_frames;
            self.dt_exit_frames = self.dt_exit_frames.saturating_sub(1);
            if prev == 1 {
                log::debug!("AEC过渡期结束，恢复强力抑制");
                config_dirty = true;
            }
        }

        if config_dirty {
            self.apply_config();
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
    
    /// 获取当前双讲状态（用于调试和监控）
    pub fn is_double_talk(&self) -> bool {
        self.double_talk
    }
    
    /// 获取当前延迟（ms）
    pub fn get_delay_ms(&self) -> i32 {
        self.delay_ms
    }
    
    /// 获取诊断信息字符串
    pub fn get_diagnostics(&self) -> String {
        format!(
            "enabled={}, active={}, delay={}ms, aggressive={}, double_talk={}, exit_frames={}, dt_duration={}ms",
            self.enabled, self.active, self.delay_ms,
            self.aggressive_base, self.double_talk, self.dt_exit_frames, self.dt_duration_ms
        )
    }

    pub fn process_capture(&mut self, buf: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let Some(proc) = self.processor.as_mut() else { return; };
        let frame = self.frame_size;
        for chunk in buf.chunks_mut(frame) {
            if chunk.len() == frame {
                if let Err(e) = proc.process_capture_frame(chunk) {
                    warn!("AEC3 捕获处理失败: {e}");
                }
            } else {
                // pad 短帧
                self.scratch[..chunk.len()].copy_from_slice(chunk);
                self.scratch[chunk.len()..frame].fill(0.0);
                if let Err(e) = proc.process_capture_frame(&mut self.scratch[..]) {
                    warn!("AEC3 捕获处理失败: {e}");
                }
                chunk.copy_from_slice(&self.scratch[..chunk.len()]);
            }
        }
    }

    pub fn process_render(&mut self, buf: &[f32]) {
        if !self.enabled {
            return;
        }
        let Some(proc) = self.processor.as_mut() else { return; };
        let frame = self.frame_size;
        for chunk in buf.chunks(frame) {
            if chunk.len() == frame {
                // 使用内部 scratch，避免每帧分配/拷贝
                self.scratch.copy_from_slice(chunk);
                if let Err(e) = proc.process_render_frame(&mut self.scratch[..frame]) {
                    warn!("AEC3 渲染参考处理失败: {e}");
                }
            } else {
                self.scratch[..chunk.len()].copy_from_slice(chunk);
                self.scratch[chunk.len()..frame].fill(0.0);
                if let Err(e) = proc.process_render_frame(&mut self.scratch[..frame]) {
                    warn!("AEC3 渲染参考处理失败: {e}");
                }
            }
        }
    }
}
