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
    aggressive_base: bool,
    double_talk: bool,
    dt_exit_frames: u16,
    // 优化1.3：自适应过渡时间
    dt_start_time: Option<Instant>,  // 双讲开始时间
    dt_duration_ms: u32,              // 双讲持续时间（毫秒）
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
                        aggressive_base: aggressive,
                        double_talk: false,
                        dt_exit_frames: 0,
                        dt_start_time: None,
                        dt_duration_ms: 0,
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
            aggressive_base: aggressive,
            double_talk: false,
            dt_exit_frames: 0,
            dt_start_time: None,
            dt_duration_ms: 0,
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
        
        // 每次配置变化时记录日志（限流：仅在状态变化时）
        log::debug!(
            "AEC配置: suppression={:?}, delay={}ms, double_talk={}, exit_frames={}, dt_duration={}ms",
            suppression, self.delay_ms, self.double_talk, self.dt_exit_frames, self.dt_duration_ms
        );
        
        let ec = EchoCancellation {
            suppression_level: suppression,
            stream_delay_ms: None,  // 不手动设置，完全依赖 delay_agnostic 自动估计
            enable_delay_agnostic: true,  // 自适应延迟估计
            enable_extended_filter: true, // 启用扩展滤波，增强对复杂回声路径的处理
        };
        let cfg = Config { 
            echo_cancellation: Some(ec), 
            enable_high_pass_filter: true,  // 启用内置高通
            ..Config::default() 
        };
        proc.set_config(cfg);
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
        self.apply_config();
    }

    pub fn set_aggressive(&mut self, aggressive: bool) {
        self.aggressive_base = aggressive;
        self.apply_config();
    }

    /// 双讲检测时调用：降低抑制档位，保护近端语音
    /// 
    /// 参数:
    /// - active: true = 检测到双讲（近端+远端同时说话），使用Low suppression保护近端
    ///          false = 单讲或静音，使用High suppression消除回声
    pub fn set_double_talk(&mut self, active: bool) {
        let state_changed = self.double_talk != active;
        
        if state_changed {
            log::debug!(
                "AEC双讲状态切换: {} -> {}",
                if self.double_talk { "双讲" } else { "单讲" },
                if active { "双讲" } else { "单讲" }
            );
            
            if active {
                // 优化1.3：进入双讲，记录开始时间
                self.dt_start_time = Some(Instant::now());
            } else if self.double_talk {
                // 优化1.3：退出双讲，计算持续时间并自适应调整过渡期
                if let Some(start) = self.dt_start_time {
                    self.dt_duration_ms = start.elapsed().as_millis() as u32;
                    
                    // 自适应过渡期策略（增强版，配合滞后保护使用）：
                    // - 短暂打断（<200ms）：标准过渡 → 15帧(150ms)
                    // - 正常对话（200-1000ms）：较长过渡 → 25帧(250ms)
                    // - 长时间双讲（>1000ms）：最长过渡 → 35帧(350ms)
                    // 
                    // 原理：
                    // 1. 短暂打断需要合理过渡，不要快速恢复
                    // 2. 正常对话需要更长的平滑过渡
                    // 3. 长对话后更谨慎，避免吞尾音
                    self.dt_exit_frames = if self.dt_duration_ms < 200 {
                        15  // 标准过渡
                    } else if self.dt_duration_ms < 1000 {
                        25  // 较长过渡
                    } else {
                        35  // 最长过渡
                    };
                    
                    log::debug!(
                        "AEC双讲退出: 持续{}ms → 过渡期{}帧({}ms)",
                        self.dt_duration_ms,
                        self.dt_exit_frames,
                        self.dt_exit_frames * 10
                    );
                } else {
                    // 没有记录开始时间，使用默认值
                    self.dt_exit_frames = 15;
                }
                self.dt_start_time = None;
            }
            
            self.double_talk = active;
        }
        
        // 过渡期倒计时
        if !self.double_talk && self.dt_exit_frames > 0 {
            self.dt_exit_frames = self.dt_exit_frames.saturating_sub(1);
            if self.dt_exit_frames == 0 {
                log::debug!("AEC过渡期结束，恢复强力抑制");
            }
        }
        
        // 状态变化或过渡期结束时，更新配置
        if state_changed || (self.dt_exit_frames == 0 && !self.double_talk) {
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
