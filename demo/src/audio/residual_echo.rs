use log::warn;
use realfft::{num_complex::Complex32, ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// 轻量 Residual Echo Suppressor（RES）
///
/// 设计目标：
/// - AEC3 后对残余回声做二次频域抑制
/// - 只在「远端活跃且非双讲」时工作，保护近端语音
/// - 预分配全部缓冲，满足实时安全
pub struct ResidualEchoSuppressor {
    fft_len: usize,
    hop_size: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    near_time: Vec<f32>,
    far_time: Vec<f32>,
    near_spec: Vec<Complex32>,
    far_spec: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    gains: Vec<f32>,
}

impl ResidualEchoSuppressor {
    pub fn new(_sample_rate: f32, hop_size: usize) -> Self {
        let fft_len = 512usize; // 10ms@48k=480，向上填充到 512 便于 FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_len);
        let c2r = planner.plan_fft_inverse(fft_len);
        let spec_len = fft_len / 2 + 1;
        let near_spec = r2c.make_output_vec();
        let far_spec = r2c.make_output_vec();
        let scratch_fwd = r2c.make_scratch_vec();
        let scratch_inv = c2r.make_scratch_vec();
        Self {
            fft_len,
            hop_size,
            r2c,
            c2r,
            near_time: vec![0.0; fft_len],
            far_time: vec![0.0; fft_len],
            near_spec,
            far_spec,
            scratch_fwd,
            scratch_inv,
            gains: vec![1.0; spec_len],
        }
    }

    /// 处理一帧近端信号（AEC 后）.
    ///
    /// - `near`: 近端时域帧（in-place）
    /// - `far`: 远端参考帧（与 near 同长度）
    /// - `far_active`: 远端是否真正有能量
    /// - `double_talk`: 是否双讲（双讲时旁路 RES）
    pub fn process(&mut self, near: &mut [f32], far: &[f32], far_active: bool, double_talk: bool) {
        if near.is_empty() {
            return;
        }

        if !far_active || double_talk {
            // 旁路时缓慢回到 1.0，避免忽然变化
            for g in self.gains.iter_mut() {
                *g = 0.9 * *g + 0.1;
            }
            return;
        }

        let n = near.len().min(self.hop_size).min(self.fft_len);
        let m = far.len().min(n);

        self.near_time[..n].copy_from_slice(&near[..n]);
        self.near_time[n..].fill(0.0);
        self.far_time[..m].copy_from_slice(&far[..m]);
        if m < self.fft_len {
            self.far_time[m..].fill(0.0);
        }

        if let Err(e) =
            self.r2c
                .process_with_scratch(&mut self.near_time, &mut self.near_spec, &mut self.scratch_fwd)
        {
            warn!("RES 前向 FFT 失败: {e}");
            return;
        }
        if let Err(e) =
            self.r2c
                .process_with_scratch(&mut self.far_time, &mut self.far_spec, &mut self.scratch_fwd)
        {
            warn!("RES 参考 FFT 失败: {e}");
            return;
        }

        const EPS: f32 = 1e-8;
        const ALPHA: f32 = 0.8; // 抑制强度
        const SMOOTH: f32 = 0.85; // 帧间平滑
        const MIN_GAIN: f32 = 0.18; // 最小保留，避免过度抽吸

        for (i, g_prev) in self.gains.iter_mut().enumerate() {
            let near_mag2 = self.near_spec[i].norm_sqr();
            let far_mag2 = self.far_spec[i].norm_sqr();
            let ratio = far_mag2 / (near_mag2 + EPS);

            // ratio 越大，越像残余回声 → gain 越小
            let target = (1.0 / (1.0 + ALPHA * ratio)).clamp(MIN_GAIN, 1.0);
            let g = SMOOTH * *g_prev + (1.0 - SMOOTH) * target;
            *g_prev = g;
            self.near_spec[i] *= g;
        }

        if let Err(e) =
            self.c2r
                .process_with_scratch(&mut self.near_spec, &mut self.near_time, &mut self.scratch_inv)
        {
            warn!("RES 逆向 FFT 失败: {e}");
            return;
        }

        // rustfft/realfft 逆变换不做 1/N 归一化
        let scale = 1.0 / self.fft_len as f32;
        for (dst, &src) in near.iter_mut().take(n).zip(self.near_time.iter().take(n)) {
            *dst = src * scale;
        }
    }
}
