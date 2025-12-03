use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

// 窗口与 DFN 对齐：单主干、无并行；512/256 Hann
const DEFAULT_WINDOW: usize = 512;
const DEFAULT_HOP: usize = DEFAULT_WINDOW / 2;
const STFT_BANDS: [(f32, f32); 8] = [
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1500.0),
    (1500.0, 3000.0),
    (3000.0, 6000.0),
    (6000.0, 10000.0),
    (10000.0, 14000.0),
];

fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

fn hann_window(n: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n);
    let denom = (n - 1) as f32;
    for i in 0..n {
        let t = std::f32::consts::PI * 2.0 * (i as f32) / denom;
        w.push(0.5 * (1.0 - t.cos()));
    }
    w
}

/// 频域静态 EQ：固定曲线补偿高频/空气感 + 轻微倾斜
pub struct StftStaticEq {
    win: usize,
    hop: usize,
    window: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    spectrum: Vec<Complex32>,
    scratch_f: Vec<Complex32>,
    scratch_b: Vec<Complex32>,
    time_buf: Vec<f32>,
    gain_curve: Vec<f32>,
    band_offsets: [f32; 8],
    input_buf: Vec<f32>,
    pending_output: Vec<f32>,
    hf_gain_db: f32,
    air_gain_db: f32,
    tilt_db: f32,
}

impl StftStaticEq {
    pub fn new(sample_rate: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(DEFAULT_WINDOW);
        let c2r = planner.plan_fft_inverse(DEFAULT_WINDOW);
        let window = hann_window(DEFAULT_WINDOW);
        let n_bins = r2c.make_output_vec().len();
        let offsets = [0.0; 8];
        let gain_curve = build_gain_curve(sample_rate as f32, n_bins, 2.0, 3.0, 0.0, &offsets);
        Self {
            win: DEFAULT_WINDOW,
            hop: DEFAULT_HOP,
            window,
            spectrum: r2c.make_output_vec(),
            time_buf: r2c.make_input_vec(),
            scratch_f: r2c.make_scratch_vec(),
            scratch_b: c2r.make_scratch_vec(),
            r2c,
            c2r,
            gain_curve,
            band_offsets: offsets,
            input_buf: Vec::new(),
            pending_output: Vec::new(),
            hf_gain_db: 2.0,
            air_gain_db: 3.0,
            tilt_db: 0.0,
        }
    }

    pub fn set_curve(
        &mut self,
        hf_gain_db: f32,
        air_gain_db: f32,
        tilt_db: f32,
        band_offsets: &[f32; 8],
        sample_rate: usize,
    ) {
        self.hf_gain_db = hf_gain_db;
        self.air_gain_db = air_gain_db;
        self.tilt_db = tilt_db;
        self.band_offsets = *band_offsets;
        let n_bins = self.r2c.make_output_vec().len();
        self.gain_curve = build_gain_curve(
            sample_rate as f32,
            n_bins,
            hf_gain_db,
            air_gain_db,
            tilt_db,
            band_offsets,
        );
    }

    /// In-place 处理一块音频；内部保持流式重叠相加，输出长度与输入相同
    pub fn process_block(&mut self, input: &mut [f32]) {
        if input.is_empty() {
            return;
        }
        let n = input.len();
        self.input_buf.extend_from_slice(input);
        let needed = self.input_buf.len() + self.win * 3;
        if self.pending_output.len() < needed {
            self.pending_output.resize(needed, 0.0);
        }
        let mut frame_start = 0usize;
        // 只要凑齐一个 hop，就执行一次 STFT 帧
        while self.input_buf.len() >= frame_start + self.win {
            let frame = self.input_buf[frame_start..frame_start + self.win].to_vec();
            self.process_frame(&frame, frame_start);
            frame_start += self.hop;
        }
        // 输出当前块
        input.copy_from_slice(&self.pending_output[..n]);
        // 滚动缓冲
        self.pending_output.drain(..n);
        // 保留最近 win-hop 的输入作为下一帧上下文
        let drain = frame_start.min(self.input_buf.len());
        self.input_buf.drain(..drain);
    }

    fn process_frame(&mut self, frame: &[f32], offset: usize) {
        debug_assert_eq!(frame.len(), self.win);
        self.time_buf.copy_from_slice(frame);
        for (x, w) in self.time_buf.iter_mut().zip(self.window.iter()) {
            *x *= *w;
        }
        if let Err(err) = self.r2c.process_with_scratch(
            &mut self.time_buf,
            &mut self.spectrum,
            &mut self.scratch_f,
        ) {
            log::warn!("STFT EQ forward FFT failed: {:?}", err);
            return;
        }
        let n_bins = self.spectrum.len();
        for i in 0..n_bins {
            self.spectrum[i] *= self.gain_curve.get(i).copied().unwrap_or(1.0);
        }
        if let Err(err) = self.c2r.process_with_scratch(
            &mut self.spectrum,
            &mut self.time_buf,
            &mut self.scratch_b,
        ) {
            log::warn!("STFT EQ inverse FFT failed: {:?}", err);
            return;
        }
        // Hann + 50% overlap → 能量校正系数 2/N
        let norm = 2.0 / self.win as f32;
        for v in self.time_buf.iter_mut() {
            *v *= norm;
        }
        let needed = offset + self.win;
        if self.pending_output.len() < needed {
            self.pending_output.resize(needed, 0.0);
        }
        for (i, v) in self.time_buf.iter().enumerate() {
            self.pending_output[offset + i] += *v;
        }
    }
}

fn build_gain_curve(
    sr: f32,
    bins: usize,
    hf_gain_db: f32,
    air_gain_db: f32,
    tilt_db: f32,
    band_offsets: &[f32; 8],
) -> Vec<f32> {
    // 静态锚点（Hz, dB），按 8 段宽 Q 基线定义：低频轻减，高频轻补偿，保证“厚/自然/空气”
    const ANCHORS: &[(f32, f32)] = &[
        (60.0, -1.5),    // rumble/boom 轻减
        (120.0, -1.0),
        (200.0, -0.8),   // 胸腔共鸣轻减
        (350.0, -0.3),   // 厚度控制
        (700.0, 0.2),    // 中频密度
        (1500.0, 0.5),   // 主体保持
        (2500.0, 1.2),   // intelligibility
        (4000.0, 1.0),   // presence 微调
        (6000.0, 1.5),   // bright/air 起点
        (8000.0, 2.0),
        (10000.0, 2.3),
        (12500.0, 2.8),
        (14000.0, 1.5),  // 超高频收尾，避免过尖
    ];
    let mut gains = Vec::with_capacity(bins);
    let bin_hz = sr / (bins.saturating_sub(1) as f32 * 2.0);
    for i in 0..bins {
        let f = i as f32 * bin_hz;
        // 基线插值
        let mut db = if f <= ANCHORS[0].0 {
            ANCHORS[0].1
        } else if f >= ANCHORS[ANCHORS.len() - 1].0 {
            ANCHORS[ANCHORS.len() - 1].1
        } else {
            let mut v = ANCHORS[0].1;
            for w in ANCHORS.windows(2) {
                let (f0, g0) = (w[0].0, w[0].1);
                let (f1, g1) = (w[1].0, w[1].1);
                if f >= f0 && f <= f1 {
                    let t = ((f - f0) / (f1 - f0)).clamp(0.0, 1.0);
                    v = g0 + t * (g1 - g0);
                    break;
                }
            }
            v
        };
        // 分段偏移（8 段宽 Q）：落在哪个区间加对应偏置
        for (idx, (low, high)) in STFT_BANDS.iter().enumerate() {
            if f >= *low && f < *high {
                db += band_offsets[idx];
                break;
            }
        }
        // UI 旋钮的附加补偿：HF/Air + 倾斜，保持可微调
        if f >= 6000.0 && f < 12000.0 {
            let t = ((f - 6000.0) / 6000.0).clamp(0.0, 1.0);
            db += t * hf_gain_db;
        } else if f >= 12000.0 {
            let t = ((f - 12000.0) / 6000.0).clamp(0.0, 1.0);
            db += air_gain_db * (0.6 + 0.4 * t);
        }
        if tilt_db.abs() > 0.01 {
            let t = (f / (sr * 0.5)).clamp(0.0, 1.0);
            db += (t * 2.0 - 1.0) * tilt_db;
        }
        db = db.clamp(-6.0, 8.0);
        gains.push(db_to_linear(db));
    }
    gains
}
