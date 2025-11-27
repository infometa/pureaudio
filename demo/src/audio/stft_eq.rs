use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const DEFAULT_WINDOW: usize = 2048;
const DEFAULT_HOP: usize = 512;

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
        let gain_curve = build_gain_curve(sample_rate as f32, n_bins, 2.0, 3.0, 0.0);
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
        sample_rate: usize,
    ) {
        self.hf_gain_db = hf_gain_db;
        self.air_gain_db = air_gain_db;
        self.tilt_db = tilt_db;
        let n_bins = self.r2c.make_output_vec().len();
        self.gain_curve =
            build_gain_curve(sample_rate as f32, n_bins, hf_gain_db, air_gain_db, tilt_db);
    }

    /// In-place 处理一块音频；内部保持流式重叠相加，输出长度与输入相同
    pub fn process_block(&mut self, input: &mut [f32]) {
        if input.is_empty() {
            return;
        }
        let n = input.len();
        self.input_buf.extend_from_slice(input);
        if self.pending_output.len() < self.input_buf.len() + self.win {
            self.pending_output.resize(self.input_buf.len() + self.win, 0.0);
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
        let norm = 1.0 / self.win as f32;
        for v in self.time_buf.iter_mut() {
            *v *= norm;
        }
        for (i, v) in self.time_buf.iter().enumerate() {
            if offset + i >= self.pending_output.len() {
                self.pending_output.push(*v);
            } else {
                self.pending_output[offset + i] += *v;
            }
        }
    }
}

fn build_gain_curve(
    sr: f32,
    bins: usize,
    hf_gain_db: f32,
    air_gain_db: f32,
    tilt_db: f32,
) -> Vec<f32> {
    // 静态锚点（Hz, dB），描述目标音色基线
    const ANCHORS: &[(f32, f32)] = &[
        (100.0, 1.2),
        (180.0, 0.5),
        (300.0, -0.3),
        (1500.0, 0.0),
        (3500.0, 0.8),
        (6000.0, 1.5),
        (8000.0, 2.2),
        (12000.0, 3.0),
        (16000.0, 3.5),
        (18000.0, 4.0),
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
