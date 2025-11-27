use super::eq::{Biquad, BiquadType};

/// 轻量谱倾斜补偿器：比较 DF 前后高/低频能量比，必要时上架提升高频
pub struct SpectralTiltCompensator {
    pre_lp: Biquad,
    pre_hp: Biquad,
    post_lp: Biquad,
    post_hp: Biquad,
    shelf: Biquad,
    pre_ratio: f32,
    post_ratio: f32,
    current_boost: f32,
}

impl SpectralTiltCompensator {
    pub fn new(sample_rate: f32) -> Self {
        // 分析滤波器：低频 <1.2k，高频 >4k，补偿点 3.5k 高搁架
        let lp_cut = 1200.0;
        let hp_cut = 4000.0;
        let shelf_freq = 3500.0;
        Self {
            pre_lp: Biquad::new(BiquadType::LowPass, sample_rate, lp_cut, 0.707, 0.0),
            pre_hp: Biquad::new(BiquadType::HighPass, sample_rate, hp_cut, 0.707, 0.0),
            post_lp: Biquad::new(BiquadType::LowPass, sample_rate, lp_cut, 0.707, 0.0),
            post_hp: Biquad::new(BiquadType::HighPass, sample_rate, hp_cut, 0.707, 0.0),
            shelf: Biquad::new(BiquadType::HighShelf, sample_rate, shelf_freq, 0.707, 0.0),
            pre_ratio: 1.0,
            post_ratio: 1.0,
            current_boost: 0.0,
        }
    }

    /// 记录 DF 前的高低频比值
    pub fn observe_input(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let ratio = band_energy_ratio(&mut self.pre_lp, &mut self.pre_hp, samples);
        // 平滑避免抖动
        self.pre_ratio = smooth(self.pre_ratio, ratio, 0.9);
    }

    /// 在 DF 后应用补偿
    pub fn compensate(&mut self, samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        let ratio = band_energy_ratio(&mut self.post_lp, &mut self.post_hp, samples);
        self.post_ratio = smooth(self.post_ratio, ratio, 0.9);

        let mut target_boost = 0.0;
        if self.pre_ratio > 1e-6 && self.post_ratio > 1e-6 {
            let drop = self.pre_ratio / self.post_ratio;
            if drop > 1.1 {
                // 转为 dB，限制最大补偿
                target_boost = (20.0 * drop.log10()).clamp(0.0, 6.0);
            }
        }
        self.current_boost = smooth(self.current_boost, target_boost, 0.85);

        if self.current_boost < 0.05 {
            // 防止残留增益
            if self.current_boost.abs() > f32::EPSILON {
                self.shelf.set_gain_db(0.0);
            }
            return;
        }
        self.shelf.set_gain_db(self.current_boost);
        for sample in samples.iter_mut() {
            *sample = self.shelf.process(*sample);
        }
    }
}

fn band_energy_ratio(lp: &mut Biquad, hp: &mut Biquad, samples: &[f32]) -> f32 {
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    for &s in samples {
        let l = lp.process(s);
        let h = hp.process(s);
        low += l * l;
        high += h * h;
    }
    let low = (low / samples.len().max(1) as f32).max(1e-9);
    let high = (high / samples.len().max(1) as f32).max(1e-9);
    high / low
}

fn smooth(current: f32, target: f32, alpha: f32) -> f32 {
    current + (target - current) * (1.0 - alpha.clamp(0.0, 1.0))
}
