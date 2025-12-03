pub fn smoothing_coeff(time_ms: f32, block_len: usize, sample_rate: f32) -> f32 {
    if block_len == 0 {
        return 1.0;
    }
    let sr = sample_rate.max(1.0);
    let block_time = block_len as f32 / sr;
    let time_seconds = (time_ms / 1000.0).max(1.0 / sr);
    1.0 - (-block_time / time_seconds).exp()
}

#[derive(Debug, Clone)]
pub struct EnvelopeDetector {
    attack_ms: f32,
    release_ms: f32,
    sample_rate: f32,
    value_db: f32,
    initialized: bool,
}

impl EnvelopeDetector {
    pub fn new(sample_rate: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_ms,
            release_ms,
            sample_rate,
            value_db: -80.0,
            initialized: false,
        }
    }

    pub fn reset(&mut self, value_db: f32) {
        self.value_db = value_db;
        self.initialized = true;
    }

    pub fn process(&mut self, input_db: f32, block_len: usize) -> f32 {
        if !self.initialized {
            self.value_db = input_db;
            self.initialized = true;
            return self.value_db;
        }
        let coeff = if input_db > self.value_db {
            smoothing_coeff(self.attack_ms, block_len, self.sample_rate)
        } else {
            smoothing_coeff(self.release_ms, block_len, self.sample_rate)
        };
        self.value_db += coeff * (input_db - self.value_db);
        self.value_db
    }

    pub fn value_db(&self) -> f32 {
        self.value_db
    }
}
