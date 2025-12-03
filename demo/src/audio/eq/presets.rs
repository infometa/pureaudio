use super::dynamic_band::{BandMode, BandSettings, FilterKind};
use serde::{Deserialize, Serialize};

pub const MAX_EQ_BANDS: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct EqPreset {
    pub name: &'static str,
    pub bands: [BandSettings; MAX_EQ_BANDS],
    pub default_mix: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EqPresetKind {
    OpenOffice,
    ConferenceHall,
}

impl EqPresetKind {
    pub const fn all() -> [Self; 2] {
        [Self::OpenOffice, Self::ConferenceHall]
    }

    pub const fn preset(self) -> &'static EqPreset {
        match self {
            EqPresetKind::OpenOffice => &OPEN_OFFICE,
            EqPresetKind::ConferenceHall => &CONFERENCE_HALL,
        }
    }

    pub fn default_mix(self) -> f32 {
        self.preset().default_mix
    }

    pub fn display_name(self) -> &'static str {
        self.preset().name
    }

    pub fn description(self) -> &'static str {
        match self {
            EqPresetKind::OpenOffice => "开放办公区：抑制浑浊/穿透性人声，聚焦近讲清晰度。",
            EqPresetKind::ConferenceHall => "会议室：削驻波、补空气感，压尾音混响但保持自然度。",
        }
    }

    pub fn tooltip_text(self) -> &'static str {
        match self {
            EqPresetKind::OpenOffice => {
                "开放办公：100/200 提厚，450 动态去浑浊，3.5k/8k 提亮并提高可懂度。"
            }
            EqPresetKind::ConferenceHall => {
                "会议室：80/180 提基座，350 削驻波，3.5k/8k 补清晰与空气，模拟近讲。"
            }
        }
    }
}

impl Default for EqPresetKind {
    fn default() -> Self {
        EqPresetKind::OpenOffice
    }
}

impl std::fmt::Display for EqPresetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

const OPEN_OFFICE: EqPreset = EqPreset {
    name: "开放办公区",
    default_mix: 1.0,
    bands: [
        BandSettings {
            label: "低频基座",
            frequency_hz: 100.0,
            q: 0.7,
            detector_q: 0.7,
            threshold_db: -50.0,
            ratio: 1.0,
            max_gain_db: 6.0,
            attack_ms: 25.0,
            release_ms: 150.0,
            mode: BandMode::Upward,
            filter: FilterKind::LowShelf,
            makeup_db: 0.0,
            static_gain_db: 0.0,
        },
        BandSettings {
            label: "胸腔",
            frequency_hz: 200.0,
            q: 1.0,
            detector_q: 1.0,
            threshold_db: -50.0,
            ratio: 1.0,
            max_gain_db: 6.0,
            attack_ms: 25.0,
            release_ms: 150.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 2.0,
        },
        BandSettings {
            label: "去浑浊",
            frequency_hz: 450.0,
            q: 2.0,
            detector_q: 2.0,
            threshold_db: -30.0,
            ratio: 3.0,
            max_gain_db: 6.0,
            attack_ms: 25.0,
            release_ms: 200.0,
            mode: BandMode::Downward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: -6.0,
        },
        BandSettings {
            label: "清晰",
            frequency_hz: 3500.0,
            q: 1.2,
            detector_q: 1.2,
            threshold_db: -40.0,
            ratio: 1.5,
            max_gain_db: 6.0,
            attack_ms: 20.0,
            release_ms: 160.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 3.0,
        },
        BandSettings {
            label: "空气",
            frequency_hz: 8000.0,
            q: 0.9,
            detector_q: 0.9,
            threshold_db: -35.0,
            ratio: 1.5,
            max_gain_db: 6.0,
            attack_ms: 30.0,
            release_ms: 200.0,
            mode: BandMode::Upward,
            filter: FilterKind::HighShelf,
            makeup_db: 0.0,
            static_gain_db: 3.0,
        },
    ],
};

const CONFERENCE_HALL: EqPreset = EqPreset {
    name: "会议室",
    default_mix: 0.85,
    bands: [
        BandSettings {
            label: "低频基座",
            frequency_hz: 80.0,
            q: 0.7,
            detector_q: 0.7,
            threshold_db: -50.0,
            ratio: 1.0,
            max_gain_db: 6.0,
            attack_ms: 30.0,
            release_ms: 220.0,
            mode: BandMode::Upward,
            filter: FilterKind::LowShelf,
            makeup_db: 0.0,
            static_gain_db: 1.5,
        },
        BandSettings {
            label: "胸腔",
            frequency_hz: 180.0,
            q: 1.0,
            detector_q: 1.0,
            threshold_db: -50.0,
            ratio: 1.0,
            max_gain_db: 6.0,
            attack_ms: 30.0,
            release_ms: 220.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 3.5,
        },
        BandSettings {
            label: "驻波削减",
            frequency_hz: 350.0,
            q: 1.4,
            detector_q: 1.4,
            threshold_db: -28.0,
            ratio: 3.0,
            max_gain_db: 8.0,
            attack_ms: 25.0,
            release_ms: 220.0,
            mode: BandMode::Downward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: -5.0,
        },
        BandSettings {
            label: "清晰",
            frequency_hz: 3500.0,
            q: 1.2,
            detector_q: 1.2,
            threshold_db: -40.0,
            ratio: 1.5,
            max_gain_db: 6.0,
            attack_ms: 20.0,
            release_ms: 160.0,
            mode: BandMode::Upward,
            filter: FilterKind::Peak,
            makeup_db: 0.0,
            static_gain_db: 1.5,
        },
        BandSettings {
            label: "空气",
            frequency_hz: 8000.0,
            q: 0.8,
            detector_q: 0.8,
            threshold_db: -35.0,
            ratio: 1.5,
            max_gain_db: 8.0,
            attack_ms: 35.0,
            release_ms: 240.0,
            mode: BandMode::Upward,
            filter: FilterKind::HighShelf,
            makeup_db: 0.0,
            static_gain_db: 5.0,
        },
    ],
};
