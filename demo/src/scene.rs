use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScenePreset {
    Broadcast,
    OpenOffice,
    ConferenceHall,
}

impl ScenePreset {
    pub const fn all() -> [Self; 3] {
        [Self::Broadcast, Self::OpenOffice, Self::ConferenceHall]
    }
}

impl Default for ScenePreset {
    fn default() -> Self {
        ScenePreset::Broadcast
    }
}

impl std::fmt::Display for ScenePreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ScenePreset::Broadcast => "播音级",
            ScenePreset::OpenOffice => "开放办公区",
            ScenePreset::ConferenceHall => "会议室去混响",
        };
        write!(f, "{name}")
    }
}
