use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScenePreset {
    Broadcast,
    OpenOffice,
    ConferenceHall,
    OpenOfficeMeeting,
}

impl ScenePreset {
    pub const fn all() -> [Self; 4] {
        [
            Self::Broadcast,
            Self::OpenOffice,
            Self::ConferenceHall,
            Self::OpenOfficeMeeting,
        ]
    }
}

impl Default for ScenePreset {
    fn default() -> Self {
        ScenePreset::OpenOfficeMeeting
    }
}

impl std::fmt::Display for ScenePreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ScenePreset::Broadcast => "播音级",
            ScenePreset::OpenOffice => "开放办公区",
            ScenePreset::ConferenceHall => "会议室去混响",
            ScenePreset::OpenOfficeMeeting => "开放办公会议",
        };
        write!(f, "{name}")
    }
}
