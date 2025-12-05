use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScenePreset {
    ConferenceHall,
    OpenOfficeMeeting,
    OpenOfficeHeadset,
}

impl ScenePreset {
    pub const fn all() -> [Self; 3] {
        [
            Self::ConferenceHall,
            Self::OpenOfficeMeeting,
            Self::OpenOfficeHeadset,
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
            ScenePreset::ConferenceHall => "会议室",
            ScenePreset::OpenOfficeMeeting => "开放办公区",
            ScenePreset::OpenOfficeHeadset => "开放办公-耳机",
        };
        write!(f, "{name}")
    }
}
