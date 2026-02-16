use std::any::Any;
use bincode::{Decode, Encode};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use crate::{Asset, AssetUrl};

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Encode, Decode)]
#[builder(setter(into))]
pub struct Material {
    #[serde(skip)]
    pub url: AssetUrl,
    #[builder(default = [1., 0., 1., 1.])]
    pub base_color: [f32; 4],
    #[builder(default = 1.0)]
    pub metallic: f32,
    #[builder(default = 0.5)]
    pub roughness: f32,
    #[builder(default = [0., 0., 0.])]
    pub emissive: [f32; 3],

    // Texture assets referenced by URL (baked separately as `.tex`).
    #[builder(default)]
    #[bincode(with_serde)]
    pub base_color_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub mra_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub normal_tex: Option<AssetUrl>,
    #[builder(default)]
    #[bincode(with_serde)]
    pub emissive_tex: Option<AssetUrl>,
}

impl Asset for Material {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self) -> &AssetUrl { &self.url }

    fn extension() -> &'static str {
        "mat"
    }
}
