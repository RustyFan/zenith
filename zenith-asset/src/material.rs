use std::any::Any;
use std::path::PathBuf;
use bincode::{Decode, Encode};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use crate::{Asset, AssetUrl};

#[derive(Debug, Clone, Builder, Serialize, Deserialize, Encode, Decode)]
#[builder(setter(into))]
pub struct Material {
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

    fn url(&self, name: &str) -> AssetUrl {
        let mut url = PathBuf::from(name);
        url.set_extension(Self::extension());
        url.into()
    }

    fn extension() -> &'static str {
        "mat"
    }

    fn gpu_size_in_bytes(&self) -> usize {
        size_of_val(&self.base_color) +
            size_of_val(&self.metallic) +
            size_of_val(&self.roughness) +
            size_of_val(&self.emissive)
    }
}
