use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use glam::Vec3;
use ispc_texcomp::{RgbaSurface};
use zenith_core::log;
use crate::{Asset, AssetBaker, AssetRegistry, RawAsset, AssetLoader, AssetUrl, serialize_asset, RawAssetType};
use crate::texture::{Texture, TextureBuilder, TextureFormat};

#[derive(Debug, Clone)]
pub struct HdrLoader;

impl HdrLoader {
    pub fn new() -> Self {
        Self
    }
}

/// Raw HDR data container (equirectangular RGB32F)
pub struct RawHdr {
    path: PathBuf,
    width: u32,
    height: u32,
    pixels: Vec<f32>,  // RGB32F data (3 floats per pixel)
}

impl RawAsset for RawHdr {
    #[inline(always)]
    fn raw_asset_type(&self) -> RawAssetType {
        RawAssetType::Hdr
    }
}

impl AssetLoader for HdrLoader {
    type Raw = RawHdr;

    #[profiling::function]
    fn load(path: &Path) -> Result<Self::Raw> {
        let img = image::open(path)?;
        let width = img.width();
        let height = img.height();

        // Convert to RGB32F
        let rgb_img = img.to_rgb32f();
        let pixels_flat: Vec<f32> = rgb_img.into_raw();

        log::info!("Loaded HDR image: {}x{} ({} pixels)", width, height, pixels_flat.len() / 3);

        Ok(RawHdr {
            path: path.to_owned(),
            width,
            height,
            pixels: pixels_flat,
        })
    }
}

pub struct RawHdrProcessor;

impl RawHdrProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Convert equirectangular HDR to 6 cubemap faces
    #[profiling::function]
    fn equirect_to_cubemap(
        equirect_pixels: &[f32],
        equirect_width: u32,
        equirect_height: u32,
        face_size: u32,
    ) -> [Vec<f32>; 6] {
        // 6 faces: +X, -X, +Y, -Y, +Z, -Z
        let mut faces = [
            vec![0.0; (face_size * face_size * 3) as usize],  // +X
            vec![0.0; (face_size * face_size * 3) as usize],  // -X
            vec![0.0; (face_size * face_size * 3) as usize],  // +Y
            vec![0.0; (face_size * face_size * 3) as usize],  // -Y
            vec![0.0; (face_size * face_size * 3) as usize],  // +Z
            vec![0.0; (face_size * face_size * 3) as usize],  // -Z
        ];

        // TODO: parallel this process
        for face_idx in 0..6 {
            log::info!("Converting cubemap face {}/6...", face_idx + 1);

            for y in 0..face_size {
                for x in 0..face_size {
                    let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;

                    let dir = Self::cubemap_direction(face_idx, u, v);
                    let (theta, phi) = Self::direction_to_spherical(&dir);
                    let equirect_u = phi / (2.0 * std::f32::consts::PI);
                    let equirect_v = theta / std::f32::consts::PI;

                    let color = Self::sample_equirect(
                        equirect_pixels,
                        equirect_width,
                        equirect_height,
                        equirect_u,
                        equirect_v,
                    );

                    // Write to face
                    let pixel_idx = ((y * face_size + x) * 3) as usize;
                    faces[face_idx][pixel_idx] = color[0];
                    faces[face_idx][pixel_idx + 1] = color[1];
                    faces[face_idx][pixel_idx + 2] = color[2];
                }
            }
        }

        faces
    }

    fn cubemap_direction(face: usize, u: f32, v: f32) -> Vec3 {
        let dir = match face {
            0 => Vec3::new(-u, 1.0, -v),  // +X (right)
            1 => Vec3::new( u, -1.0, -v), // -X (left)
            2 => Vec3::new( v, u, 1.0),   // +Y (forward)
            3 => Vec3::new(-v, u, -1.0),  // -Y (backward)
            4 => Vec3::new( 1.0, u, -v),  // +Z (up)
            5 => Vec3::new(-1.0, -u, -v), // -Z (down)
            _ => unreachable!(),
        };

        dir.normalize()
    }

    fn direction_to_spherical(dir: &Vec3) -> (f32, f32) {
        let theta = dir.x.acos(); // [0, PI]
        let phi = dir.y.atan2(dir.z); // [-PI, PI]
        let phi = if phi < 0.0 { phi + 2.0 * std::f32::consts::PI } else { phi };
        (theta, phi)
    }

    fn sample_equirect(
        pixels: &[f32],
        width: u32,
        height: u32,
        u: f32,
        v: f32,
    ) -> [f32; 3] {
        // Bilinear interpolation
        let x = u * (width - 1) as f32;
        let y = v * (height - 1) as f32;

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let get_pixel = |px: u32, py: u32| {
            let idx = ((py * width + px) * 3) as usize;
            [pixels[idx], pixels[idx + 1], pixels[idx + 2]]
        };

        let c00 = get_pixel(x0, y0);
        let c10 = get_pixel(x1, y0);
        let c01 = get_pixel(x0, y1);
        let c11 = get_pixel(x1, y1);

        // Bilinear blend
        let c0 = [
            c00[0] * (1.0 - fx) + c10[0] * fx,
            c00[1] * (1.0 - fx) + c10[1] * fx,
            c00[2] * (1.0 - fx) + c10[2] * fx,
        ];
        let c1 = [
            c01[0] * (1.0 - fx) + c11[0] * fx,
            c01[1] * (1.0 - fx) + c11[1] * fx,
            c01[2] * (1.0 - fx) + c11[2] * fx,
        ];

        [
            c0[0] * (1.0 - fy) + c1[0] * fy,
            c0[1] * (1.0 - fy) + c1[1] * fy,
            c0[2] * (1.0 - fy) + c1[2] * fy,
        ]
    }

    fn compress_bc6h(pixels: &[f32], width: u32, height: u32) -> Vec<u8> {
        // Convert RGB f32 to RGBA f16 (ispc_texcomp bc6h expects f16 input)
        let pixel_count = (width * height) as usize;
        let mut rgba_f16 = Vec::with_capacity(pixel_count * 4);
        for i in (0..pixels.len()).step_by(3) {
            rgba_f16.push(half::f16::from_f32(pixels[i]));
            rgba_f16.push(half::f16::from_f32(pixels[i + 1]));
            rgba_f16.push(half::f16::from_f32(pixels[i + 2]));
            rgba_f16.push(half::f16::from_f32(1.0));
        }

        let output_size = ispc_texcomp::bc6h::calc_output_size(width, height);
        let mut output = vec![0u8; output_size];
        let rgba_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                rgba_f16.as_ptr() as *const u8,
                rgba_f16.len() * size_of::<half::f16>(),
            )
        };
        let surface = RgbaSurface {
            data: rgba_bytes,
            width,
            height,
            stride: width * 4 * 2, // 4 channels * 2 bytes per f16
        };
        let settings = ispc_texcomp::bc6h::fast_settings();
        ispc_texcomp::bc6h::compress_blocks_into(&settings, &surface, &mut output);
        output
    }

    fn split_asset_path(asset_url: &str, fallback: &str) -> (PathBuf, String) {
        let main = Path::new(asset_url);
        let parent = main.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        let stem = main
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(fallback)
            .to_owned();
        (parent, stem)
    }
}

impl AssetBaker for RawHdrProcessor {
    type Raw = RawHdr;

    #[profiling::function]
    fn bake(raw: Self::Raw, url: AssetUrl) -> Result<Vec<Box<dyn Asset>>> {
        let RawHdr {
            width,
            height,
            pixels,
            ..
        } = raw;

        let face_size = width.min(height).next_power_of_two();

        log::info!("Converting {}x{} equirectangular HDR to {}x{} cubemap faces",
                   width, height, face_size, face_size);

        // Convert to 6 cubemap faces
        let faces = Self::equirect_to_cubemap(&pixels, width, height, face_size);

        // Compress each face with BC6H
        let compressed_faces: Vec<Vec<u8>> = faces
            .iter()
            .enumerate()
            .map(|(idx, face_pixels)| {
                log::info!("Converting cubemap face {}/6 to f16 format", idx + 1);
                Self::compress_bc6h(face_pixels, face_size, face_size)
            })
            .collect();

        // Concatenate all 6 faces into one pixel buffer
        // Order: +X, -X, +Y, -Y, +Z, -Z (Vulkan standard)
        let total_pixels: Vec<u8> = compressed_faces
            .into_iter()
            .flatten()
            .collect();

        log::info!("Total compressed cubemap size: {} bytes", total_pixels.len());

        let asset_url_str = url.path.to_str().ok_or(anyhow!("Invalid asset url"))?;
        let (parent, stem) = Self::split_asset_path(asset_url_str, "cubemap");

        let mut cubemap_path = parent.join(format!("{}_cubemap", stem));
        cubemap_path.set_extension(Texture::extension());
        let cubemap_url: AssetUrl = cubemap_path.into();

        // let asset_serialize_path = base_directory.join(&cubemap_url);
        // serialize_asset(&texture, &asset_serialize_path)?;
        // registry.register(cubemap_url.clone(), texture);

        let texture = TextureBuilder::default()
            .url(cubemap_url.clone())
            .width(face_size)
            .height(face_size)
            .format(TextureFormat::Bc6hUfloat)  // Using f16 uncompressed for now
            .pixels(total_pixels)
            .is_cubemap(true)
            .build()?;

        log::info!("HDR cubemap baked successfully: {:?}", cubemap_url);
        Ok(vec![Box::new(texture) as Box<dyn Asset>])
    }
}
