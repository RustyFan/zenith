use std::sync::Arc;
use zenith_rendergraph::{RenderGraphBuilder, RenderGraphResource};
use zenith_rhi::{vk, BindlessPool, Buffer, BufferDesc, BufferState, ComputePipelineDesc, ImmediateCommandEncoder, RenderDevice, Shader, ShaderStage, Texture, TextureDesc, TextureLayout, TextureState, UploadPool};

pub struct ImageBasedLightingRenderer {
    pub skybox_texture: Option<Arc<Texture>>,
    buffer: Arc<Buffer>,
    diffuse_inter_shader: Arc<Shader>,
    is_dirty: bool,
}

impl ImageBasedLightingRenderer {
    pub fn new(device: &Arc<RenderDevice>) -> anyhow::Result<Self> {
        let diffuse_inter_shader = Arc::new(Shader::from_file(
            "shader.ibl.diffuse_integration.cs",
            device,
            "content/shaders/ibl_diffuse.slang",
            ShaderStage::Compute,
        )?);

        let buffer = Arc::new(Buffer::new(device, &BufferDesc::storage("ibl_sh_coefficients", 3 * 3 * 4 * size_of::<f32>() as u64))?);

        Ok(Self {
            skybox_texture: None,
            buffer,
            diffuse_inter_shader,
            is_dirty: true,
        })
    }

    pub fn set_skybox(&mut self, device: &Arc<RenderDevice>, texture_asset: &zenith_asset::texture::Texture) -> anyhow::Result<()> {
        // Create GPU cubemap texture
        let gpu_texture = Texture::new(
            device,
            &TextureDesc::new_cube(
                "skybox_cubemap",
                texture_asset.width,
                texture_asset.format.to_vk(),
            )
                .with_mip_levels(texture_asset.mip_levels)
                .with_usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        )?;

        // Upload texture data using immediate command encoder
        {
            let mut upload_pool = UploadPool::new()?;
            upload_pool.enqueue_texture_upload(
                device,
                gpu_texture.as_range(TextureLayout::Undefined, .., ..),
                &texture_asset.pixels,
                TextureState::Sampled,
            )?;

            let immediate = ImmediateCommandEncoder::new(device, device.graphics_queue())?;
            upload_pool.flush(&immediate, device)?;
        }

        self.skybox_texture = Some(Arc::new(gpu_texture));
        Ok(())
    }

    pub fn render(
        &mut self,
        builder: &mut RenderGraphBuilder,
        bindless_pool: &mut BindlessPool,
    ) -> RenderGraphResource<Buffer> {
        if !self.is_dirty {
            return builder.import(self.buffer.clone(), BufferState::Undefined);
        }

        let cubemap = if let Some(texture) = self.skybox_texture.as_ref() {
            builder.import(texture.clone(), TextureState::Sampled)
        } else {
            return builder.import(self.buffer.clone(), BufferState::Undefined);
        };
        let mut buffer = builder.import(self.buffer.clone(), BufferState::Undefined);

        let mut node = builder.add_compute_node("integrate_ibl");

        let cubemap = node.read(&cubemap, TextureState::Sampled);
        let buffer_access = node.write(&mut buffer, BufferState::StorageWrite);

        let pipeline_desc = ComputePipelineDesc::new("integrate_ibl", self.diffuse_inter_shader.clone());

        let pipeline = node.register_pipeline(pipeline_desc);
        let bindless_set = bindless_pool.set();
        node.execute(move |ctx| {
            if let Some(pipe) = ctx.bind_pipeline(pipeline) {
                pipe.bind_raw(BindlessPool::SET_INDEX, &[bindless_set], &[])
                    .bind("skybox_cubemap", cubemap)?
                    .bind("outSHCoefficients", buffer_access)?;

                ctx.encoder().dispatch(1, 1, 1);
            }

            Ok(())
        });

        self.is_dirty = true;

        buffer
    }
}