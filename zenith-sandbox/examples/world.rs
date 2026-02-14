use std::path::PathBuf;
use std::sync::Arc;
use glam::Vec3;
use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;
use winit::keyboard::KeyCode;

use zenith::{launch, App, Args, RenderableApp, RenderContext};
use zenith::rhi::TextureState;
use zenith::renderer::WorldRenderer;
use zenith::asset::manager::AssetRequestor;
use zenith::asset::{AssetHandle};
use zenith::asset::mesh::Scene;
use zenith::core::camera::{Camera, CameraController, NEAR_PLANE};
use zenith::core::input::InputActionMapper;
use zenith::core::log;
use zenith::core::math::Degree;
use zenith::core::time::{Milliseconds, Timer};

pub struct WorldApp {
    world_renderer: Option<WorldRenderer>,
    input: InputActionMapper,
    camera: Camera,
    controller: CameraController,
    asset_manager: AssetRequestor,
}

impl App for WorldApp {
    fn new(_args: &Args) -> Result<Self, anyhow::Error> {
        Ok(Self {
            world_renderer: None,
            input: InputActionMapper::new(),
            camera: Camera::default(),
            controller: CameraController::new(10.0),
            asset_manager: AssetRequestor::new(),
        })
    }

    fn on_window_event(&mut self, event: &WindowEvent, window: &Window) {
        self.input.on_window_event(event);
        self.controller.on_window_event(event, window);
    }

    fn on_device_event(&mut self, event: &DeviceEvent) {
        self.controller.on_device_event(event);
    }

    fn tick(&mut self, delta_time: f32) {
        self.input.tick(delta_time);

        let forward = self.input.get_axis("walk");
        let right = self.input.get_axis("strafe");
        let up = self.input.get_axis("lift");

        self.controller.update_cameras(
            delta_time,
            forward,
            right,
            up,
            std::iter::once(&mut self.camera),
        );
    }
}

impl RenderableApp for WorldApp {
    fn prepare(&mut self, render_device: &zenith::rhi::RenderDevice, window: Arc<Window>) -> Result<(), anyhow::Error> {
        let mut prepare_timer = Timer::new();
        prepare_timer.start();

        self.input.register_axis("walk", [KeyCode::KeyW], [KeyCode::KeyS], 0.2);
        self.input.register_axis("strafe", [KeyCode::KeyD], [KeyCode::KeyA], 0.2);
        self.input.register_axis("lift", [KeyCode::KeyE], [KeyCode::KeyQ], 0.2);

        let size = window.inner_size();
        let aspect = if size.height == 0 {
            1.0
        } else {
            (size.width as f32) / (size.height as f32)
        };
        let mut camera = Camera::new(Degree::from(90.0), aspect, NEAR_PLANE);
        camera.set_position(Vec3::new(0.0, -90.0, 0.0));
        self.camera = camera;

        let mut load_timer = Timer::new();
        load_timer.start();
        self.asset_manager.request_load("mesh/cerberus/scene.gltf")?;
        self.asset_manager.request_load("texture/minedump_flats_4k.hdr")?;
        load_timer.stop();
        let load_ms = load_timer.elapsed_total::<Milliseconds>().value();

        let mut renderer_new_timer = Timer::new();
        renderer_new_timer.start();
        let mut renderer = WorldRenderer::new(render_device, size.width, size.height)?;
        renderer_new_timer.stop();
        let renderer_new_ms = renderer_new_timer.elapsed_total::<Milliseconds>().value();

        let collection = AssetHandle::<Scene>::new(PathBuf::from("mesh/cerberus/scene.mscl").into());
        let mut upload_timer = Timer::new();
        upload_timer.start();
        renderer.add_mesh(render_device, collection)?;

        let skybox_handle = AssetHandle::<zenith::asset::texture::Texture>::new(
            PathBuf::from("texture/minedump_flats_4k_cubemap_2048x2048.tex").into()
        );
        if let Some(skybox_tex) = skybox_handle.get() {
            renderer.set_skybox(render_device, &skybox_tex)?;
            log::info!("Skybox loaded and set successfully");
        } else {
            log::warn!("Skybox texture not found after loading HDR");
        }

        upload_timer.stop();
        let upload_ms = upload_timer.elapsed_total::<Milliseconds>().value();

        self.world_renderer = Some(renderer);

        prepare_timer.stop();
        let prepare_ms = prepare_timer.elapsed_total::<Milliseconds>().value();
        log::info!(
            "WorldApp prepare timings: prepare={:.3}ms, request_load={:.3}ms, world_renderer_new={:.3}ms, gpu_upload={:.3}ms",
            prepare_ms,
            load_ms,
            renderer_new_ms,
            upload_ms
        );
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);
        self.world_renderer.as_mut().unwrap().resize(width, height);
    }

    fn render(&mut self, mut context: RenderContext) {
        let extent = context.extent();
        if extent.width == 0 || extent.height == 0 {
            return;
        }

        let output = context.swapchain_texture();
        let builder = context.builder();
        let mut output = builder.import(output, TextureState::Undefined);

        self.world_renderer
            .as_mut()
            .unwrap()
            .render(builder, &self.camera, &mut output);
    }
}

fn main() {
    launch::<WorldApp>().expect("Failed to launch zenith engine loop!");
}

