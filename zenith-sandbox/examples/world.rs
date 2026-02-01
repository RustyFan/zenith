use std::path::PathBuf;
use std::sync::Arc;
use glam::Vec3;
use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;
use winit::keyboard::KeyCode;

use zenith::{launch, App, Args, RenderableApp, RenderContext};
use zenith::rhi::TextureState;
use zenith::renderer::WorldRenderer;
use zenith::asset::manager::AssetManager;
use zenith::asset::{AssetHandle};
use zenith::asset::render::MeshCollection;
use zenith::core::camera::{Camera, CameraController, NEAR_PLANE};
use zenith::core::input::InputActionMapper;
use zenith::core::math::Degree;

pub struct WorldApp {
    world_renderer: Option<WorldRenderer>,
    input: InputActionMapper,
    camera: Camera,
    controller: CameraController,
}

impl App for WorldApp {
    fn new(_args: &Args) -> Result<Self, anyhow::Error> {
        Ok(Self {
            world_renderer: None,
            input: InputActionMapper::new(),
            camera: Camera::default(),
            controller: CameraController::new(20.0),
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

        // Blocking load & bake (first run may be slower).
        let manager = AssetManager::new();
        manager.request_load("mesh/cerberus/scene.gltf")?;

        let mut renderer = WorldRenderer::new(render_device)?;

        // mesh/cerberus/scene.gltf -> mesh/cerberus/scene.mscl
        let collection = AssetHandle::<MeshCollection>::new(PathBuf::from("mesh/cerberus/scene.mscl").into());
        renderer.add_mesh(render_device, collection)?;

        self.world_renderer = Some(renderer);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.camera.set_aspect_ratio(width as f32 / height as f32);
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
            .as_ref()
            .unwrap()
            .render(builder, &mut output, extent.width, extent.height, &self.camera);
    }
}

fn main() {
    launch::<WorldApp>().expect("Failed to launch zenith engine loop!");
}

