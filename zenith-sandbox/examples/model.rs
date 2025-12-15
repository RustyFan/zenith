use std::env;
use std::sync::{Arc, Weak};
use glam::{Quat, Vec3};
use log::error;
use winit::event::{DeviceEvent, WindowEvent};
use winit::keyboard::KeyCode;
use winit::window::Window;
use zenith::{launch, App, RenderableApp};
use zenith::asset::manager::AssetManager;
use zenith::core::camera::{Camera, CameraController};
use zenith::core::input::InputActionMapper;
use zenith::render::RenderDevice;
use zenith::renderer::{MeshRenderData, SimpleMeshRenderer};
use zenith::rendergraph::{RenderGraphBuilder, RenderGraphResource, Texture};

pub struct GltfRendererApp {
    main_window: Option<Weak<Window>>,
    mesh_renderer: Option<SimpleMeshRenderer>,

    camera: Camera,
    controller: CameraController,

    mapper: InputActionMapper,
}

impl App for GltfRendererApp {
    #[profiling::function]
    fn new() -> Result<Self, anyhow::Error> {
        let args: Vec<String> = env::args().collect();
        if args.len() != 2 {
            error!("Example: {} mesh/cerberus/scene.gltf", args[0]);
            std::process::exit(1);
        }

        let gltf_path = args[1].clone();
        let manager = AssetManager::new();
        // Load synchronously - this will block until complete
        manager.request_load(gltf_path)?;

        let mut mapper = InputActionMapper::new();
        mapper.register_axis("strafe", [KeyCode::KeyD], [KeyCode::KeyA], 0.5);
        mapper.register_axis("walk", [KeyCode::KeyW], [KeyCode::KeyS], 0.5);
        mapper.register_axis("lift", [KeyCode::KeyE], [KeyCode::KeyQ], 0.5);

        Ok(Self {
            main_window: None,
            mesh_renderer: None,

            camera: Default::default(),
            controller: Default::default(),

            mapper,
        })
    }

    #[profiling::function]
    fn on_window_event(&mut self, event: &WindowEvent, window: &Window) {
        self.mapper.on_window_event(event);
        self.controller.on_window_event(event, &window);
    }

    #[profiling::function]
    fn on_device_event(&mut self, event: &DeviceEvent) {
        self.controller.on_device_event(event);
    }

    #[profiling::function]
    fn tick(&mut self, delta_time: f32) {
        self.mapper.tick(delta_time);

        let forward_axis = self.mapper.get_axis("walk");
        let right_axis = self.mapper.get_axis("strafe");
        let up_axis = self.mapper.get_axis("lift");

        self.controller.update_cameras(delta_time, forward_axis, right_axis, up_axis, [&mut self.camera]);
    }
}

impl RenderableApp for GltfRendererApp {
    #[profiling::function]
    fn prepare(&mut self, render_device: &mut RenderDevice, main_window: Arc<Window>) -> Result<(), anyhow::Error> {
        let data = MeshRenderData::new("mesh/cerberus/scene");
        // Asset is already loaded synchronously in new()
        let mut mesh_renderer = SimpleMeshRenderer::from_model(&render_device, data);
        mesh_renderer.set_base_color([0.7, 0.5, 0.3]);

        self.main_window = Some(Arc::downgrade(&main_window));
        self.mesh_renderer = Some(mesh_renderer);
        Ok(())
    }

    #[profiling::function]
    fn render(&mut self, builder: &mut RenderGraphBuilder) -> Option<RenderGraphResource<Texture>> {
        let (width, height) = if let Some(window) = self.main_window.as_ref().and_then(|window| window.upgrade()) {
            (window.inner_size().width, window.inner_size().height)
        } else {
            return None;
        };

        let model_matrix = glam::Mat4::from_scale_rotation_translation(Vec3::splat(0.5), Quat::IDENTITY, Vec3::new(0., 100.0, 0.));

        let view = self.camera.view();
        let proj = self.camera.projection();

        Some(self.mesh_renderer.as_ref().unwrap().build_render_graph(
            builder,
            view,
            proj,
            model_matrix,
            width,
            height
        ))
    }
}

fn main() {
    launch::<GltfRendererApp>().expect("Failed to launch zenith engine loop!");
}