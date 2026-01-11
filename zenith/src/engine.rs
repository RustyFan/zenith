use crate::RenderableApp;
use std::sync::Arc;
use winit::window::Window;
use zenith_rendergraph::RenderGraphBuilder;
use zenith_rhi::core::{select_physical_device, PhysicalDevice};
use zenith_rhi::swapchain::SwapchainWindow;
use zenith_rhi::{vk, PipelineCache, RenderDevice, RhiCore, Swapchain, SwapchainConfig};
use crate::app::RenderContext;

pub struct Engine {
    pub main_window: Arc<Window>,

    pipeline_cache: PipelineCache,
    swapchain: Swapchain,
    pub render_device: RenderDevice,
    _physical_device: PhysicalDevice,
    _rhi_core: RhiCore,

    // _puffin_server: puffin_http::Server,

    should_exit: bool,
}

impl Engine {
    pub fn new(main_window: Arc<Window>) -> Result<Self, anyhow::Error> {
        // let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        // let _puffin_server = puffin_http::Server::new(&server_addr)?;

        let core = RhiCore::new(&main_window)?;
        let swapchain_window = SwapchainWindow::new(&main_window, &core)?;
        let physical_device = select_physical_device(core.instance(), &swapchain_window)?;
        let device = core.create_render_device(&physical_device)?;

        let swapchain_config = SwapchainConfig::default();
        let swapchain = Swapchain::new(
            &core,
            &device,
            swapchain_window,
            swapchain_config,
        )?;

        let pipeline_cache = PipelineCache::new(device.handle())?;

        Ok(Self {
            main_window,
            _rhi_core: core,
            _physical_device: physical_device,
            render_device: device,

            swapchain,
            pipeline_cache,

            // _puffin_server,

            should_exit: false,
        })
    }

    #[profiling::function]
    pub fn tick(&mut self, _delta_time: f32) {
    }

    #[profiling::function]
    pub fn render<A: RenderableApp>(&mut self, app: &mut A) {
        self.render_device.begin_frame();

        let mut builder = RenderGraphBuilder::new();
        let render_context = RenderContext::new(
            &mut builder,
            &self.swapchain,
            self.render_device.frame_index(),
        );
        app.render(render_context);

        let render_graph = builder.build();
        let mut compiled = render_graph.compile(&mut self.render_device, &mut self.pipeline_cache);

        compiled.execute(&mut self.render_device);

        let retired = compiled.present(&mut self.swapchain, &mut self.render_device)
            .expect("Failed to present swapchain!");

        retired.release_frame_resources(&mut self.render_device);
        self.render_device.end_frame();
    }

    fn recreate_swapchain(&mut self) {
        let inner_size = self.main_window.inner_size();
        if inner_size.width == 0 || inner_size.height == 0 {
            return;
        }

        let window_extent = vk::Extent2D {
            width: inner_size.width,
            height: inner_size.height,
        };

        self.swapchain.resize(window_extent).unwrap();
    }

    #[profiling::function]
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.recreate_swapchain();
        }
    }

    #[inline]
    pub fn request_exit(&mut self) { self.should_exit = true; }

    #[inline]
    pub fn should_exit(&self) -> bool { self.should_exit }

    #[inline]
    pub fn pipeline_cache_size(&self) -> usize { self.pipeline_cache.len() }
}
