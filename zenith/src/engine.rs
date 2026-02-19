use crate::app::RenderContext;
use crate::RenderableApp;
use std::sync::{Arc};
use winit::window::Window;
use zenith_rendergraph::RenderGraphBuilder;
use zenith_rhi::core::{select_physical_device, PhysicalDevice};
use zenith_rhi::swapchain::SwapchainWindow;
use zenith_rhi::{vk, CommandPool, PipelineRegistry, RenderDevice, RhiCore, Swapchain, SwapchainConfig};

pub struct Engine {
    execute_command_pools: Vec<CommandPool>,
    present_command_pools: Vec<CommandPool>,
    pipeline_cache: PipelineRegistry,
    swapchain: Swapchain,
    pub render_device: RenderDevice,
    _physical_device: PhysicalDevice,
    _rhi_core: RhiCore,

    pub main_window: Arc<Window>,
    // _puffin_server: puffin_http::Server,

    should_exit: bool,
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.render_device.wait_until_idle().unwrap();

        zenith_renderer::deinitialize();
    }
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
            "swapchain.main",
            &core,
            &device,
            swapchain_window,
            swapchain_config,
        )?;

        let pipeline_cache = PipelineRegistry::new();

        let num_frames = device.num_frames();
        let (execute_command_pools, present_command_pools) = (0..num_frames)
            .map(|idx| -> Result<(CommandPool, CommandPool), vk::Result> {
                Ok((
                    CommandPool::new(
                        &format!("command_pool.execute.f{idx}"),
                        &device,
                        physical_device.graphics_queue_family(),
                        vk::CommandPoolCreateFlags::empty(),
                    )?,
                    CommandPool::new(
                        &format!("command_pool.present.f{idx}"),
                        &device,
                        physical_device.present_queue_family(),
                        vk::CommandPoolCreateFlags::empty(),
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        zenith_renderer::initialize(&device)?;

        Ok(Self {
            execute_command_pools,
            present_command_pools,
            pipeline_cache,
            swapchain,
            render_device: device,
            _physical_device: physical_device,
            _rhi_core: core,

            // _puffin_server,

            main_window,
            should_exit: false,
        })
    }

    #[profiling::function]
    pub fn tick(&mut self, _delta_time: f32) {
    }

    #[profiling::function]
    pub fn render<A: RenderableApp>(&mut self, app: &mut A) {
        let frame_index = self.render_device.begin_frame();

        let (image_index, _) = self.swapchain.acquire_next_image(self.render_device.handle())
            .expect("Failed to acquire next swapchain image!");
        self.swapchain.reset_current_fence(self.render_device.handle())
            .expect("Failed to reset swapchain fence!");

        self.render_device.reset_frame_resources();
        self.execute_command_pools[frame_index].reset().expect("Failed to reset execute command pool");

        let mut builder = RenderGraphBuilder::new(&mut self.pipeline_cache, &self.render_device);
        let context = RenderContext::new(&self.render_device, &self.swapchain, frame_index);
        app.render(&mut builder, context);
        let render_graph = builder.build();
        let mut compiled = render_graph.compile(&mut self.render_device, &mut self.pipeline_cache);

        compiled.execute(&mut self.render_device, &self.execute_command_pools[frame_index], &self.pipeline_cache)
            .expect("Failed to execute render graph!");

        let retired = compiled.present(&mut self.render_device, &self.present_command_pools[frame_index], &self.pipeline_cache, &mut self.swapchain, image_index)
            .expect("Failed to present swapchain!");

        retired.release_frame_resources(&mut self.render_device);
        self.render_device.end_frame();
    }

    #[profiling::function]
    pub fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        let window_extent = vk::Extent2D {
            width,
            height,
        };

        self.swapchain.resize(&self.render_device, window_extent)?;

        Ok(())
    }

    #[inline]
    pub fn request_exit(&mut self) { self.should_exit = true; }

    #[inline]
    pub fn should_exit(&self) -> bool { self.should_exit }

    #[inline]
    pub fn pipeline_cache_size(&self) -> usize { self.pipeline_cache.len() }
}
