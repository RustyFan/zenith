use crate::RenderableApp;
use std::sync::Arc;
use winit::window::Window;
use zenith_rendergraph::RenderGraphBuilder;
use zenith_rhi::core::{select_physical_device, PhysicalDevice};
use zenith_rhi::swapchain::SwapchainWindow;
use zenith_rhi::{vk, CommandPool, PipelineCache, RenderDevice, RhiCore, Swapchain, SwapchainConfig};
use crate::app::RenderContext;

pub struct Engine {
    execute_command_pools: Vec<CommandPool>,
    present_command_pools: Vec<CommandPool>,
    pipeline_cache: PipelineCache,
    swapchain: Swapchain,
    pub render_device: RenderDevice,
    _physical_device: PhysicalDevice,
    _rhi_core: RhiCore,

    pub main_window: Arc<Window>,
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

        let pipeline_cache = PipelineCache::new(&device)?;

        let num_frames = device.num_frames();
        let (execute_command_pools, present_command_pools) = (0..num_frames)
            .map(|_| -> Result<(CommandPool, CommandPool), vk::Result> {
                Ok((
                    CommandPool::new(
                        &device,
                        physical_device.graphics_queue_family(),
                        vk::CommandPoolCreateFlags::empty(),
                    )?,
                    CommandPool::new(
                        &device,
                        physical_device.present_queue_family(),
                        vk::CommandPoolCreateFlags::empty(),
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

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
        self.execute_command_pools[frame_index].reset().expect("Failed to reset execute command pool");

        let mut builder = RenderGraphBuilder::new();
        let render_context = RenderContext::new(
            &mut builder,
            &self.swapchain,
            frame_index,
        );
        app.render(render_context);

        let render_graph = builder.build();
        let mut compiled = render_graph.compile(&mut self.render_device, &mut self.pipeline_cache);

        compiled.execute(&mut self.render_device, &self.execute_command_pools[frame_index])
            .expect("Failed to execute render graph!");

        let retired = compiled.present(&mut self.render_device, &self.present_command_pools[frame_index], &mut self.swapchain)
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
