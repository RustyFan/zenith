use crate::app::RenderContext;
use crate::RenderableApp;
use std::sync::{Arc};
use itertools::multiunzip;
use winit::window::Window;
use zenith_rendergraph::RenderGraphBuilder;
use zenith_rhi::core::{select_physical_device, PhysicalDevice};
use zenith_rhi::swapchain::SwapchainWindow;
use zenith_rhi::{vk, BindlessPool, CommandPool, Fence, PipelineRegistry, RenderDevice, RhiCore, Swapchain, SwapchainConfig, TransientResourceCache, NUM_BACK_BUFFERS};
use zenith_rhi::defer_release::DeferReleaseQueue;

pub struct Engine {
    pub render_device: Arc<RenderDevice>,

    frame_resource_fences: Vec<Fence>,
    resource_caches: Vec<TransientResourceCache>,
    defer_release_queues: Vec<DeferReleaseQueue>,
    pub bindless_pool: BindlessPool,

    execute_command_pools: Vec<CommandPool>,
    present_command_pools: Vec<CommandPool>,
    pipeline_cache: PipelineRegistry,
    swapchain: Swapchain,
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
        let device = RenderDevice::new(&core, &physical_device, NUM_BACK_BUFFERS)?;

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
        let frame_iter = (0..num_frames)
            .map(|idx| {
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
                    Fence::new(&format!("fence.execution.f{idx}"), &device, true)?,
                    TransientResourceCache::default(),
                    DeferReleaseQueue::new(),
                ))
            })
            .collect::<Result<Vec<_>, vk::Result>>()?
            .into_iter();
        let (execute_command_pools,
            present_command_pools,
            frame_resource_fences,
            resource_caches,
            defer_release_queues) = multiunzip(frame_iter);

        let bindless_pool = BindlessPool::new(&device)?;

        zenith_renderer::initialize(&device)?;

        Ok(Self {
            execute_command_pools,
            present_command_pools,
            pipeline_cache,
            swapchain,
            render_device: device,
            frame_resource_fences,
            resource_caches,
            defer_release_queues,
            _physical_device: physical_device,
            _rhi_core: core,

            // _puffin_server,

            main_window,
            should_exit: false,
            bindless_pool,
        })
    }

    #[inline]
    pub fn defer_release_queue(&self) -> &DeferReleaseQueue {
        &self.defer_release_queues[self.render_device.current_frame_index()]
    }

    #[profiling::function]
    pub fn tick(&mut self, _delta_time: f32) {
    }

    #[profiling::function]
    pub fn render<A: RenderableApp>(&mut self, app: &mut A) {
        let frame_index = self.render_device.begin_frame();
        unsafe {
            let fence = self.frame_resource_fences[frame_index].handle();
            self.render_device.handle().wait_for_fences(&[fence], true, u64::MAX).unwrap();
            self.render_device.handle().reset_fences(&[fence]).unwrap();
        }

        // reset frame fences
        let (image_index, _) = self.swapchain.acquire_next_image(self.render_device.handle())
            .expect("Failed to acquire next swapchain image!");
        self.swapchain.reset_current_fence(self.render_device.handle())
            .expect("Failed to reset swapchain fence!");

        let defer_release = &mut self.defer_release_queues[frame_index];
        let resource_cache = &mut self.resource_caches[frame_index];
        let frame_resource_fence = &mut self.frame_resource_fences[frame_index];

        // reset frame resources
        defer_release.release_all();
        self.execute_command_pools[frame_index].reset().expect("Failed to reset execute command pool");

        let mut builder = RenderGraphBuilder::new(&self.render_device, &mut self.pipeline_cache);
        let context = RenderContext::new(&mut self.bindless_pool, &self.swapchain, frame_index);
        app.render(&mut builder, context);

        let graph = builder.build();
        let mut compiled = graph.compile(&mut self.render_device, resource_cache);

        compiled.execute(&mut self.render_device, &self.execute_command_pools[frame_index], &self.pipeline_cache, defer_release, &frame_resource_fence)
            .expect("Failed to execute render graph!");

        let retired = compiled.present(&mut self.render_device, &self.present_command_pools[frame_index], &self.pipeline_cache, defer_release, &mut self.swapchain, image_index)
            .expect("Failed to present swapchain!");

        retired.release_frame_resources(resource_cache);
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
