use std::sync::Arc;
use log::info;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use crate::app::{RenderableApp};
use crate::Engine;

pub struct EngineLoop<A> {
    app: A,
    engine: Option<Engine>,

    frame_count: u64,
    last_tick_time: std::time::Instant,
    last_time_printed: std::time::Instant,
}

impl<A: RenderableApp> ApplicationHandler for EngineLoop<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_size = LogicalSize::new(1920, 1080);

        // Calculate center position on primary monitor
        let position = event_loop.primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
            .map(|monitor| {
                let monitor_size = monitor.size();
                let monitor_pos = monitor.position();
                let scale_factor = monitor.scale_factor();

                // Convert logical window size to physical pixels
                let physical_window_width = (window_size.width as f64 * scale_factor) as i32;
                let physical_window_height = (window_size.height as f64 * scale_factor) as i32;

                let x = monitor_pos.x + (monitor_size.width as i32 - physical_window_width) / 2;
                let y = monitor_pos.y + (monitor_size.height as i32 - physical_window_height) / 2;
                PhysicalPosition::new(x, y)
            });

        let mut window_attributes = Window::default_attributes()
            .with_min_inner_size(LogicalSize::new(32, 32))
            .with_inner_size(window_size);

        if let Some(pos) = position {
            window_attributes = window_attributes.with_position(pos);
        }

        // TODO: only renderable app should create window
        let main_window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .unwrap(),
        );

        let engine = Engine::new(main_window.clone()).unwrap();

        self.app.prepare(&engine.render_device, main_window.clone()).unwrap();
        self.engine = Some(engine);

        main_window.request_redraw();
    }

    #[profiling::function]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let engine = self.engine.as_mut().unwrap();
        if engine.should_exit() {
            event_loop.exit();
            engine.render_device.wait_until_idle().unwrap();
        }

        self.process_window_event(&event);
    }

    #[profiling::function]
    fn device_event(&mut self, event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        let engine = self.engine.as_mut().unwrap();
        if engine.should_exit() {
            event_loop.exit();
            engine.render_device.wait_until_idle().unwrap();
        }
        
        self.app.on_device_event(&event);
    }
}

impl<A: RenderableApp> EngineLoop<A> {
    pub(super) fn new(app: A) -> Result<Self, anyhow::Error> {
        Ok(Self {
            engine: None,
            app,

            frame_count: 0u64,
            last_tick_time: std::time::Instant::now(),
            last_time_printed: std::time::Instant::now(),
        })
    }

    pub fn run(mut self) -> Result<(), anyhow::Error> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run_app(&mut self)?;
        Ok(())
    }

    #[profiling::function("main_loop")]
    fn process_window_event(&mut self, event: &WindowEvent) {
        // TODO: multi-window support
        self.app.on_window_event(event, self.engine.as_ref().unwrap().main_window.as_ref());
        
        match event {
            WindowEvent::Resized(_) => {
                let engine = self.engine.as_mut().unwrap();
                let app = &mut self.app;

                let inner_size = engine.main_window.inner_size();
                engine.resize(inner_size.width, inner_size.height);
                app.resize(inner_size.width, inner_size.height);
            }
            WindowEvent::CloseRequested => {
                let engine = self.engine.as_mut().unwrap();

                engine.request_exit();
            }
            WindowEvent::RedrawRequested => {
                self.tick();

                let engine = self.engine.as_mut().unwrap();
                let app = &mut self.app;

                engine.render(app);
                engine.main_window.request_redraw();

                // profiling::finish_frame!();
            }
            _ => {}
        }
    }

    #[profiling::function]
    fn tick(&mut self) {
        let delta_time = {
            let now = std::time::Instant::now();
            let delta_time = now - self.last_tick_time;
            self.last_tick_time = now;

            let last_time_print_elapsed = (now - self.last_time_printed).as_secs_f32();
            if last_time_print_elapsed > 1. {
                let fps: u32 = (self.frame_count as f32 / last_time_print_elapsed).ceil() as u32;
                let engine = self.engine.as_ref().unwrap();
                let stats = engine.render_device.last_defer_release_stats();
                info!(
                    "Frame rate: {} fps, pipelines: {}, deferred: {}b/{}t/{}p",
                    fps,
                    engine.pipeline_cache_size(),
                    stats.buffer_count,
                    stats.texture_count,
                    stats.pool_count,
                );
                self.last_time_printed = now;
                self.frame_count = 0;
            }

            delta_time.as_secs_f32()
        };

        let engine = self.engine.as_mut().unwrap();
        let app = &mut self.app;

        engine.tick(delta_time);
        app.tick(delta_time);

        self.frame_count += 1;
    }
}