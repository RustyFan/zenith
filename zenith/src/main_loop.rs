use std::sync::Arc;
use log::info;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use crate::app::{RenderableApp};
use crate::Engine;

pub struct EngineLoop<A> {
    engine: Option<Engine>,
    app: A,

    frame_count: u64,
    last_tick: std::time::Instant,
    last_time_printed: std::time::Instant,
    should_exit: bool,
}

impl<A: RenderableApp> ApplicationHandler for EngineLoop<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_min_inner_size(LogicalSize::new(32, 32))
            .with_inner_size(LogicalSize::new(1920, 1080));

        // TODO: only renderable app should create window
        let main_window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .unwrap(),
        );

        let mut engine = Engine::new(main_window.clone()).unwrap();

        self.app.prepare(&mut engine.render_device, main_window.clone()).unwrap();
        self.engine = Some(engine);

        main_window.request_redraw();
    }

    #[profiling::function]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let engine = self.engine.as_mut().unwrap();
        if engine.should_exit() {
            event_loop.exit();
        }

        self.process_window_event(&event);
    }

    #[profiling::function]
    fn device_event(&mut self, event_loop: &ActiveEventLoop, _device_id: DeviceId, event: DeviceEvent) {
        let engine = self.engine.as_mut().unwrap();
        if engine.should_exit() {
            event_loop.exit();
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
            last_tick: std::time::Instant::now(),
            last_time_printed: std::time::Instant::now(),
            should_exit: false,
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

                engine.should_exit = true;
            }
            WindowEvent::RedrawRequested => {
                self.tick();

                let engine = self.engine.as_mut().unwrap();
                let app = &mut self.app;

                engine.render(app);
                engine.main_window.request_redraw();

                profiling::finish_frame!();
            }
            _ => {}
        }
    }

    #[profiling::function]
    fn tick(&mut self) {
        if self.should_exit {
            return;
        }

        let delta_time = {
            let now = std::time::Instant::now();
            let delta_time = now - self.last_tick;
            self.last_tick = now;

            let last_time_print_elapsed = (now - self.last_time_printed).as_secs_f32();
            if last_time_print_elapsed > 1. {
                info!("Frame rate: {} fps", self.frame_count as f32 / last_time_print_elapsed);
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