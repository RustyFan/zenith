use crate::main_loop::EngineLoop;

mod engine;
mod main_loop;
mod app;

pub use app::{App, RenderableApp};
pub use engine::Engine;

pub use paste::paste;

macro_rules! module_facade {
    ($name:ident) => {
        $crate::paste!{
            pub mod $name {
                pub use [<zenith_ $name>]::*;
            }
        }
    };
}

module_facade!(core);
module_facade!(asset);
module_facade!(render);
module_facade!(renderer);
module_facade!(rendergraph);

/// Launch main engine loop with specific App.
pub fn launch<A: RenderableApp>() -> Result<(), anyhow::Error> {
    zenith_core::profile::initialize()?;
    zenith_core::log::initialize()?;
    zenith_asset::initialize()?;

    let app = A::new()?;

    let main_loop = EngineLoop::new(app)?;
    main_loop.run()?;

    Ok(())
}
