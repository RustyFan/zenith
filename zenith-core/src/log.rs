pub use log::{trace, debug, info, warn, error, LevelFilter};

pub fn initialize(level: LevelFilter) -> Result<(), anyhow::Error> {
    env_logger::builder()
        .filter_level(level)
        .filter_module("wgpu_core", log::LevelFilter::Warn)
        .filter_module("wgpu_hal", log::LevelFilter::Error)
        .filter_module("naga", log::LevelFilter::Error)
        .parse_default_env()
        .init();

    Ok(())
}