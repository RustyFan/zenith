pub use log::{trace, debug, info, warn, error, LevelFilter};

pub fn initialize(level: LevelFilter) -> Result<(), anyhow::Error> {
    env_logger::builder()
        .filter_level(level)
        .parse_default_env()
        .init();

    Ok(())
}