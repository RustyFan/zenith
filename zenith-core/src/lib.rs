use std::path::PathBuf;

pub mod log;
pub mod collections;
pub mod camera;
pub mod math;
pub mod input;
pub mod file;
pub mod profile;
pub mod cli;
pub mod time;

pub fn workspace_root() -> PathBuf {
    // Get the directory where Cargo.toml for the workspace is located
    let mut current_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cargo_toml = current_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return current_dir;
                }
            }
        }
        if !current_dir.pop() {
            break;
        }
    }
    // Fallback to parent directory of current crate
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}