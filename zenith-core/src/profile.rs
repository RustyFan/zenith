pub fn initialize() -> anyhow::Result<()> {
    profiling::puffin::set_scopes_on(true);

    Ok(())
}