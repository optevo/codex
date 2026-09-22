//! Lomor model provider integration for Codex.
//!
//! Lomor is a local inference server for Apple Silicon that exposes an
//! OpenAI-compatible HTTP API and a native Unix domain socket (UDS) interface.
//!
//! # Transports
//!
//! Two transports are available, selectable at compile time via feature flags:
//!
//! - **HTTP** (default): Routes through lomord's OpenAI-compatible HTTP API at
//!   `http://localhost:8080/v1`. Uses the same provider infrastructure as
//!   `ollama` and `lmstudio`. No additional dependencies required.
//!
//! - **UDS** (`features = ["uds"]`): Direct Unix domain socket connection to
//!   lomord at `/tmp/lomord.sock`. Bincode-framed protocol with no HTTP or JSON
//!   overhead. Pulls in `lomor-ipc` and `lomor-data` only — no MLX/Metal stack.
//!
//! # Switching transports
//!
//! In the workspace `Cargo.toml` dependency on this crate:
//!
//! ```toml
//! # HTTP (default — no extra dependencies):
//! codex-lomor = { path = "lomor" }
//!
//! # UDS (bincode/socket only; no MLX/Metal):
//! codex-lomor = { path = "lomor", features = ["uds"] }
//! ```

pub use codex_model_provider_info::{DEFAULT_LOMOR_PORT, LOMOR_OSS_PROVIDER_ID};

/// Native UDS transport — direct bincode-framed socket connection to lomord.
///
/// Enabled by the `uds` feature flag. Uses `lomor-ipc` wire types directly,
/// with no HTTP stack and no MLX/Metal dependency.
#[cfg(feature = "uds")]
pub mod uds;

// ── OSS provider readiness ─────────────────────────────────────────────────────

/// Ensures lomord is running and ready to accept requests.
///
/// Probes `GET http://localhost:{DEFAULT_LOMOR_PORT}/health`. If the probe
/// succeeds the function returns immediately. If lomord is not reachable:
///
/// 1. Locates the `lomord` binary (PATH → `~/.local/bin/lomord` →
///    `~/projects/lomor/target/release/lomord`).
/// 2. Spawns `lomord start` as a detached background process.
/// 3. Polls `/health` every 2 seconds for up to 120 seconds.
///
/// Model weights are memory-mapped at startup (`newBufferWithBytesNoCopy`) —
/// pages are not read from NVMe until Metal first accesses each region.
/// Startup itself is fast (a few seconds); the first inference request bears
/// the page-fault cost as weight pages load on demand from NVMe.
/// The 120-second ceiling is conservative insurance against unexpected
/// startup failures, not an expected load time.
///
/// # Errors
///
/// Returns an `std::io::Error` when:
/// - No `lomord` binary can be found.
/// - `lomord start` fails to spawn.
/// - The health check does not succeed within the timeout.
pub async fn ensure_oss_ready() -> std::io::Result<()> {
    let health_url = format!("http://127.0.0.1:{DEFAULT_LOMOR_PORT}/health");

    // Fast path: server is already up.
    if probe_health(&health_url).await {
        tracing::debug!("lomord is already running");
        return Ok(());
    }

    // Locate the lomord binary.
    let lomord_bin = find_lomord_binary().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot find lomord binary — ensure it is on PATH, at ~/.local/bin/lomord, \
             or at ~/projects/lomor/target/release/lomord",
        )
    })?;

    tracing::info!("starting lomord from {}", lomord_bin.display());
    eprintln!("lomor: starting inference server (this may take up to 2 minutes while the model loads)…");

    // Spawn `lomord start` — it backgrounds itself and returns immediately.
    std::process::Command::new(&lomord_bin)
        .arg("start")
        .spawn()
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("failed to spawn {}: {e}", lomord_bin.display()),
            )
        })?
        .wait()
        .map_err(|e| std::io::Error::new(e.kind(), format!("lomord start failed: {e}")))?;

    // Poll until the server is reachable or we time out.
    let timeout = std::time::Duration::from_secs(120);
    let poll_interval = std::time::Duration::from_secs(2);
    let deadline = std::time::Instant::now() + timeout;

    while std::time::Instant::now() < deadline {
        tokio::time::sleep(poll_interval).await;
        if probe_health(&health_url).await {
            tracing::info!("lomord is ready");
            eprintln!("lomor: inference server is ready");
            return Ok(());
        }
        tracing::debug!("lomord not yet ready, retrying…");
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!(
            "lomord did not become ready within {} seconds; \
             check lomord logs for startup errors",
            timeout.as_secs()
        ),
    ))
}

/// Returns `true` when `GET {url}` responds with a 2xx status.
async fn probe_health(url: &str) -> bool {
    // Build a minimal reqwest client with a short connect timeout.
    let client = match reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(2))
        .timeout(std::time::Duration::from_secs(4))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    match client.get(url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Searches common locations for the `lomord` binary.
///
/// Search order:
/// 1. `lomord` on `PATH` (system-wide install via Nix `home.file`).
/// 2. `~/.local/bin/lomord` (Nix-managed user binary).
/// 3. `~/projects/lomor/target/release/lomord` (development build).
fn find_lomord_binary() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // 1. PATH lookup.
    if let Ok(path) = which::which("lomord") {
        return Some(path);
    }

    // 2. Nix-managed user binary.
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".local/bin/lomord");
        if p.exists() {
            return Some(p);
        }
    }

    // 3. Development build in the lomor project.
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join("projects/lomor/target/release/lomord");
        if p.exists() {
            return Some(p);
        }
    }

    None
}
