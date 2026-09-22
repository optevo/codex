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
