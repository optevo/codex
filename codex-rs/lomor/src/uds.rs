//! UDS transport — bincode-framed Unix domain socket client for lomord.
//!
//! Connects to a running `lomord` process at `/tmp/lomord.sock` (or a custom
//! path via [`LomorUdsClient::with_socket`]). Each call opens a fresh
//! connection, sends one [`IpcRequest`], streams [`IpcEvent`]s until a
//! terminal event, then closes the socket — matching lomord's session model.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tokio::net::UnixStream;

pub use lomor_ipc::{
    // Wire protocol
    IpcError,
    IpcEvent,
    IpcRequest,
    // Shared data types (re-exported from lomor-data via lomor-ipc)
    AbortReason,
    CheckpointRef,
    JobId,
    JobSubmission,
    ModelStatus,
    Priority,
    QueueUsage,
    QueuedJobInfo,
    RunningJobInfo,
};
pub use lomor_data::{
    EmbedInputs,
    EmbedMode,
    EmbedRequest,
    FormatConstraint,
    GenerateInput,
    GenerateParams,
    GenerateRequest,
    JobKind,
    LomorErrorCode,
    SubmitMode,
};

/// Default socket path used by `lomord`.
pub const SOCKET_PATH: &str = "/tmp/lomord.sock";

/// Errors from the UDS client.
#[derive(Debug, Error)]
pub enum UdsError {
    /// Failed to open the Unix domain socket.
    #[error("connect to {path}: {source}")]
    Connect {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// IPC framing or protocol error.
    #[error("IPC framing: {0}")]
    Ipc(#[from] IpcError),
    /// lomord returned an error event.
    #[error("lomord error {code:?}: {message}")]
    Remote { code: LomorErrorCode, message: String },
    /// Received an unexpected event type for this request.
    #[error("unexpected event from lomord: {0}")]
    UnexpectedEvent(String),
}

/// A lightweight client for a running `lomord` process.
///
/// Each method opens its own connection and closes it on completion,
/// matching lomord's one-request-per-connection session model.
#[derive(Debug, Clone)]
pub struct LomorUdsClient {
    socket_path: PathBuf,
}

impl Default for LomorUdsClient {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(SOCKET_PATH),
        }
    }
}

impl LomorUdsClient {
    /// Create a client pointing at a custom socket path.
    pub fn with_socket(path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: path.as_ref().to_owned(),
        }
    }

    /// Submit a job and collect all events until the terminal event.
    ///
    /// Returns the full event sequence. Terminal events are `Done`, `Aborted`,
    /// `Expired`, and `Error`. The latter two are mapped to [`UdsError`].
    pub async fn submit(&self, job: JobSubmission) -> Result<Vec<IpcEvent>, UdsError> {
        let mut stream = self.connect().await?;
        lomor_ipc::frame::write_frame(&mut stream, &IpcRequest::Submit(Box::new(job)))
            .await
            .map_err(UdsError::Ipc)?;

        let mut events = Vec::new();
        loop {
            let event: IpcEvent = lomor_ipc::frame::read_frame(&mut stream)
                .await
                .map_err(UdsError::Ipc)?;
            let terminal = matches!(
                event,
                IpcEvent::Done { .. }
                    | IpcEvent::Aborted { .. }
                    | IpcEvent::Expired { .. }
                    | IpcEvent::Error { .. }
            );
            match &event {
                IpcEvent::Error { code, message } => {
                    return Err(UdsError::Remote {
                        code: code.clone(),
                        message: message.clone(),
                    });
                }
                _ => {}
            }
            events.push(event);
            if terminal {
                break;
            }
        }
        Ok(events)
    }

    /// Send a health check and return the responding model id.
    pub async fn health(&self) -> Result<String, UdsError> {
        let mut stream = self.connect().await?;
        lomor_ipc::frame::write_frame(&mut stream, &IpcRequest::Health)
            .await
            .map_err(UdsError::Ipc)?;
        let event: IpcEvent = lomor_ipc::frame::read_frame(&mut stream)
            .await
            .map_err(UdsError::Ipc)?;
        match event {
            IpcEvent::Healthy { model_id } => Ok(model_id.unwrap_or_default()),
            other => Err(UdsError::UnexpectedEvent(format!("{other:?}"))),
        }
    }

    async fn connect(&self) -> Result<UnixStream, UdsError> {
        UnixStream::connect(&self.socket_path)
            .await
            .map_err(|source| UdsError::Connect {
                path: self.socket_path.clone(),
                source,
            })
    }
}
