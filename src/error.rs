use thiserror::Error;

#[derive(Debug, Error)]
pub enum TurboVecError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid bit width {0}: must be 1-8")]
    InvalidBitWidth(u8),

    #[error("codebook computation failed: {0}")]
    CodebookError(String),

    #[error("index is empty")]
    EmptyIndex,

    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TurboVecError>;
