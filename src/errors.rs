use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Database error: {0}")]
    DbError(#[from] slatedb::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Namespace not found: {0}")]
    NamespaceNotFound(String),

    #[error("Namespace already exists: {0}")]
    NamespaceAlreadyExists(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

impl IntoResponse for VectorStoreError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            VectorStoreError::DbError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            VectorStoreError::SerializationError(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            VectorStoreError::NamespaceNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            VectorStoreError::NamespaceAlreadyExists(_) => (StatusCode::CONFLICT, self.to_string()),
            VectorStoreError::DimensionMismatch { .. } => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            VectorStoreError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
        };

        (status, axum::Json(serde_json::json!({ "error": message }))).into_response()
    }
}
