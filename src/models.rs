use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexStatus {
    Building,
    Ready,
    Stale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub status: IndexStatus,
    pub num_centroids: u32,
    pub vector_dim: usize,
    pub num_indexed_vectors: u64,
    pub built_at: DateTime<Utc>,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceMetadata {
    pub name: String,
    pub vector_dim: usize,
    pub distance: String,
    pub approx_row_count: u64,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CreateNamespaceRequest {
    pub name: String,
    pub vector_dim: usize,
    #[serde(default = "default_distance")]
    pub distance: String,
}

fn default_distance() -> String {
    "cosine".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct WriteRequest {
    pub id: String,
    pub vector: Vec<f32>,
    pub attributes: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WriteResponse {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueryRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    pub results: Vec<QueryResultItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResultItem {
    pub id: String,
    pub score: f32,
    pub attributes: Option<serde_json::Value>,
}
