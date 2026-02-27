use axum::Json;
use axum::extract::{Path, State};
use std::sync::Arc;

use crate::errors::VectorStoreError;
use crate::models::*;
use crate::store::VectorStore;

pub async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "smolpuff",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

pub async fn create_namespace(
    State(store): State<Arc<VectorStore>>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<Json<NamespaceMetadata>, VectorStoreError> {
    let meta = store
        .create_namespace(&req.name, req.vector_dim, &req.distance)
        .await?;
    Ok(Json(meta))
}

pub async fn get_namespace(
    State(store): State<Arc<VectorStore>>,
    Path(ns): Path<String>,
) -> Result<Json<NamespaceMetadata>, VectorStoreError> {
    let meta = store.get_namespace(&ns).await?;
    Ok(Json(meta))
}

pub async fn delete_namespace(
    State(store): State<Arc<VectorStore>>,
    Path(ns): Path<String>,
) -> Result<Json<serde_json::Value>, VectorStoreError> {
    store.delete_namespace(&ns).await?;
    Ok(Json(serde_json::json!({ "status": "deleted" })))
}

pub async fn write(
    State(store): State<Arc<VectorStore>>,
    Path(ns): Path<String>,
    Json(req): Json<WriteRequest>,
) -> Result<Json<WriteResponse>, VectorStoreError> {
    store
        .upsert(&ns, &req.id, req.vector, req.attributes)
        .await?;
    Ok(Json(WriteResponse {
        id: req.id,
        status: "ok".to_string(),
    }))
}

pub async fn query(
    State(store): State<Arc<VectorStore>>,
    Path(ns): Path<String>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, VectorStoreError> {
    let results = store.query_ns(&ns, &req.vector, req.top_k).await?;
    Ok(Json(QueryResponse { results }))
}
