use object_store::memory::InMemory;
use object_store::ObjectStore;
use reqwest::Client;
use smolpuff::handlers;
use smolpuff::VectorStore;
use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;

async fn spawn_server() -> String {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let store = VectorStore::open("/test", object_store)
        .await
        .expect("Failed to open store");
    let store = Arc::new(store);

    let app = Router::new()
        .route("/v1/namespaces", post(handlers::create_namespace))
        .route("/v1/namespaces/{ns}", get(handlers::get_namespace))
        .route("/v1/namespaces/{ns}", delete(handlers::delete_namespace))
        .route("/v1/namespaces/{ns}/write", post(handlers::write))
        .route("/v1/namespaces/{ns}/query", post(handlers::query))
        .with_state(store);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind");
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}");

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    base_url
}

#[tokio::test]
async fn test_full_lifecycle() {
    let base = spawn_server().await;
    let client = Client::new();

    // Create namespace
    let resp = client
        .post(format!("{base}/v1/namespaces"))
        .json(&serde_json::json!({
            "name": "test_ns",
            "vector_dim": 3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "test_ns");
    assert_eq!(body["vector_dim"], 3);
    assert_eq!(body["distance"], "cosine");

    // Get namespace
    let resp = client
        .get(format!("{base}/v1/namespaces/test_ns"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["name"], "test_ns");

    // Upsert vectors
    for (id, vec) in [
        ("v1", vec![1.0, 0.0, 0.0]),
        ("v2", vec![0.9, 0.1, 0.0]),
        ("v3", vec![0.0, 1.0, 0.0]),
    ] {
        let resp = client
            .post(format!("{base}/v1/namespaces/test_ns/write"))
            .json(&serde_json::json!({
                "id": id,
                "vector": vec,
                "attributes": {"label": id}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
    }

    // Query
    let resp = client
        .post(format!("{base}/v1/namespaces/test_ns/query"))
        .json(&serde_json::json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 2
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["id"], "v1");
    assert_eq!(results[1]["id"], "v2");

    // Delete namespace
    let resp = client
        .delete(format!("{base}/v1/namespaces/test_ns"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Verify deleted
    let resp = client
        .get(format!("{base}/v1/namespaces/test_ns"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn test_duplicate_namespace() {
    let base = spawn_server().await;
    let client = Client::new();

    let body = serde_json::json!({ "name": "dup", "vector_dim": 3 });

    let resp = client
        .post(format!("{base}/v1/namespaces"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let resp = client
        .post(format!("{base}/v1/namespaces"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn test_dimension_mismatch() {
    let base = spawn_server().await;
    let client = Client::new();

    client
        .post(format!("{base}/v1/namespaces"))
        .json(&serde_json::json!({ "name": "dim3", "vector_dim": 3 }))
        .send()
        .await
        .unwrap();

    let resp = client
        .post(format!("{base}/v1/namespaces/dim3/write"))
        .json(&serde_json::json!({
            "id": "bad",
            "vector": [1.0, 2.0]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn test_namespace_not_found() {
    let base = spawn_server().await;
    let client = Client::new();

    let resp = client
        .get(format!("{base}/v1/namespaces/nonexistent"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(format!("{base}/v1/namespaces/nonexistent/write"))
        .json(&serde_json::json!({
            "id": "x",
            "vector": [1.0]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}
