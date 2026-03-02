use axum::Router;
use axum::middleware;
use axum::routing::{delete, get, post};
use metrics_exporter_prometheus::PrometheusBuilder;
use object_store::ObjectStore;
use smolpuff::VectorStore;
use smolpuff::handlers;
use smolpuff::metrics::track_metrics;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    // Configure object store: try S3/LocalStack from env, fallback to InMemory
    let object_store: Arc<dyn ObjectStore> =
        match (std::env::var("AWS_ENDPOINT"), std::env::var("S3_BUCKET")) {
            (Ok(endpoint), Ok(bucket)) => Arc::new(
                object_store::aws::AmazonS3Builder::new()
                    .with_endpoint(&endpoint)
                    .with_access_key_id(
                        std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_else(|_| "test".to_string()),
                    )
                    .with_secret_access_key(
                        std::env::var("AWS_SECRET_ACCESS_KEY")
                            .unwrap_or_else(|_| "test".to_string()),
                    )
                    .with_allow_http(true)
                    .with_bucket_name(&bucket)
                    .with_region(
                        std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    )
                    .build()?,
            ),
            _ => {
                tracing::info!("No S3 config found, using in-memory object store");
                Arc::new(object_store::memory::InMemory::new())
            }
        };

    let prometheus_handle = PrometheusBuilder::new()
        .set_buckets(&[
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0,
        ])
        .unwrap()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    let store = VectorStore::open("/smolpuff", object_store).await?;
    let store = Arc::new(store);

    let app = Router::new()
        .route("/", get(handlers::root))
        .route(
            "/metrics",
            get(move || async move { prometheus_handle.render() }),
        )
        .route("/v1/namespaces", post(handlers::create_namespace))
        .route("/v1/namespaces/{ns}", get(handlers::get_namespace))
        .route("/v1/namespaces/{ns}", delete(handlers::delete_namespace))
        .route("/v1/namespaces/{ns}/write", post(handlers::write))
        .route("/v1/namespaces/{ns}/query", post(handlers::query))
        .route("/v1/namespaces/{ns}/index", post(handlers::build_index))
        .layer(middleware::from_fn(track_metrics))
        .layer(TraceLayer::new_for_http())
        .with_state(store);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;

    Ok(())
}
