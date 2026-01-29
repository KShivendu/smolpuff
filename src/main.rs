use object_store::memory::InMemory;
use object_store::ObjectStore;
use smolpuff::VectorStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use in-memory store for demo (replace with S3 for production)
    // let object_store: Arc<dyn ObjectStore + 'static> = Arc::new(InMemory::new());

    // For S3, uncomment and configure:
    // localstack start
    // awslocal s3 mb s3://smolpuff --region us-east-1
    let object_store: Arc<dyn ObjectStore> = Arc::new(
        object_store::aws::AmazonS3Builder::new()
            .with_endpoint("http://localhost:4566") // LocalStack or real S3
            .with_access_key_id("access_key_id")
            .with_secret_access_key("secret_acess_key")
            .with_allow_http(true)
            .with_bucket_name("smolpuff")
            .with_region("us-east-1")
            .build()?,
    );

    let store = VectorStore::open("/vectors", object_store).await?;

    // Add some test vectors
    println!("Adding vectors...");

    store
        .add(
            "doc1",
            vec![1.0, 0.0, 0.0],
            Some(serde_json::json!({"title": "Document 1", "category": "A"})),
        )
        .await?;

    store
        .add(
            "doc2",
            vec![0.9, 0.1, 0.0],
            Some(serde_json::json!({"title": "Document 2", "category": "A"})),
        )
        .await?;

    store
        .add(
            "doc3",
            vec![0.0, 1.0, 0.0],
            Some(serde_json::json!({"title": "Document 3", "category": "B"})),
        )
        .await?;

    store
        .add(
            "doc4",
            vec![0.0, 0.0, 1.0],
            Some(serde_json::json!({"title": "Document 4", "category": "C"})),
        )
        .await?;

    store.add("doc5", vec![0.5, 0.5, 0.0], None).await?;

    // Query for similar vectors
    println!("\nQuerying for vectors similar to [1.0, 0.0, 0.0]...");
    let results = store.query(&[1.0, 0.0, 0.0], 3).await?;

    for result in &results {
        println!(
            "  ID: {}, Score: {:.4}, Metadata: {:?}",
            result.id, result.score, result.metadata
        );
    }

    println!("\nQuerying for vectors similar to [0.0, 1.0, 0.0]...");
    let results = store.query(&[0.0, 1.0, 0.0], 3).await?;

    for result in &results {
        println!(
            "  ID: {}, Score: {:.4}, Metadata: {:?}",
            result.id, result.score, result.metadata
        );
    }

    println!("\nQuerying for vectors similar to [0.5, 0.5, 0.0]...");
    let results = store.query(&[0.5, 0.5, 0.0], 3).await?;

    for result in &results {
        println!(
            "  ID: {}, Score: {:.4}, Metadata: {:?}",
            result.id, result.score, result.metadata
        );
    }

    store.close().await?;
    Ok(())
}
