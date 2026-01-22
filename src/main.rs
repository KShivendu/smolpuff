use object_store::memory::InMemory;
use object_store::ObjectStore;
use serde::{Deserialize, Serialize};
use slatedb::Db;
use std::collections::BinaryHeap;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Database error: {0}")]
    DbError(#[from] slatedb::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct ScoredItem {
    score: f32,
    id: String,
    metadata: Option<serde_json::Value>,
}

impl Eq for ScoredItem {}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap (we want to keep the highest scores)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub struct VectorStore {
    db: Db,
}

impl VectorStore {
    /// Open a vector store at the given path using the provided object store
    pub async fn open<P: AsRef<str>>(
        path: P,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<Self, VectorStoreError> {
        let db = Db::open(path.as_ref(), object_store).await?;
        Ok(Self { db })
    }

    /// Add a vector to the store
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The embedding vector
    /// * `metadata` - Optional JSON metadata associated with the vector
    pub async fn add(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), VectorStoreError> {
        let record = VectorRecord {
            id: id.to_string(),
            vector,
            metadata,
        };

        // Serialize the record using JSON
        let value = serde_json::to_vec(&record)?;

        // Use a prefix for vector records
        let key = format!("vec:{}", id);
        self.db.put(key.as_bytes(), &value).await?;

        Ok(())
    }

    /// Query for the k nearest vectors to the given query vector
    ///
    /// # Arguments
    /// * `query_vector` - The vector to search for
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// A vector of QueryResult sorted by similarity (highest first)
    pub async fn query(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<QueryResult>, VectorStoreError> {
        // Use a min-heap to keep track of top-k results
        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::new();

        // Scan all vectors with the "vec:" prefix
        let mut iter = self.db.scan("vec:".."vec;").await?; // ";" comes after ":" in ASCII

        while let Ok(Some(item)) = iter.next().await {
            // Deserialize the record
            let record: VectorRecord = serde_json::from_slice(&item.value)?;

            // Calculate cosine similarity
            let score = cosine_similarity(query_vector, &record.vector);

            let scored_item = ScoredItem {
                score,
                id: record.id,
                metadata: record.metadata,
            };

            if heap.len() < k {
                heap.push(scored_item);
            } else if let Some(min_item) = heap.peek() {
                if score > min_item.score {
                    heap.pop();
                    heap.push(scored_item);
                }
            }
        }

        // Convert heap to sorted results (highest score first)
        let mut results: Vec<QueryResult> = heap
            .into_iter()
            .map(|item| QueryResult {
                id: item.id,
                score: item.score,
                metadata: item.metadata,
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Close the database
    pub async fn close(self) -> Result<(), VectorStoreError> {
        self.db.close().await?;
        Ok(())
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use in-memory store for demo (replace with S3 for production)
    let object_store: Arc<dyn ObjectStore + 'static> = Arc::new(InMemory::new());

    // For S3, uncomment and configure:
    // let object_store: Arc<dyn ObjectStore> = Arc::new(
    //     object_store::aws::AmazonS3Builder::new()
    //         .with_endpoint("http://localhost:4566") // LocalStack or real S3
    //         .with_access_key_id("your-access-key")
    //         .with_secret_access_key("your-secret-key")
    //         .with_bucket_name("smolpuff-vectors")
    //         .with_region("us-east-1")
    //         .build()?,
    // );

    let store = VectorStore::open("/smolpuff/vectors", object_store).await?;

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
