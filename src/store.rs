use crate::errors::VectorStoreError;
use crate::models::NamespaceMetadata;
use chrono::Utc;
use object_store::ObjectStore;
use slatedb::Db;
use std::collections::BinaryHeap;
use std::sync::Arc;

const DEFAULT_NS: &str = "_default";
const DEFAULT_DIM: usize = 0; // 0 means "any dimension" for backward compat

#[derive(Debug, Clone, PartialEq)]
struct ScoredItem {
    score: f32,
    id: String,
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
    pub async fn open<P: AsRef<str>>(
        path: P,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<Self, VectorStoreError> {
        let db = Db::open(path.as_ref(), object_store).await?;
        Ok(Self { db })
    }

    // --- Namespace operations ---

    pub async fn create_namespace(
        &self,
        name: &str,
        vector_dim: usize,
        distance: &str,
    ) -> Result<NamespaceMetadata, VectorStoreError> {
        let meta_key = format!("ns:{name}:meta");

        // Check if namespace already exists
        if self.db.get(meta_key.as_bytes()).await?.is_some() {
            return Err(VectorStoreError::NamespaceAlreadyExists(name.to_string()));
        }

        let metadata = NamespaceMetadata {
            name: name.to_string(),
            vector_dim,
            distance: distance.to_string(),
            approx_row_count: 0,
            created_at: Utc::now(),
        };

        let value = serde_json::to_vec(&metadata)?;
        self.db.put(meta_key.as_bytes(), &value).await?;

        Ok(metadata)
    }

    pub async fn get_namespace(
        &self,
        name: &str,
    ) -> Result<NamespaceMetadata, VectorStoreError> {
        let meta_key = format!("ns:{name}:meta");

        match self.db.get(meta_key.as_bytes()).await? {
            Some(value) => Ok(serde_json::from_slice(&value)?),
            None => Err(VectorStoreError::NamespaceNotFound(name.to_string())),
        }
    }

    pub async fn delete_namespace(&self, name: &str) -> Result<(), VectorStoreError> {
        let meta_key = format!("ns:{name}:meta");

        // Verify namespace exists
        if self.db.get(meta_key.as_bytes()).await?.is_none() {
            return Err(VectorStoreError::NamespaceNotFound(name.to_string()));
        }

        // Delete all vec keys for this namespace
        let vec_prefix = format!("ns:{name}:vec:");
        let vec_end = format!("ns:{name}:vec;");
        let mut iter = self.db.scan(vec_prefix.as_bytes()..vec_end.as_bytes()).await?;
        while let Ok(Some(item)) = iter.next().await {
            self.db.delete(&item.key).await?;
        }

        // Delete all doc keys for this namespace
        let doc_prefix = format!("ns:{name}:doc:");
        let doc_end = format!("ns:{name}:doc;");
        let mut iter = self.db.scan(doc_prefix.as_bytes()..doc_end.as_bytes()).await?;
        while let Ok(Some(item)) = iter.next().await {
            self.db.delete(&item.key).await?;
        }

        // Delete the metadata key
        self.db.delete(meta_key.as_bytes()).await?;

        Ok(())
    }

    // --- Data operations ---

    pub async fn upsert(
        &self,
        ns: &str,
        id: &str,
        vector: Vec<f32>,
        attributes: Option<serde_json::Value>,
    ) -> Result<(), VectorStoreError> {
        // Get namespace metadata to validate dimensions
        let meta = self.get_namespace(ns).await?;

        if meta.vector_dim > 0 && vector.len() != meta.vector_dim {
            return Err(VectorStoreError::DimensionMismatch {
                expected: meta.vector_dim,
                got: vector.len(),
            });
        }

        // Store vector as raw f32 le_bytes
        let vec_key = format!("ns:{ns}:vec:{id}");
        let vec_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.db.put(vec_key.as_bytes(), &vec_bytes).await?;

        // Store attributes separately as JSON
        let doc_key = format!("ns:{ns}:doc:{id}");
        if let Some(attrs) = &attributes {
            let doc_bytes = serde_json::to_vec(attrs)?;
            self.db.put(doc_key.as_bytes(), &doc_bytes).await?;
        }

        // Update approx row count (best effort — not atomic)
        let meta_key = format!("ns:{ns}:meta");
        let updated_meta = NamespaceMetadata {
            approx_row_count: meta.approx_row_count + 1,
            ..meta
        };
        let meta_bytes = serde_json::to_vec(&updated_meta)?;
        self.db.put(meta_key.as_bytes(), &meta_bytes).await?;

        Ok(())
    }

    pub async fn query_ns(
        &self,
        ns: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<crate::models::QueryResultItem>, VectorStoreError> {
        // Verify namespace exists
        let meta = self.get_namespace(ns).await?;

        if meta.vector_dim > 0 && query_vector.len() != meta.vector_dim {
            return Err(VectorStoreError::DimensionMismatch {
                expected: meta.vector_dim,
                got: query_vector.len(),
            });
        }

        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::new();

        // Scan all vectors in this namespace
        let vec_prefix = format!("ns:{ns}:vec:");
        let vec_end = format!("ns:{ns}:vec;");
        let mut iter = self.db.scan(vec_prefix.as_bytes()..vec_end.as_bytes()).await?;

        while let Ok(Some(item)) = iter.next().await {
            // Extract id from key: "ns:{ns}:vec:{id}"
            let key_str = String::from_utf8_lossy(&item.key);
            let id = key_str
                .strip_prefix(&vec_prefix)
                .unwrap_or("")
                .to_string();

            // Decode vector from le_bytes
            let vec_data = decode_f32_vec(&item.value);
            let score = cosine_similarity(query_vector, &vec_data);

            let scored = ScoredItem { score, id };

            if heap.len() < top_k {
                heap.push(scored);
            } else if let Some(min_item) = heap.peek() {
                if score > min_item.score {
                    heap.pop();
                    heap.push(scored);
                }
            }
        }

        // Collect top-k IDs, sorted by score descending
        let mut scored_ids: Vec<ScoredItem> = heap.into_iter().collect();
        scored_ids.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fetch attributes only for top-k results
        let mut results = Vec::with_capacity(scored_ids.len());
        for si in scored_ids {
            let doc_key = format!("ns:{ns}:doc:{}", si.id);
            let attributes = match self.db.get(doc_key.as_bytes()).await? {
                Some(val) => Some(serde_json::from_slice(&val)?),
                None => None,
            };
            results.push(crate::models::QueryResultItem {
                id: si.id,
                score: si.score,
                attributes,
            });
        }

        Ok(results)
    }

    // --- Backward-compatible methods for benchmarks ---

    pub async fn add(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), VectorStoreError> {
        // Ensure _default namespace exists
        self.ensure_default_namespace(vector.len()).await?;
        self.upsert(DEFAULT_NS, id, vector, metadata).await
    }

    pub async fn query(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<QueryResult>, VectorStoreError> {
        // Ensure _default namespace exists
        self.ensure_default_namespace(DEFAULT_DIM).await?;

        let items = self.query_ns(DEFAULT_NS, query_vector, k).await?;
        Ok(items
            .into_iter()
            .map(|item| QueryResult {
                id: item.id,
                score: item.score,
                metadata: item.attributes,
            })
            .collect())
    }

    async fn ensure_default_namespace(&self, dim: usize) -> Result<(), VectorStoreError> {
        let meta_key = format!("ns:{DEFAULT_NS}:meta");
        if self.db.get(meta_key.as_bytes()).await?.is_none() {
            self.create_namespace(DEFAULT_NS, dim, "cosine").await?;
        }
        Ok(())
    }

    pub async fn close(self) -> Result<(), VectorStoreError> {
        self.db.close().await?;
        Ok(())
    }
}

/// Backward-compatible QueryResult for benchmarks
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub id: String,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

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

fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
