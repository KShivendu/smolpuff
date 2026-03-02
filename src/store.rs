use crate::errors::VectorStoreError;
use crate::index::builder::{build_index, decode_f32_vec, encode_f32_vec, load_existing_indices};
use crate::index::kmeans::choose_n_probe;
use crate::index::manager::IndexManager;
use crate::index::posting::{PostingEntry, PostingList};
use crate::models::{IndexMetadata, IndexStatus, NamespaceMetadata};
use chrono::Utc;
use metrics::{counter, histogram};
use object_store::ObjectStore;
use slatedb::Db;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

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
    index_manager: Arc<IndexManager>,
}

fn record_op(operation: &str, namespace: &str, start: Instant, succeeded: bool) {
    let status = if succeeded { "ok" } else { "error" };
    let labels = [
        ("operation", operation.to_string()),
        ("namespace", namespace.to_string()),
        ("status", status.to_string()),
    ];
    counter!("store_operations_total", &labels).increment(1);
    histogram!(
        "store_operation_duration_seconds",
        &[
            ("operation", operation.to_string()),
            ("namespace", namespace.to_string()),
        ]
    )
    .record(start.elapsed().as_secs_f64());
}

impl VectorStore {
    pub async fn open<P: AsRef<str>>(
        path: P,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<Self, VectorStoreError> {
        let db = Db::open(path.as_ref(), object_store).await?;
        let index_manager = Arc::new(IndexManager::new());

        // Load existing indices from DB
        load_existing_indices(&db, &index_manager).await?;

        Ok(Self { db, index_manager })
    }

    // --- Namespace operations ---

    pub async fn create_namespace(
        &self,
        name: &str,
        vector_dim: usize,
        distance: &str,
    ) -> Result<NamespaceMetadata, VectorStoreError> {
        let start = Instant::now();
        let result = async {
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
        .await;
        record_op("create_namespace", name, start, result.is_ok());
        result
    }

    pub async fn get_namespace(&self, name: &str) -> Result<NamespaceMetadata, VectorStoreError> {
        let start = Instant::now();
        let result = async {
            let meta_key = format!("ns:{name}:meta");
            match self.db.get(meta_key.as_bytes()).await? {
                Some(value) => Ok(serde_json::from_slice(&value)?),
                None => Err(VectorStoreError::NamespaceNotFound(name.to_string())),
            }
        }
        .await;
        record_op("get_namespace", name, start, result.is_ok());
        result
    }

    pub async fn delete_namespace(&self, name: &str) -> Result<(), VectorStoreError> {
        let start = Instant::now();
        let result = async {
            let meta_key = format!("ns:{name}:meta");

            // Verify namespace exists
            if self.db.get(meta_key.as_bytes()).await?.is_none() {
                return Err(VectorStoreError::NamespaceNotFound(name.to_string()));
            }

            // Delete all vec keys for this namespace
            let vec_prefix = format!("ns:{name}:vec:");
            let vec_end = format!("ns:{name}:vec;");
            let mut iter = self
                .db
                .scan(vec_prefix.as_bytes()..vec_end.as_bytes())
                .await?;
            while let Ok(Some(item)) = iter.next().await {
                self.db.delete(&item.key).await?;
            }

            // Delete all doc keys for this namespace
            let doc_prefix = format!("ns:{name}:doc:");
            let doc_end = format!("ns:{name}:doc;");
            let mut iter = self
                .db
                .scan(doc_prefix.as_bytes()..doc_end.as_bytes())
                .await?;
            while let Ok(Some(item)) = iter.next().await {
                self.db.delete(&item.key).await?;
            }

            // Delete index keys (centroids, posting lists, meta)
            let idx_prefix = format!("ns:{name}:idx:");
            let idx_end = format!("ns:{name}:idx;");
            let mut iter = self
                .db
                .scan(idx_prefix.as_bytes()..idx_end.as_bytes())
                .await?;
            while let Ok(Some(item)) = iter.next().await {
                self.db.delete(&item.key).await?;
            }

            // Remove from in-memory index manager
            self.index_manager.remove_index(name).await;

            // Delete the metadata key
            self.db.delete(meta_key.as_bytes()).await?;

            Ok(())
        }
        .await;
        record_op("delete_namespace", name, start, result.is_ok());
        result
    }

    // --- Data operations ---

    pub async fn upsert(
        &self,
        ns: &str,
        id: &str,
        vector: Vec<f32>,
        attributes: Option<serde_json::Value>,
    ) -> Result<(), VectorStoreError> {
        let start = Instant::now();
        let result = async {
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
            let vec_bytes = encode_f32_vec(&vector);
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

            // Append to posting list if index exists
            self.append_to_posting_list(ns, id, &vector).await?;

            Ok(())
        }
        .await;
        record_op("upsert", ns, start, result.is_ok());
        result
    }

    /// Append a vector to the nearest centroid's posting list (if an index exists).
    async fn append_to_posting_list(
        &self,
        ns: &str,
        id: &str,
        vector: &[f32],
    ) -> Result<(), VectorStoreError> {
        let ns_index = match self.index_manager.get_index(ns).await {
            Some(idx) => idx,
            None => return Ok(()), // No index, nothing to do
        };

        // Find nearest centroid
        let mut best_c = 0u32;
        let mut best_sim = f32::NEG_INFINITY;
        for c in &ns_index.centroids {
            let sim = cosine_similarity(vector, &c.vector);
            if sim > best_sim {
                best_sim = sim;
                best_c = c.id;
            }
        }

        // Read-modify-write the posting list
        let posting_key = format!("ns:{ns}:idx:posting:{best_c}");
        let mut posting_list = match self.db.get(posting_key.as_bytes()).await? {
            Some(bytes) => PostingList::deserialize(&bytes, ns_index.meta.vector_dim),
            None => PostingList::new(),
        };

        // Remove existing entry with same ID (upsert semantics)
        posting_list.remove_by_id(id);

        // Append new entry
        posting_list.entries.push(PostingEntry {
            id: id.to_string(),
            vector: vector.to_vec(),
        });

        // Write back
        self.db
            .put(posting_key.as_bytes(), &posting_list.serialize())
            .await?;

        Ok(())
    }

    pub async fn query_ns(
        &self,
        ns: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<crate::models::QueryResultItem>, VectorStoreError> {
        let start = Instant::now();
        let result = async {
            // Verify namespace exists
            let meta = self.get_namespace(ns).await?;

            if meta.vector_dim > 0 && query_vector.len() != meta.vector_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    expected: meta.vector_dim,
                    got: query_vector.len(),
                });
            }

            // Try index-accelerated query first
            if let Some(ns_index) = self.index_manager.get_index(ns).await
                && ns_index.meta.status == IndexStatus::Ready
            {
                return self
                    .query_with_index(ns, query_vector, top_k, &ns_index)
                    .await;
            }

            // Fall back to brute-force
            self.query_brute_force(ns, query_vector, top_k).await
        }
        .await;
        record_op("query", ns, start, result.is_ok());
        result
    }

    async fn query_with_index(
        &self,
        ns: &str,
        query_vector: &[f32],
        top_k: usize,
        ns_index: &crate::index::manager::NamespaceIndex,
    ) -> Result<Vec<crate::models::QueryResultItem>, VectorStoreError> {
        // 1. Find nearest centroids
        let n_probe = choose_n_probe(ns_index.centroids.len());
        let mut centroid_scores: Vec<(u32, f32)> = ns_index
            .centroids
            .iter()
            .map(|c| (c.id, cosine_similarity(query_vector, &c.vector)))
            .collect();
        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Fetch posting lists for top n_probe centroids
        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::new();

        for &(c_id, _) in centroid_scores.iter().take(n_probe) {
            let posting_key = format!("ns:{ns}:idx:posting:{c_id}");
            let posting_list = match self.db.get(posting_key.as_bytes()).await? {
                Some(bytes) => PostingList::deserialize(&bytes, ns_index.meta.vector_dim),
                None => continue,
            };

            // 3. Score all candidates
            for entry in &posting_list.entries {
                let score = cosine_similarity(query_vector, &entry.vector);
                let scored = ScoredItem {
                    score,
                    id: entry.id.clone(),
                };
                if heap.len() < top_k {
                    heap.push(scored);
                } else if let Some(min_item) = heap.peek()
                    && score > min_item.score
                {
                    heap.pop();
                    heap.push(scored);
                }
            }
        }

        // 4. Collect results sorted by score descending
        let mut scored_ids: Vec<ScoredItem> = heap.into_iter().collect();
        scored_ids.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 5. Fetch attributes for top-k results
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

    async fn query_brute_force(
        &self,
        ns: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<Vec<crate::models::QueryResultItem>, VectorStoreError> {
        let mut heap: BinaryHeap<ScoredItem> = BinaryHeap::new();

        // Scan all vectors in this namespace
        let vec_prefix = format!("ns:{ns}:vec:");
        let vec_end = format!("ns:{ns}:vec;");
        let mut iter = self
            .db
            .scan(vec_prefix.as_bytes()..vec_end.as_bytes())
            .await?;

        while let Ok(Some(item)) = iter.next().await {
            let key_str = String::from_utf8_lossy(&item.key);
            let id = key_str.strip_prefix(&vec_prefix).unwrap_or("").to_string();

            let vec_data = decode_f32_vec(&item.value);
            let score = cosine_similarity(query_vector, &vec_data);

            let scored = ScoredItem { score, id };

            if heap.len() < top_k {
                heap.push(scored);
            } else if let Some(min_item) = heap.peek()
                && score > min_item.score
            {
                heap.pop();
                heap.push(scored);
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

    // --- Index operations ---

    pub async fn build_index(&self, ns: &str) -> Result<IndexMetadata, VectorStoreError> {
        let meta = self.get_namespace(ns).await?;
        build_index(&self.db, &self.index_manager, ns, meta.vector_dim).await
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
