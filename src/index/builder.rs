use chrono::Utc;
use slatedb::Db;

use crate::errors::VectorStoreError;
use crate::models::{IndexMetadata, IndexStatus};

use super::kmeans::{choose_num_centroids, kmeans};
use super::manager::{Centroid, IndexManager, NamespaceIndex};
use super::posting::{PostingEntry, PostingList};

/// Build an IVF index for a namespace by reading all vectors, running k-means,
/// and writing centroids + posting lists to the DB.
pub async fn build_index(
    db: &Db,
    index_manager: &IndexManager,
    namespace: &str,
    dim: usize,
) -> Result<IndexMetadata, VectorStoreError> {
    // 1. Read all vectors from the namespace
    let vec_prefix = format!("ns:{namespace}:vec:");
    let vec_end = format!("ns:{namespace}:vec;");
    let mut ids = Vec::new();
    let mut vectors = Vec::new();

    let mut iter = db.scan(vec_prefix.as_bytes()..vec_end.as_bytes()).await?;
    while let Ok(Some(item)) = iter.next().await {
        let key_str = String::from_utf8_lossy(&item.key);
        let id = key_str.strip_prefix(&vec_prefix).unwrap_or("").to_string();
        let vector = decode_f32_vec(&item.value);
        ids.push(id);
        vectors.push(vector);
    }

    if vectors.is_empty() {
        return Err(VectorStoreError::InvalidRequest(
            "No vectors in namespace to index".to_string(),
        ));
    }

    let num_vectors = vectors.len();
    let num_centroids = choose_num_centroids(num_vectors);

    // 2. Write "Building" status
    let meta = IndexMetadata {
        status: IndexStatus::Building,
        num_centroids: num_centroids as u32,
        vector_dim: dim,
        num_indexed_vectors: num_vectors as u64,
        built_at: Utc::now(),
        version: 1,
    };
    let meta_key = format!("ns:{namespace}:idx:meta");
    db.put(meta_key.as_bytes(), &serde_json::to_vec(&meta)?)
        .await?;

    // 3. Run k-means
    let centroid_vectors = kmeans(&vectors, num_centroids, 20);
    let actual_k = centroid_vectors.len();

    // 4. Assign vectors to nearest centroid and build posting lists
    let mut posting_lists: Vec<PostingList> = (0..actual_k).map(|_| PostingList::new()).collect();

    for (i, vec) in vectors.iter().enumerate() {
        let nearest = find_nearest_centroid(vec, &centroid_vectors);
        posting_lists[nearest].entries.push(PostingEntry {
            id: ids[i].clone(),
            vector: vec.clone(),
        });
    }

    // 5. Write centroids and posting lists to DB
    for (c_id, centroid_vec) in centroid_vectors.iter().enumerate() {
        let centroid_key = format!("ns:{namespace}:idx:centroid:{c_id}");
        let centroid_bytes: Vec<u8> = centroid_vec.iter().flat_map(|f| f.to_le_bytes()).collect();
        db.put(centroid_key.as_bytes(), &centroid_bytes).await?;

        let posting_key = format!("ns:{namespace}:idx:posting:{c_id}");
        let posting_bytes = posting_lists[c_id].serialize();
        db.put(posting_key.as_bytes(), &posting_bytes).await?;
    }

    // 6. Update metadata to Ready
    let meta = IndexMetadata {
        status: IndexStatus::Ready,
        num_centroids: actual_k as u32,
        vector_dim: dim,
        num_indexed_vectors: num_vectors as u64,
        built_at: Utc::now(),
        version: 1,
    };
    db.put(meta_key.as_bytes(), &serde_json::to_vec(&meta)?)
        .await?;

    // 7. Load into in-memory IndexManager
    let centroids: Vec<Centroid> = centroid_vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| Centroid {
            id: i as u32,
            vector: v,
        })
        .collect();

    let ns_index = NamespaceIndex {
        centroids,
        meta: meta.clone(),
    };
    index_manager.set_index(namespace, ns_index).await;

    tracing::info!(
        namespace,
        num_vectors,
        num_centroids = actual_k,
        "Index built"
    );

    Ok(meta)
}

/// Load existing indices from DB on startup.
pub async fn load_existing_indices(
    db: &Db,
    index_manager: &IndexManager,
) -> Result<(), VectorStoreError> {
    // Scan for all index metadata keys
    let prefix = b"ns:";
    let end = b"ns;";
    let mut iter = db.scan(prefix.as_slice()..end.as_slice()).await?;

    while let Ok(Some(item)) = iter.next().await {
        let key_str = String::from_utf8_lossy(&item.key);
        if !key_str.ends_with(":idx:meta") {
            continue;
        }

        let meta: IndexMetadata = match serde_json::from_slice(&item.value) {
            Ok(m) => m,
            Err(_) => continue,
        };

        if meta.status != IndexStatus::Ready {
            continue;
        }

        // Extract namespace name: "ns:{ns}:idx:meta"
        let ns = match key_str
            .strip_prefix("ns:")
            .and_then(|s| s.strip_suffix(":idx:meta"))
        {
            Some(ns) => ns.to_string(),
            None => continue,
        };

        // Load centroids
        let mut centroids = Vec::with_capacity(meta.num_centroids as usize);
        for c_id in 0..meta.num_centroids {
            let centroid_key = format!("ns:{ns}:idx:centroid:{c_id}");
            match db.get(centroid_key.as_bytes()).await? {
                Some(bytes) => {
                    let vector = decode_f32_vec(&bytes);
                    centroids.push(Centroid { id: c_id, vector });
                }
                None => {
                    tracing::warn!(namespace = %ns, centroid_id = c_id, "Missing centroid, skipping index");
                    break;
                }
            }
        }

        if centroids.len() == meta.num_centroids as usize {
            let ns_index = NamespaceIndex {
                centroids,
                meta: meta.clone(),
            };
            index_manager.set_index(&ns, ns_index).await;
            tracing::info!(
                namespace = %ns,
                num_centroids = meta.num_centroids,
                num_vectors = meta.num_indexed_vectors,
                "Loaded index from DB"
            );
        }
    }

    Ok(())
}

fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d: f32 = vector
            .iter()
            .zip(c.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

pub fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn encode_f32_vec(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|f| f.to_le_bytes()).collect()
}
