use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::IndexMetadata;

pub struct Centroid {
    pub id: u32,
    pub vector: Vec<f32>,
}

pub struct NamespaceIndex {
    pub centroids: Vec<Centroid>,
    pub meta: IndexMetadata,
}

#[derive(Default)]
pub struct IndexManager {
    indices: RwLock<HashMap<String, Arc<NamespaceIndex>>>,
}

impl IndexManager {
    pub fn new() -> Self {
        Self {
            indices: RwLock::new(HashMap::new()),
        }
    }

    pub async fn get_index(&self, namespace: &str) -> Option<Arc<NamespaceIndex>> {
        self.indices.read().await.get(namespace).cloned()
    }

    pub async fn set_index(&self, namespace: &str, index: NamespaceIndex) {
        self.indices
            .write()
            .await
            .insert(namespace.to_string(), Arc::new(index));
    }

    pub async fn remove_index(&self, namespace: &str) {
        self.indices.write().await.remove(namespace);
    }
}
