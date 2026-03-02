/// Binary-serialized posting list for a centroid.
///
/// Layout: [num_entries:u32] then per entry: [id_len:u16][id_bytes][vector:dim*4 bytes]
#[derive(Default)]
pub struct PostingList {
    pub entries: Vec<PostingEntry>,
}

pub struct PostingEntry {
    pub id: String,
    pub vector: Vec<f32>,
}

impl PostingList {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            let id_bytes = entry.id.as_bytes();
            buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(id_bytes);
            for &val in &entry.vector {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        buf
    }

    pub fn deserialize(bytes: &[u8], dim: usize) -> Self {
        let mut pos = 0;
        if bytes.len() < 4 {
            return Self::new();
        }
        let num_entries =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            if pos + 2 > bytes.len() {
                break;
            }
            let id_len = u16::from_le_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            if pos + id_len > bytes.len() {
                break;
            }
            let id = String::from_utf8_lossy(&bytes[pos..pos + id_len]).to_string();
            pos += id_len;

            let vec_bytes = dim * 4;
            if pos + vec_bytes > bytes.len() {
                break;
            }
            let vector: Vec<f32> = bytes[pos..pos + vec_bytes]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            pos += vec_bytes;

            entries.push(PostingEntry { id, vector });
        }

        Self { entries }
    }

    /// Remove entry by id (for upsert dedup), returns whether it was found.
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() < before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_list_roundtrip() {
        let pl = PostingList {
            entries: vec![
                PostingEntry {
                    id: "a".to_string(),
                    vector: vec![1.0, 2.0, 3.0],
                },
                PostingEntry {
                    id: "bb".to_string(),
                    vector: vec![4.0, 5.0, 6.0],
                },
            ],
        };
        let bytes = pl.serialize();
        let pl2 = PostingList::deserialize(&bytes, 3);
        assert_eq!(pl2.entries.len(), 2);
        assert_eq!(pl2.entries[0].id, "a");
        assert_eq!(pl2.entries[0].vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(pl2.entries[1].id, "bb");
        assert_eq!(pl2.entries[1].vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_posting_list_empty() {
        let pl = PostingList::new();
        let bytes = pl.serialize();
        let pl2 = PostingList::deserialize(&bytes, 3);
        assert_eq!(pl2.entries.len(), 0);
    }

    #[test]
    fn test_remove_by_id() {
        let mut pl = PostingList {
            entries: vec![
                PostingEntry {
                    id: "a".to_string(),
                    vector: vec![1.0],
                },
                PostingEntry {
                    id: "b".to_string(),
                    vector: vec![2.0],
                },
            ],
        };
        assert!(pl.remove_by_id("a"));
        assert_eq!(pl.entries.len(), 1);
        assert_eq!(pl.entries[0].id, "b");
        assert!(!pl.remove_by_id("a"));
    }
}
