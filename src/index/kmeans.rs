/// Run Lloyd's k-means with k-means++ initialization.
/// Returns centroids as Vec<Vec<f32>>.
pub fn kmeans(vectors: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(vectors.len());
    let dim = vectors[0].len();

    // K-means++ initialization
    let mut centroids = kmeans_pp_init(vectors, k);

    for _ in 0..max_iter {
        // Assign each vector to nearest centroid
        let mut assignments = vec![0usize; vectors.len()];
        for (i, v) in vectors.iter().enumerate() {
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = squared_euclidean(v, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        // Recompute centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0u64; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &val) in v.iter().enumerate() {
                new_centroids[c][j] += val;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for val in new_centroids[c].iter_mut() {
                    *val /= counts[c] as f32;
                }
            } else {
                // Keep old centroid for empty clusters
                new_centroids[c] = centroids[c].clone();
            }
        }

        // Check convergence
        let max_shift: f32 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| squared_euclidean(old, new))
            .fold(0.0f32, f32::max);

        centroids = new_centroids;

        if max_shift < 1e-4 {
            break;
        }
    }

    centroids
}

fn kmeans_pp_init(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = vectors.len();
    let mut centroids = Vec::with_capacity(k);

    // Pick first centroid randomly
    let first = rng.gen_range(0..n);
    centroids.push(vectors[first].clone());

    let mut min_dists = vec![f32::MAX; n];

    for _ in 1..k {
        // Update min distances to nearest chosen centroid
        let last = centroids.last().unwrap();
        for (i, v) in vectors.iter().enumerate() {
            let d = squared_euclidean(v, last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }

        // Sample proportional to distance squared
        let total: f32 = min_dists.iter().sum();
        if total <= 0.0 {
            centroids.push(vectors[rng.gen_range(0..n)].clone());
            continue;
        }
        let threshold = rng.gen_range(0.0..total);
        let mut cumulative = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen].clone());
    }

    centroids
}

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Determine number of centroids: sqrt(n) clamped to [4, 4096]
pub fn choose_num_centroids(num_vectors: usize) -> usize {
    let k = (num_vectors as f64).sqrt() as usize;
    k.clamp(4, 4096)
}

/// Determine n_probe: sqrt(num_centroids) clamped to [1, 20]
pub fn choose_n_probe(num_centroids: usize) -> usize {
    let p = (num_centroids as f64).sqrt() as usize;
    p.clamp(1, 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Two obvious clusters
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![9.9, 9.9],
        ];
        let centroids = kmeans(&vectors, 2, 20);
        assert_eq!(centroids.len(), 2);

        // Each centroid should be near one cluster
        let mut near_zero = false;
        let mut near_ten = false;
        for c in &centroids {
            if c[0] < 1.0 {
                near_zero = true;
            }
            if c[0] > 9.0 {
                near_ten = true;
            }
        }
        assert!(near_zero && near_ten);
    }

    #[test]
    fn test_choose_num_centroids() {
        assert_eq!(choose_num_centroids(4), 4); // sqrt(4)=2, clamped to 4
        assert_eq!(choose_num_centroids(100), 10);
        assert_eq!(choose_num_centroids(1_000_000), 1000);
        assert_eq!(choose_num_centroids(100_000_000), 4096); // clamped
    }

    #[test]
    fn test_choose_n_probe() {
        assert_eq!(choose_n_probe(1), 1);
        assert_eq!(choose_n_probe(100), 10);
        assert_eq!(choose_n_probe(1000), 20); // clamped
    }
}
