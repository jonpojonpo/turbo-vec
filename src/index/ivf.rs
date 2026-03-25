use std::collections::BinaryHeap;

use rand::SeedableRng;
use rand::Rng;
use rayon::prelude::*;

use crate::quant::mse::{QuantizedMse, TurboQuantMse};
use crate::simd::portable::dot_f32;
use crate::simd::DistanceTable;
use crate::types::SearchResult;

/// Inverted File Index with TurboQuant compression.
///
/// Partitions vectors into clusters via k-means, then stores quantized
/// vectors within each cluster. Search probes the nearest clusters.
pub struct IvfIndex {
    quantizer: TurboQuantMse,
    /// Cluster centroids (full-precision float vectors).
    centroids: Vec<Vec<f32>>,
    /// Per-cluster storage: (external_id, quantized_vector).
    buckets: Vec<Vec<(u64, QuantizedMse)>>,
    n_clusters: usize,
    built: bool,
}

impl IvfIndex {
    /// Create a new IVF index.
    ///
    /// - `n_clusters`: number of Voronoi partitions (typical: sqrt(n))
    pub fn new(quantizer: TurboQuantMse, n_clusters: usize) -> Self {
        Self {
            quantizer,
            centroids: Vec::new(),
            buckets: Vec::new(),
            n_clusters,
            built: false,
        }
    }

    /// Build the index from a dataset.
    ///
    /// Runs k-means to find cluster centroids, then assigns and quantizes
    /// each vector into its nearest cluster.
    pub fn build(&mut self, ids: &[u64], vectors: &[&[f32]]) {
        assert_eq!(ids.len(), vectors.len());
        assert!(!vectors.is_empty());

        let dim = self.quantizer.dim();

        // Run k-means to find centroids
        self.centroids = kmeans(vectors, self.n_clusters, dim, 20, 42);
        self.buckets = vec![Vec::new(); self.n_clusters];

        // Assign each vector to its nearest centroid and quantize
        for (i, &vec) in vectors.iter().enumerate() {
            let cluster = nearest_centroid(vec, &self.centroids);
            let quantized = self.quantizer.quantize(vec);
            self.buckets[cluster].push((ids[i], quantized));
        }

        self.built = true;
    }

    /// Search for top-k nearest neighbors.
    ///
    /// - `n_probe`: number of clusters to search (higher = more accurate)
    pub fn search(&self, query: &[f32], top_k: usize, n_probe: usize) -> Vec<SearchResult> {
        if !self.built || self.centroids.is_empty() {
            return Vec::new();
        }

        // Find the n_probe nearest cluster centroids
        let mut centroid_scores: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot_f32(query, c)))
            .collect();
        centroid_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build distance table
        let q_vec = nalgebra::DVector::from_column_slice(query);
        let q_rot = self.quantizer.rotation() * &q_vec;
        let q_rot_slice: Vec<f32> = q_rot.iter().copied().collect();
        let dist_table =
            DistanceTable::from_rotated_query(&q_rot_slice, self.quantizer.codebook());

        // Search the nearest n_probe clusters
        let mut heap = BinaryHeap::with_capacity(top_k + 1);

        for &(cluster_idx, _) in centroid_scores.iter().take(n_probe) {
            for (id, qvec) in &self.buckets[cluster_idx] {
                let score = dist_table.dot(&qvec.packed_indices, qvec.norm);
                heap.push(SearchResult { id: *id, score });
                if heap.len() > top_k {
                    heap.pop();
                }
            }
        }

        let mut results: Vec<SearchResult> = heap.into_vec();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.is_empty())
    }
}

/// Simple k-means clustering.
fn kmeans(
    vectors: &[&[f32]],
    k: usize,
    dim: usize,
    max_iters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = k.min(n);

    // Initialize centroids via k-means++ style: random selection
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut centroid_indices: Vec<usize> = Vec::with_capacity(k);
    centroid_indices.push(rng.random_range(0..n));

    for _ in 1..k {
        // Pick the point farthest from existing centroids (simplified)
        let mut best_idx = 0;
        let mut best_dist = f32::NEG_INFINITY;
        for i in 0..n {
            let min_dist = centroid_indices
                .iter()
                .map(|&ci| dot_f32(vectors[i], vectors[ci]))
                .fold(f32::INFINITY, f32::min);
            // We want maximum distance (minimum similarity)
            let neg_sim = -min_dist;
            if neg_sim > best_dist {
                best_dist = neg_sim;
                best_idx = i;
            }
        }
        centroid_indices.push(best_idx);
    }

    let mut centroids: Vec<Vec<f32>> = centroid_indices
        .iter()
        .map(|&i| vectors[i].to_vec())
        .collect();

    // Lloyd's iterations
    for _ in 0..max_iters {
        // Assignment step
        let assignments: Vec<usize> = vectors
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids))
            .collect();

        // Update step
        let mut new_centroids = vec![vec![0.0_f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            for (j, &val) in vectors[i].iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }

        let mut converged = true;
        for (ci, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[ci] > 0 {
                for v in centroid.iter_mut() {
                    *v /= counts[ci] as f32;
                }
            } else {
                // Empty cluster: reinitialize to a random point
                let idx = rng.random_range(0..n);
                *centroid = vectors[idx].to_vec();
            }
            // Check convergence
            let shift: f32 = centroid
                .iter()
                .zip(centroids[ci].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if shift > 1e-6 {
                converged = false;
            }
        }

        centroids = new_centroids;
        if converged {
            break;
        }
    }

    centroids
}

/// Find the index of the nearest centroid (by inner product).
fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, dot_f32(vector, c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn test_ivf_search() {
        let dim = 64;
        let quantizer = TurboQuantMse::new(dim, 4, 42);
        let mut index = IvfIndex::new(quantizer, 4);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = StandardNormal;

        let n = 100;
        let raw_vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let vec_refs: Vec<&[f32]> = raw_vecs.iter().map(|v| v.as_slice()).collect();

        index.build(&ids, &vec_refs);

        let query: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let results = index.search(&query, 5, 2);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Scores should be descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
