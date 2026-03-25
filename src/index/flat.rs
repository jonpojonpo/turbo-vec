use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::quant::mse::{QuantizedMse, TurboQuantMse};
use crate::simd::DistanceTable;
use crate::types::SearchResult;

/// Brute-force flat index using TurboQuant compression.
///
/// Stores all vectors in quantized form and scans all of them for each query.
/// Suitable for small-to-medium datasets or as a correctness baseline.
pub struct FlatIndex {
    quantizer: TurboQuantMse,
    vectors: Vec<QuantizedMse>,
    ids: Vec<u64>,
}

impl FlatIndex {
    pub fn new(quantizer: TurboQuantMse) -> Self {
        Self {
            quantizer,
            vectors: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Add a single vector with an ID.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        let quantized = self.quantizer.quantize(vector);
        self.vectors.push(quantized);
        self.ids.push(id);
    }

    /// Add a batch of vectors.
    pub fn add_batch(&mut self, ids: &[u64], vectors: &[&[f32]]) {
        assert_eq!(ids.len(), vectors.len());
        let quantized: Vec<QuantizedMse> = vectors
            .par_iter()
            .map(|v| self.quantizer.quantize(v))
            .collect();
        self.vectors.extend(quantized);
        self.ids.extend_from_slice(ids);
    }

    /// Search for top-k nearest neighbors by inner product.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Pre-rotate query and build distance table
        let q_vec = nalgebra::DVector::from_column_slice(query);
        let q_rot = self.quantizer.rotation() * &q_vec;
        let q_rot_slice: Vec<f32> = q_rot.iter().copied().collect();
        let dist_table = DistanceTable::from_rotated_query(&q_rot_slice, self.quantizer.codebook());

        // Score all vectors
        let n = self.vectors.len();
        if n > 10_000 {
            // Parallel scoring for large datasets
            self.search_parallel(&dist_table, top_k)
        } else {
            self.search_sequential(&dist_table, top_k)
        }
    }

    fn search_sequential(&self, dist_table: &DistanceTable, top_k: usize) -> Vec<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(top_k + 1);

        for (i, qvec) in self.vectors.iter().enumerate() {
            let score = dist_table.dot(&qvec.packed_indices, qvec.norm);
            let result = SearchResult {
                id: self.ids[i],
                score,
            };
            heap.push(result);
            if heap.len() > top_k {
                heap.pop(); // Remove the lowest score
            }
        }

        let mut results: Vec<SearchResult> = heap.into_vec();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn search_parallel(&self, dist_table: &DistanceTable, top_k: usize) -> Vec<SearchResult> {
        // Parallel scoring, then merge
        let chunk_size = (self.vectors.len() + rayon::current_num_threads() - 1)
            / rayon::current_num_threads();

        let partial_results: Vec<Vec<SearchResult>> = self
            .vectors
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base = chunk_idx * chunk_size;
                let mut heap = BinaryHeap::with_capacity(top_k + 1);
                for (i, qvec) in chunk.iter().enumerate() {
                    let score = dist_table.dot(&qvec.packed_indices, qvec.norm);
                    heap.push(SearchResult {
                        id: self.ids[base + i],
                        score,
                    });
                    if heap.len() > top_k {
                        heap.pop();
                    }
                }
                heap.into_vec()
            })
            .collect();

        // Merge partial results
        let mut final_heap = BinaryHeap::with_capacity(top_k + 1);
        for partial in partial_results {
            for r in partial {
                final_heap.push(r);
                if final_heap.len() > top_k {
                    final_heap.pop();
                }
            }
        }

        let mut results: Vec<SearchResult> = final_heap.into_vec();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::portable::dot_f32;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn test_flat_index_search() {
        let dim = 64;
        let quantizer = TurboQuantMse::new(dim, 4, 42);
        let mut index = FlatIndex::new(quantizer);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = StandardNormal;

        let n = 100;
        let mut raw_vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            raw_vecs.push(v.clone());
            index.add(i as u64, &v);
        }

        // Query
        let query: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let results = index.search(&query, 5);

        assert_eq!(results.len(), 5);

        // Verify ordering: scores should be descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }

        // The top result should be among the true top results
        // (with quantization there may be some reranking)
        let mut true_scores: Vec<(u64, f32)> = raw_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, dot_f32(&query, v)))
            .collect();
        true_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // The quantized top-1 should be in the true top-10
        let top1_id = results[0].id;
        let true_top10_ids: Vec<u64> = true_scores.iter().take(10).map(|s| s.0).collect();
        assert!(
            true_top10_ids.contains(&top1_id),
            "Top-1 result {top1_id} not in true top-10"
        );
    }
}
