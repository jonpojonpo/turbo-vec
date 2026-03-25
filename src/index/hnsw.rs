use std::collections::{BinaryHeap, HashSet};

use rand::Rng;
use rand::SeedableRng;

use crate::quant::mse::{QuantizedMse, TurboQuantMse};
use crate::simd::DistanceTable;
use crate::types::SearchResult;

/// HNSW (Hierarchical Navigable Small World) index with TurboQuant compression.
pub struct HnswIndex {
    quantizer: TurboQuantMse,
    /// Graph layers. `layers[0]` is the bottom (densest) layer.
    layers: Vec<GraphLayer>,
    /// Quantized vectors.
    vectors: Vec<QuantizedMse>,
    /// External IDs.
    ids: Vec<u64>,
    /// Max connections per node per layer (except layer 0).
    m: usize,
    /// Max connections at layer 0.
    m_max0: usize,
    /// Size of the dynamic candidate list during construction.
    ef_construction: usize,
    /// Entry point node index (in the topmost layer).
    entry_point: Option<usize>,
    /// Level of the entry point.
    max_level: usize,
    /// Normalization factor for level generation: 1/ln(m).
    ml: f64,
    /// RNG seed for level generation.
    rng: rand::rngs::StdRng,
}

struct GraphLayer {
    /// Adjacency list: `neighbors[node_idx]` = vec of neighbor indices.
    neighbors: Vec<Vec<u32>>,
}

impl GraphLayer {
    fn new() -> Self {
        Self {
            neighbors: Vec::new(),
        }
    }

    fn add_node(&mut self) {
        self.neighbors.push(Vec::new());
    }

    fn connect(&mut self, a: usize, b: usize, max_connections: usize) {
        if a == b {
            return;
        }
        let neighbors = &mut self.neighbors[a];
        if !neighbors.contains(&(b as u32)) {
            neighbors.push(b as u32);
            if neighbors.len() > max_connections {
                // Simple pruning: keep only `max_connections` nearest
                // In a full implementation, use heuristic pruning
                neighbors.truncate(max_connections);
            }
        }
    }
}

/// Entry in the search beam (min-heap by negated score for top-k).
#[derive(Clone)]
struct Candidate {
    node: usize,
    score: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap on score (we want highest inner product first)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Candidate for the "worst" tracking (min-heap).
#[derive(Clone)]
struct MinCandidate {
    node: usize,
    score: f32,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse ordering
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl HnswIndex {
    /// Create a new HNSW index.
    ///
    /// - `m`: max connections per layer (typical: 16)
    /// - `ef_construction`: beam width during construction (typical: 200)
    pub fn new(quantizer: TurboQuantMse, m: usize, ef_construction: usize) -> Self {
        Self {
            quantizer,
            layers: Vec::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            m,
            m_max0: m * 2,
            ef_construction,
            entry_point: None,
            max_level: 0,
            ml: 1.0 / (m as f64).ln(),
            rng: rand::rngs::StdRng::seed_from_u64(12345),
        }
    }

    /// Generate a random level for a new node.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.random();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Compute the inner product score between a distance table and a stored vector.
    fn score(&self, dist_table: &DistanceTable, node: usize) -> f32 {
        let qvec = &self.vectors[node];
        dist_table.dot(&qvec.packed_indices, qvec.norm)
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: u64, vector: &[f32]) {
        let quantized = self.quantizer.quantize(vector);
        let node_idx = self.vectors.len();
        self.vectors.push(quantized);
        self.ids.push(id);

        // Build distance table for the new vector
        let q_vec = nalgebra::DVector::from_column_slice(vector);
        let q_rot = self.quantizer.rotation() * &q_vec;
        let q_rot_slice: Vec<f32> = q_rot.iter().copied().collect();
        let dist_table =
            DistanceTable::from_rotated_query(&q_rot_slice, self.quantizer.codebook());

        let level = self.random_level();

        // Ensure enough layers exist
        while self.layers.len() <= level {
            self.layers.push(GraphLayer::new());
        }

        // Add node to all layers up to its level
        // First, ensure all layers have enough nodes
        for layer in &mut self.layers {
            while layer.neighbors.len() <= node_idx {
                layer.add_node();
            }
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_level = level;
            return;
        }

        let entry = self.entry_point.unwrap();

        // Phase 1: Greedy descent from top layer to `level + 1`
        let mut current = entry;
        for l in (level + 1..=self.max_level).rev() {
            if l >= self.layers.len() {
                continue;
            }
            current = self.greedy_closest(&dist_table, current, l);
        }

        // Phase 2: Insert into layers [min(level, max_level)..=0]
        let insert_top = level.min(self.max_level);
        let mut ep = vec![current];

        for l in (0..=insert_top).rev() {
            let max_conn = if l == 0 { self.m_max0 } else { self.m };
            let neighbors =
                self.search_layer(&dist_table, &ep, self.ef_construction, l);

            // Connect to the nearest `m` neighbors
            let take = neighbors.len().min(self.m);
            for i in 0..take {
                let neighbor = neighbors[i].node;
                self.layers[l].connect(node_idx, neighbor, max_conn);
                self.layers[l].connect(neighbor, node_idx, max_conn);
            }

            // Use the neighbors as entry points for the next layer down
            ep = neighbors.iter().map(|c| c.node).collect();
        }

        // Update entry point if the new node has a higher level
        if level > self.max_level {
            self.entry_point = Some(node_idx);
            self.max_level = level;
        }
    }

    /// Greedy search to find the single closest node in a layer.
    fn greedy_closest(&self, dist_table: &DistanceTable, entry: usize, layer: usize) -> usize {
        let mut current = entry;
        let mut best_score = self.score(dist_table, current);

        loop {
            let mut improved = false;
            let neighbors = &self.layers[layer].neighbors[current];
            for &n in neighbors {
                let s = self.score(dist_table, n as usize);
                if s > best_score {
                    best_score = s;
                    current = n as usize;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    /// Beam search in a layer, returning up to `ef` candidates sorted by score (descending).
    fn search_layer(
        &self,
        dist_table: &DistanceTable,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // max-heap
        let mut results = BinaryHeap::new(); // min-heap (worst at top)

        for &ep in entry_points {
            if visited.insert(ep) {
                let s = self.score(dist_table, ep);
                candidates.push(Candidate {
                    node: ep,
                    score: s,
                });
                results.push(MinCandidate {
                    node: ep,
                    score: s,
                });
            }
        }

        while let Some(current) = candidates.pop() {
            // If current candidate is worse than the worst in results, stop
            if let Some(worst) = results.peek() {
                if current.score < worst.score && results.len() >= ef {
                    break;
                }
            }

            if layer < self.layers.len() {
                let neighbors = &self.layers[layer].neighbors[current.node];
                for &n in neighbors {
                    let n = n as usize;
                    if visited.insert(n) {
                        let s = self.score(dist_table, n);
                        let should_add = results.len() < ef
                            || results.peek().is_some_and(|w| s > w.score);

                        if should_add {
                            candidates.push(Candidate { node: n, score: s });
                            results.push(MinCandidate { node: n, score: s });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted Vec<Candidate>
        let mut out: Vec<Candidate> = results
            .into_vec()
            .into_iter()
            .map(|mc| Candidate {
                node: mc.node,
                score: mc.score,
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Search for top-k nearest neighbors.
    ///
    /// - `ef`: beam width during search (higher = more accurate but slower; must be >= top_k)
    pub fn search(&self, query: &[f32], top_k: usize, ef: usize) -> Vec<SearchResult> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let ef = ef.max(top_k);

        // Build distance table
        let q_vec = nalgebra::DVector::from_column_slice(query);
        let q_rot = self.quantizer.rotation() * &q_vec;
        let q_rot_slice: Vec<f32> = q_rot.iter().copied().collect();
        let dist_table =
            DistanceTable::from_rotated_query(&q_rot_slice, self.quantizer.codebook());

        let entry = self.entry_point.unwrap();

        // Phase 1: Greedy descent from top layer to layer 1
        let mut current = entry;
        for l in (1..=self.max_level).rev() {
            if l >= self.layers.len() {
                continue;
            }
            current = self.greedy_closest(&dist_table, current, l);
        }

        // Phase 2: Beam search at layer 0
        let candidates = self.search_layer(&dist_table, &[current], ef, 0);

        // Return top-k
        candidates
            .into_iter()
            .take(top_k)
            .map(|c| SearchResult {
                id: self.ids[c.node],
                score: c.score,
            })
            .collect()
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
    fn test_hnsw_basic_search() {
        let dim = 64;
        let quantizer = TurboQuantMse::new(dim, 4, 42);
        let mut index = HnswIndex::new(quantizer, 16, 100);

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = StandardNormal;

        let n = 200;
        let mut raw_vecs = Vec::new();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            raw_vecs.push(v.clone());
            index.add(i as u64, &v);
        }

        let query: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let results = index.search(&query, 10, 50);

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Check scores are descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }

        // Compute ground truth
        let mut true_scores: Vec<(u64, f32)> = raw_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, dot_f32(&query, v)))
            .collect();
        true_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Check recall: at least some of the true top-10 should appear in results
        let true_top10: HashSet<u64> = true_scores.iter().take(10).map(|s| s.0).collect();
        let result_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();
        let recall = true_top10.intersection(&result_ids).count();
        assert!(
            recall >= 3,
            "Recall too low: only {recall}/10 true top-10 found"
        );
    }
}
