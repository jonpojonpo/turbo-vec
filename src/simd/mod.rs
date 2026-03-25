pub mod portable;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

use crate::quant::codebook::Codebook;

/// Precomputed distance table for fast asymmetric dot products.
///
/// For a query vector q, stores `table[j][k] = q_rot[j] * centroid[k]`
/// so that the dot product with a quantized vector becomes pure table lookups.
pub struct DistanceTable {
    /// Flattened table: `[dim][num_centroids]`.
    pub table: Vec<f32>,
    pub bit_width: u8,
    pub dim: usize,
}

impl DistanceTable {
    /// Build a distance table from a rotated query vector.
    pub fn from_rotated_query(q_rot: &[f32], codebook: &Codebook) -> Self {
        let num_centroids = 1 << codebook.bit_width;
        let dim = q_rot.len();
        let mut table = Vec::with_capacity(dim * num_centroids);
        for j in 0..dim {
            for k in 0..num_centroids {
                table.push(q_rot[j] * codebook.centroids[k]);
            }
        }
        Self {
            table,
            bit_width: codebook.bit_width,
            dim,
        }
    }

    /// Compute dot product with a quantized vector using table lookups.
    #[inline]
    pub fn dot(&self, packed: &[u8], norm: f32) -> f32 {
        let score = portable::dot_table_lookup(&self.table, packed, self.bit_width, self.dim);
        score * norm
    }
}
