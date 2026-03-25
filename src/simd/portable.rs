use crate::quant::packed;

/// Scalar dot product using distance table lookups.
///
/// `table` is flattened `[dim][num_centroids]`, `packed` contains the quantized indices.
pub fn dot_table_lookup(table: &[f32], packed_indices: &[u8], bit_width: u8, dim: usize) -> f32 {
    let indices = packed::unpack_indices(packed_indices, bit_width, dim);
    let num_centroids = 1usize << bit_width;
    let mut sum = 0.0_f32;
    for (j, &idx) in indices.iter().enumerate() {
        sum += table[j * num_centroids + idx as usize];
    }
    sum
}

/// Dot product between two float slices.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a float slice.
#[inline]
pub fn l2_norm(x: &[f32]) -> f32 {
    dot_f32(x, x).sqrt()
}
