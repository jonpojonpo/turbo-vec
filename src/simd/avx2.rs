//! AVX2-accelerated dot product for 4-bit quantized vectors.
//!
//! Uses `vpshufb` (_mm256_shuffle_epi8) as a 16-entry lookup table
//! to compute partial dot products on nibble-packed indices.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated distance table lookup for 4-bit packed indices.
///
/// # Safety
/// Caller must ensure AVX2 is available (`is_x86_feature_detected!("avx2")`).
/// `table` must have layout `[dim][16]` and `packed` must contain `dim/2` bytes
/// of nibble-packed 4-bit indices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_4bit_avx2(table: &[f32], packed: &[u8], dim: usize) -> f32 {
    unsafe {
        let num_centroids = 16usize;
        let mut sum = _mm256_setzero_ps();

        // Process 8 coordinates at a time (4 packed bytes → 8 nibbles)
        let chunks = dim / 8;
        let remainder = dim % 8;

        for chunk in 0..chunks {
            let byte_offset = chunk * 4;
            let b0 = packed[byte_offset] as i32;
            let b1 = packed[byte_offset + 1] as i32;
            let b2 = packed[byte_offset + 2] as i32;
            let b3 = packed[byte_offset + 3] as i32;

            let indices = _mm256_set_epi32(
                (b3 >> 4) & 0x0F,
                b3 & 0x0F,
                (b2 >> 4) & 0x0F,
                b2 & 0x0F,
                (b1 >> 4) & 0x0F,
                b1 & 0x0F,
                (b0 >> 4) & 0x0F,
                b0 & 0x0F,
            );

            let coord_base = chunk * 8;
            let table_offsets = _mm256_set_epi32(
                ((coord_base + 7) * num_centroids) as i32,
                ((coord_base + 6) * num_centroids) as i32,
                ((coord_base + 5) * num_centroids) as i32,
                ((coord_base + 4) * num_centroids) as i32,
                ((coord_base + 3) * num_centroids) as i32,
                ((coord_base + 2) * num_centroids) as i32,
                ((coord_base + 1) * num_centroids) as i32,
                (coord_base * num_centroids) as i32,
            );

            let gather_indices = _mm256_add_epi32(table_offsets, indices);
            let values = _mm256_i32gather_ps::<4>(table.as_ptr(), gather_indices);
            sum = _mm256_add_ps(sum, values);
        }

        let mut result = hsum_avx2(sum);

        // Handle remainder
        let start = chunks * 8;
        for i in start..start + remainder {
            let byte_idx = i / 2;
            let idx = if i % 2 == 0 {
                (packed[byte_idx] & 0x0F) as usize
            } else {
                ((packed[byte_idx] >> 4) & 0x0F) as usize
            };
            result += table[i * num_centroids + idx];
        }

        result
    }
}

/// Horizontal sum of 8 f32 lanes in a __m256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}
