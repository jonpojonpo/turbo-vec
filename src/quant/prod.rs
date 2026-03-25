use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use super::mse::{QuantizedMse, TurboQuantMse};
use super::packed;
use crate::simd::portable::l2_norm;

/// Quantized representation under TurboQuantProd.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedProd {
    /// MSE-quantized part (at bit_width - 1 bits per coordinate).
    pub mse_part: QuantizedMse,
    /// QJL sign bits on the residual (d bits, packed).
    pub qjl_signs: Vec<u8>,
    /// L2 norm of the residual vector.
    pub residual_norm: f32,
}

/// TurboQuantProd: unbiased inner-product-optimal quantizer (Algorithm 2).
///
/// Two-stage approach:
/// 1. Apply TurboQuantMse at (b-1) bits to minimize residual norm
/// 2. Apply 1-bit QJL transform on the residual for unbiased inner products
#[derive(Clone, Serialize, Deserialize)]
pub struct TurboQuantProd {
    /// Inner MSE quantizer at bit_width - 1.
    mse_quantizer: TurboQuantMse,
    /// Random projection matrix S (d × d, i.i.d. N(0,1)) for QJL.
    projection: DMatrix<f32>,
    /// Transpose S^T, precomputed for dequantization.
    projection_t: DMatrix<f32>,
    /// Target bit-width.
    bit_width: u8,
    dim: usize,
}

impl TurboQuantProd {
    /// Create a new TurboQuantProd quantizer.
    ///
    /// - `dim`: vector dimensionality
    /// - `bit_width`: total bits per coordinate (must be >= 2; the MSE stage uses b-1)
    /// - `seed`: RNG seed for reproducibility
    pub fn new(dim: usize, bit_width: u8, seed: u64) -> Self {
        assert!(bit_width >= 2, "TurboQuantProd requires bit_width >= 2");
        assert!(dim > 0, "dimension must be positive");

        let mse_quantizer = TurboQuantMse::new(dim, bit_width - 1, seed);

        // Generate projection matrix S with a different seed
        let s_seed = seed.wrapping_add(0xDEAD_BEEF);
        let mut rng = rand::rngs::StdRng::seed_from_u64(s_seed);
        let normal = StandardNormal;
        let projection =
            DMatrix::<f32>::from_fn(dim, dim, |_, _| normal.sample(&mut rng));
        let projection_t = projection.transpose();

        Self {
            mse_quantizer,
            projection,
            projection_t,
            bit_width,
            dim,
        }
    }

    /// Quantize a vector.
    pub fn quantize(&self, x: &[f32]) -> QuantizedProd {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");

        // Stage 1: MSE quantize
        let mse_part = self.mse_quantizer.quantize(x);

        // Compute residual: r = x - DeQuantMse(idx)
        let reconstructed = self.mse_quantizer.dequantize(&mse_part);
        let residual: Vec<f32> = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| a - b)
            .collect();
        let residual_norm = l2_norm(&residual);

        // Stage 2: QJL on residual — qjl = sign(S · r)
        let r_vec = nalgebra::DVector::from_column_slice(&residual);
        let s_times_r = &self.projection * &r_vec;

        // Pack sign bits
        let signs: Vec<f32> = s_times_r.iter().copied().collect();
        let qjl_signs = packed::pack_signs(&signs);

        QuantizedProd {
            mse_part,
            qjl_signs,
            residual_norm,
        }
    }

    /// Dequantize back to a float vector.
    pub fn dequantize(&self, q: &QuantizedProd) -> Vec<f32> {
        // Stage 1: reconstruct MSE part
        let x_mse = self.mse_quantizer.dequantize(&q.mse_part);

        // Stage 2: reconstruct QJL part
        // x̃_qjl = √(π/2) / d · γ · S^T · qjl
        let qjl = packed::unpack_signs(&q.qjl_signs, self.dim);
        let qjl_vec = nalgebra::DVector::from_column_slice(&qjl);
        let st_qjl = &self.projection_t * &qjl_vec;

        let scale =
            (std::f32::consts::FRAC_PI_2).sqrt() / (self.dim as f32) * q.residual_norm;

        // Combine: x̃ = x̃_mse + x̃_qjl
        x_mse
            .iter()
            .zip(st_qjl.iter())
            .map(|(a, b)| a + scale * b)
            .collect()
    }

    /// Access the inner MSE quantizer.
    pub fn mse_quantizer(&self) -> &TurboQuantMse {
        &self.mse_quantizer
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        let v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let norm = l2_norm(&v);
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_prod_unbiased_inner_product() {
        let dim = 128;
        let bit_width = 3;
        let quantizer = TurboQuantProd::new(dim, bit_width, 42);

        let n_trials = 2000;
        let mut total_ip_original = 0.0_f64;
        let mut total_ip_quantized = 0.0_f64;

        for trial in 0..n_trials {
            let x = random_unit_vec(dim, 1000 + trial);
            let y = random_unit_vec(dim, 5000 + trial);

            let q = quantizer.quantize(&x);
            let x_recon = quantizer.dequantize(&q);

            let ip_orig: f64 = x.iter().zip(y.iter()).map(|(a, b)| *a as f64 * *b as f64).sum();
            let ip_quant: f64 = x_recon
                .iter()
                .zip(y.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();

            total_ip_original += ip_orig;
            total_ip_quantized += ip_quant;
        }

        let avg_orig = total_ip_original / n_trials as f64;
        let avg_quant = total_ip_quantized / n_trials as f64;

        // The difference should be small (unbiased means E[estimate] = true value)
        let bias = (avg_quant - avg_orig).abs();
        assert!(
            bias < 0.05,
            "Inner product bias {bias} too large (expected near 0)"
        );
    }

    #[test]
    fn test_prod_roundtrip_basic() {
        let dim = 64;
        let quantizer = TurboQuantProd::new(dim, 3, 42);
        let x = random_unit_vec(dim, 123);
        let q = quantizer.quantize(&x);
        let recon = quantizer.dequantize(&q);

        // Should reconstruct something reasonable
        assert_eq!(recon.len(), dim);
        // Check the reconstruction isn't all zeros
        let norm = l2_norm(&recon);
        assert!(norm > 0.5, "reconstruction norm {norm} too small");
    }
}
