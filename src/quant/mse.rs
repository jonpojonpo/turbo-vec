use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use super::codebook::Codebook;
use super::packed;
use crate::simd::portable::l2_norm;

/// Quantized representation under TurboQuantMse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedMse {
    /// Packed b-bit centroid indices (b bits per coordinate, d coordinates).
    pub packed_indices: Vec<u8>,
    /// Original L2 norm of the input vector.
    pub norm: f32,
}

/// TurboQuantMse: MSE-optimal vector quantizer (Algorithm 1 from the paper).
///
/// Applies a random rotation Π to induce a Beta distribution on coordinates,
/// then quantizes each coordinate independently using precomputed Lloyd-Max
/// centroids.
#[derive(Clone, Serialize, Deserialize)]
pub struct TurboQuantMse {
    /// Random rotation matrix Π (d × d orthogonal).
    rotation: DMatrix<f32>,
    /// Transpose Π^T, precomputed for dequantization.
    rotation_t: DMatrix<f32>,
    /// Lloyd-Max codebook for this bit-width and dimension.
    codebook: Codebook,
    dim: usize,
    bit_width: u8,
}

impl TurboQuantMse {
    /// Create a new TurboQuantMse quantizer.
    ///
    /// - `dim`: vector dimensionality
    /// - `bit_width`: bits per coordinate (1–8)
    /// - `seed`: RNG seed for reproducible rotation matrix generation
    pub fn new(dim: usize, bit_width: u8, seed: u64) -> Self {
        assert!((1..=8).contains(&bit_width), "bit_width must be 1-8");
        assert!(dim > 0, "dimension must be positive");

        // Generate random rotation via QR decomposition of Gaussian matrix
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        let gauss = DMatrix::<f32>::from_fn(dim, dim, |_, _| normal.sample(&mut rng));

        let qr = gauss.qr();
        let mut q = qr.q();
        let r = qr.r();

        // Sign correction: multiply each column of Q by sign(R[i,i])
        // to get a Haar-distributed random orthogonal matrix.
        for i in 0..dim {
            if r[(i, i)] < 0.0 {
                for j in 0..dim {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }

        let rotation_t = q.transpose();
        let codebook = Codebook::compute(dim as u32, bit_width);

        Self {
            rotation: q,
            rotation_t,
            codebook,
            dim,
            bit_width,
        }
    }

    /// Access the codebook.
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Access the rotation matrix.
    pub fn rotation(&self) -> &DMatrix<f32> {
        &self.rotation
    }

    /// Quantize a vector.
    pub fn quantize(&self, x: &[f32]) -> QuantizedMse {
        assert_eq!(x.len(), self.dim, "input dimension mismatch");

        let norm = l2_norm(x);
        if norm == 0.0 {
            return QuantizedMse {
                packed_indices: packed::pack_indices(&vec![0; self.dim], self.bit_width),
                norm: 0.0,
            };
        }

        // Normalize to unit sphere
        let x_normed = DVector::from_fn(self.dim, |i, _| x[i] / norm);

        // y = Π · x_normed
        let y = &self.rotation * &x_normed;

        // Quantize each coordinate
        let mut indices = Vec::with_capacity(self.dim);
        for j in 0..self.dim {
            indices.push(self.codebook.quantize_scalar(y[j]));
        }

        QuantizedMse {
            packed_indices: packed::pack_indices(&indices, self.bit_width),
            norm,
        }
    }

    /// Dequantize back to a float vector.
    pub fn dequantize(&self, q: &QuantizedMse) -> Vec<f32> {
        let indices = packed::unpack_indices(&q.packed_indices, self.bit_width, self.dim);

        // Reconstruct rotated vector from centroids
        let mut y_tilde = DVector::zeros(self.dim);
        for j in 0..self.dim {
            y_tilde[j] = self.codebook.dequantize_scalar(indices[j]);
        }

        // Rotate back: x̃ = Π^T · ỹ, then scale by norm
        let x_tilde = &self.rotation_t * &y_tilde;
        x_tilde.iter().map(|&v| v * q.norm).collect()
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

    #[test]
    fn test_quantize_dequantize_identity_approx() {
        let dim = 128;
        let bit_width = 4;
        let quantizer = TurboQuantMse::new(dim, bit_width, 42);

        // Create a random unit vector
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let normal = StandardNormal;
        let x: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        let norm = l2_norm(&x);
        let x_unit: Vec<f32> = x.iter().map(|v| v / norm).collect();

        let quantized = quantizer.quantize(&x_unit);
        let reconstructed = quantizer.dequantize(&quantized);

        // MSE should be bounded: D_mse ≤ sqrt(3π)/2 * 1/4^b
        let mse: f32 = x_unit
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let theoretical_bound = (3.0_f32 * std::f32::consts::PI).sqrt() / 2.0 / 4.0_f32.powi(bit_width as i32);
        // Allow some slack for finite dimension
        assert!(
            mse < theoretical_bound * 2.0,
            "MSE {mse} exceeds relaxed bound {}",
            theoretical_bound * 2.0
        );
    }

    #[test]
    fn test_mse_distortion_empirical() {
        let dim = 256;
        let bit_width = 2;
        let quantizer = TurboQuantMse::new(dim, bit_width, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let normal = StandardNormal;

        let n_trials = 1000;
        let mut total_mse = 0.0_f64;

        for _ in 0..n_trials {
            let x: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let norm = l2_norm(&x);
            let x_unit: Vec<f32> = x.iter().map(|v| v / norm).collect();

            let q = quantizer.quantize(&x_unit);
            let recon = quantizer.dequantize(&q);

            let mse: f64 = x_unit
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            total_mse += mse;
        }

        let avg_mse = total_mse / n_trials as f64;
        // Paper says b=2 MSE ≈ 0.117
        assert!(
            avg_mse < 0.20,
            "Average MSE {avg_mse} too high (expected ~0.117)"
        );
    }

    #[test]
    fn test_zero_vector() {
        let quantizer = TurboQuantMse::new(64, 2, 42);
        let x = vec![0.0; 64];
        let q = quantizer.quantize(&x);
        let recon = quantizer.dequantize(&q);
        assert!(recon.iter().all(|&v| v == 0.0));
    }
}
