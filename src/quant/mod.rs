pub mod codebook;
pub mod mse;
pub mod packed;
pub mod prod;

pub use codebook::Codebook;
pub use mse::TurboQuantMse;
pub use prod::TurboQuantProd;

use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

/// Trait for vector quantizers.
pub trait Quantizer: Send + Sync {
    type Quantized: Clone + Send + Sync + Serialize + DeserializeOwned;

    fn dim(&self) -> usize;
    fn bit_width(&self) -> u8;
    fn quantize(&self, x: &[f32]) -> Self::Quantized;
    fn dequantize(&self, q: &Self::Quantized) -> Vec<f32>;

    fn quantize_batch(&self, vectors: &[&[f32]]) -> Vec<Self::Quantized> {
        vectors.par_iter().map(|v| self.quantize(v)).collect()
    }
}
