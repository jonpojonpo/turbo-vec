# turbo-vec

A Rust library for near-optimal vector quantization and similarity search, based on the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025).

## Key Features

- **Near-optimal distortion** — within ~2.7× of the information-theoretic lower bound
- **Virtually zero indexing time** — data-oblivious quantization, no k-means training on your data
- **Unbiased inner products** — `TurboQuantProd` guarantees E[⟨y, x̃⟩] = ⟨y, x⟩
- **1–8 bit quantization** — compress 32-bit floats down to 1–8 bits per dimension
- **AVX2-accelerated search** — SIMD dot products via gather instructions on 4-bit packed vectors
- **Three index types** — Flat (brute-force), HNSW, and IVF

## Quick Start

```toml
[dependencies]
turbo-vec = "0.1"
```

```rust
use turbo_vec::{TurboQuantMse, FlatIndex};

// Create a 4-bit quantizer for 128-dimensional vectors
let quantizer = TurboQuantMse::new(128, 4, /*seed=*/ 42);

// Build a flat index
let mut index = FlatIndex::new(quantizer);
for (id, vec) in vectors.iter().enumerate() {
    index.add(id as u64, vec);
}

// Search for the 10 nearest neighbors
let results = index.search(&query, 10);
for r in &results {
    println!("id={}, score={:.4}", r.id, r.score);
}
```

## How It Works

TurboQuant is a two-stage quantization scheme:

### Stage 1: TurboQuantMse (MSE-optimal)

1. Multiply the input vector by a random orthogonal matrix Π (generated once via QR decomposition)
2. Each coordinate of the rotated vector follows a Beta distribution, which in high dimensions converges to Gaussian
3. Quantize each coordinate independently using precomputed Lloyd-Max centroids
4. Dequantize by looking up centroids and rotating back with Π^T

This achieves MSE distortion ≤ √(3π)/2 · 1/4^b for bit-width b.

### Stage 2: TurboQuantProd (unbiased inner product)

1. Apply TurboQuantMse at (b-1) bits
2. Compute the residual r = x - dequantize(quantize(x))
3. Apply a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual: store sign(S·r)
4. Store the residual norm ‖r‖₂ separately

This yields an **unbiased** inner product estimator — critical for nearest neighbor search and attention score computation.

## Index Types

| Index | Use Case | Build Time | Search |
|-------|----------|------------|--------|
| `FlatIndex` | Small datasets, exact baseline | O(n) quantize | O(n) scan |
| `HnswIndex` | General purpose ANN | O(n log n) | O(log n) |
| `IvfIndex` | Large datasets with clustering | O(n) + k-means | O(n/k × n_probe) |

```rust
use turbo_vec::{TurboQuantMse, HnswIndex};

let quantizer = TurboQuantMse::new(768, 4, 42);
let mut index = HnswIndex::new(quantizer, /*m=*/ 16, /*ef_construction=*/ 200);

// Add vectors
for (id, vec) in vectors.iter().enumerate() {
    index.add(id as u64, vec);
}

// Search with ef=50
let results = index.search(&query, /*top_k=*/ 10, /*ef=*/ 50);
```

## Quantizer Comparison

| Quantizer | Bits | MSE (d=1536) | Unbiased IP? | Use Case |
|-----------|------|-------------|--------------|----------|
| `TurboQuantMse` | b | ~0.36, 0.117, 0.03, 0.009 (b=1,2,3,4) | No (biased at low b) | Storage compression, approximate search |
| `TurboQuantProd` | b | Slightly higher | Yes | Nearest neighbor search, KV cache, attention |

```rust
use turbo_vec::TurboQuantProd;

// Unbiased inner product quantizer at 3 bits total
// (2-bit MSE on the vector + 1-bit QJL on the residual)
let quantizer = TurboQuantProd::new(1536, 3, 42);
let compressed = quantizer.quantize(&vector);
let reconstructed = quantizer.dequantize(&compressed);
```

## Performance

- **Indexing**: ~0.001s for 100k vectors (the paper's killer feature — no training step)
- **Compression**: 4-bit quantization = 8× memory reduction vs f32
- **Search**: AVX2-accelerated asymmetric distance computation using precomputed lookup tables

## Architecture

```
src/
├── quant/
│   ├── codebook.rs   — Lloyd-Max solver + precomputed centroids (1–8 bit)
│   ├── mse.rs        — TurboQuantMse (Algorithm 1)
│   ├── prod.rs       — TurboQuantProd (Algorithm 2, MSE + QJL)
│   └── packed.rs     — Bit-packing with fast paths for b=1,2,4,8
├── index/
│   ├── flat.rs       — Brute-force search
│   ├── hnsw.rs       — HNSW graph index
│   └── ivf.rs        — Inverted file index with k-means
└── simd/
    ├── avx2.rs       — AVX2 gather-based 4-bit dot product
    └── portable.rs   — Scalar fallback
```

## References

- Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT
