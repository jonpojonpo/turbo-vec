use serde::{Deserialize, Serialize};

/// Precomputed Lloyd-Max centroids for N(0,1) distribution.
/// For dimension d, actual centroids = value / sqrt(d).
/// Index by `GAUSSIAN_CENTROIDS[bit_width - 1]`.
const GAUSSIAN_CENTROIDS: [&[f64]; 8] = [
    // b=1: ±sqrt(2/π) ≈ ±0.7978845608
    &[-0.7978845608028654, 0.7978845608028654],
    // b=2: Lloyd-Max for N(0,1) with 4 levels
    &[-1.5104176087114887, -0.4527800398860679, 0.4527800398860679, 1.5104176087114887],
    // b=3: Lloyd-Max for N(0,1) with 8 levels
    &[
        -2.1519775164788833, -1.3439092613750225, -0.7560052489539643, -0.2451209526195130,
        0.2451209526195130, 0.7560052489539643, 1.3439092613750225, 2.1519775164788833,
    ],
    // b=4: Lloyd-Max for N(0,1) with 16 levels
    &[
        -2.7326368750393808, -2.0690764059673088, -1.6180463604498692, -1.2562298709498996,
        -0.9423401767187408, -0.6568029548039882, -0.3880823422946209, -0.1284369060498739,
        0.1284369060498739, 0.3880823422946209, 0.6568029548039882, 0.9423401767187408,
        1.2562298709498996, 1.6180463604498692, 2.0690764059673088, 2.7326368750393808,
    ],
    // b=5: Lloyd-Max for N(0,1) with 32 levels
    &[
        -3.2607497689456020, -2.6955523892900958, -2.3411807724498007, -2.0697600653498503,
        -1.8435792951499088, -1.6472192768499318, -1.4720970437499432, -1.3124582659499506,
        -1.1647578419499559, -1.0264815294499600, -0.8957620099499636, -0.7712192869499667,
        -0.6518011359499693, -0.5367418109499717, -0.4254470919499739, -0.3174414259499759,
        0.3174414259499759, 0.4254470919499739, 0.5367418109499717, 0.6518011359499693,
        0.7712192869499667, 0.8957620099499636, 1.0264815294499600, 1.1647578419499559,
        1.3124582659499506, 1.4720970437499432, 1.6472192768499318, 1.8435792951499088,
        2.0697600653498503, 2.3411807724498007, 2.6955523892900958, 3.2607497689456020,
    ],
    // b=6,7,8: Compute at runtime from Lloyd-Max on N(0,1)
    &[],
    &[],
    &[],
];

/// A codebook for quantizing scalar values at a given bit-width and dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codebook {
    /// Sorted centroids. Length = 2^bit_width.
    pub centroids: Vec<f32>,
    /// Decision boundaries (midpoints between consecutive centroids).
    /// Length = 2^bit_width - 1.
    pub boundaries: Vec<f32>,
    pub bit_width: u8,
    pub dimension: u32,
}

impl Codebook {
    /// Compute a codebook for the given dimension and bit-width.
    ///
    /// For d >= 64, uses the Gaussian approximation (centroids scale as 1/√d).
    /// For d < 64, runs Lloyd-Max on the exact Beta distribution.
    pub fn compute(dim: u32, bit_width: u8) -> Self {
        assert!((1..=8).contains(&bit_width), "bit_width must be 1-8");
        let centroids = if dim >= 64 {
            Self::gaussian_codebook(dim, bit_width)
        } else {
            Self::beta_codebook(dim, bit_width)
        };
        let boundaries = Self::compute_boundaries(&centroids);
        Self {
            centroids,
            boundaries,
            bit_width,
            dimension: dim,
        }
    }

    /// Use precomputed N(0,1) centroids scaled by 1/√d.
    fn gaussian_codebook(dim: u32, bit_width: u8) -> Vec<f32> {
        let idx = (bit_width - 1) as usize;
        let precomputed = GAUSSIAN_CENTROIDS[idx];
        if !precomputed.is_empty() {
            let scale = 1.0 / (dim as f64).sqrt();
            return precomputed.iter().map(|&c| (c * scale) as f32).collect();
        }
        // For b=6,7,8: run Lloyd-Max on N(0,1) then scale
        let gaussian_centroids = lloyd_max_gaussian(bit_width);
        let scale = 1.0 / (dim as f64).sqrt();
        gaussian_centroids
            .iter()
            .map(|&c| (c * scale) as f32)
            .collect()
    }

    /// Run Lloyd-Max on the exact Beta distribution for low dimensions.
    fn beta_codebook(dim: u32, bit_width: u8) -> Vec<f32> {
        let centroids = lloyd_max_beta(dim, bit_width);
        centroids.iter().map(|&c| c as f32).collect()
    }

    fn compute_boundaries(centroids: &[f32]) -> Vec<f32> {
        centroids
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect()
    }

    /// Quantize a scalar to its nearest centroid index.
    #[inline]
    pub fn quantize_scalar(&self, x: f32) -> u32 {
        // For small codebooks (b <= 4, up to 16 centroids), linear scan is
        // competitive with binary search due to branch prediction.
        if self.bit_width <= 4 {
            self.quantize_scalar_linear(x)
        } else {
            self.quantize_scalar_binary(x)
        }
    }

    #[inline]
    fn quantize_scalar_linear(&self, x: f32) -> u32 {
        for (i, &b) in self.boundaries.iter().enumerate() {
            if x < b {
                return i as u32;
            }
        }
        self.boundaries.len() as u32
    }

    #[inline]
    fn quantize_scalar_binary(&self, x: f32) -> u32 {
        match self
            .boundaries
            .binary_search_by(|b| b.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i as u32,
            Err(i) => i as u32,
        }
    }

    /// Dequantize: return centroid value for a given index.
    #[inline]
    pub fn dequantize_scalar(&self, idx: u32) -> f32 {
        self.centroids[idx as usize]
    }
}

// ---------------------------------------------------------------------------
// Lloyd-Max algorithm
// ---------------------------------------------------------------------------

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}


/// Beta distribution PDF on [-1, 1] from Lemma 1.
/// f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
fn beta_pdf(x: f64, dim: u32) -> f64 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let d = dim as f64;
    // Use log-space to avoid Gamma overflow
    let log_coeff = ln_gamma(d / 2.0) - 0.5 * std::f64::consts::PI.ln() - ln_gamma((d - 1.0) / 2.0);
    let log_body = ((d - 3.0) / 2.0) * (1.0 - x * x).ln();
    (log_coeff + log_body).exp()
}

/// Log-gamma function using Stirling's approximation.
fn ln_gamma(x: f64) -> f64 {
    // Use the Lanczos approximation for better accuracy
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        const COEFFS: [f64; 7] = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
        ];
        let g = 5.0_f64;
        let mut sum = COEFFS[0];
        for (i, &c) in COEFFS.iter().enumerate().skip(1) {
            sum += c / (x + i as f64);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
    }
}

/// Numerical integration using adaptive Simpson's rule.
fn integrate<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
    adaptive_simpson(f, a, b, 1e-12, 50)
}

fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    let mid = (a + b) / 2.0;
    let h = b - a;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let s = (h / 6.0) * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(f, a, b, tol, s, fa, fb, fm, max_depth)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    whole: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    depth: u32,
) -> f64 {
    let mid = (a + b) / 2.0;
    let h = b - a;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let f1 = f(m1);
    let f2 = f(m2);
    let left = (h / 12.0) * (fa + 4.0 * f1 + fm);
    let right = (h / 12.0) * (fm + 4.0 * f2 + fb);
    let combined = left + right;
    if depth == 0 || (combined - whole).abs() <= 15.0 * tol {
        combined + (combined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, tol / 2.0, left, fa, fm, f1, depth - 1)
            + adaptive_simpson_rec(f, mid, b, tol / 2.0, right, fm, fb, f2, depth - 1)
    }
}

/// Run Lloyd-Max on N(0,1) distribution to find optimal centroids for given bit-width.
fn lloyd_max_gaussian(bit_width: u8) -> Vec<f64> {
    let n = 1usize << bit_width;
    // Initialize centroids from quantiles of N(0,1)
    let mut centroids: Vec<f64> = (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            quantile_normal(p)
        })
        .collect();

    for _ in 0..200 {
        // Compute boundaries (midpoints)
        let mut boundaries = Vec::with_capacity(n + 1);
        boundaries.push(-8.0); // -∞ approximation
        for w in centroids.windows(2) {
            boundaries.push((w[0] + w[1]) / 2.0);
        }
        boundaries.push(8.0); // +∞ approximation

        // Update centroids: c_i = E[X | X ∈ (b_{i-1}, b_i)]
        let mut new_centroids = Vec::with_capacity(n);
        let mut max_shift = 0.0_f64;
        for i in 0..n {
            let a = boundaries[i];
            let b = boundaries[i + 1];
            let numer = integrate(&|x| x * normal_pdf(x), a, b);
            let denom = integrate(&normal_pdf, a, b);
            let c = if denom.abs() > 1e-15 {
                numer / denom
            } else {
                centroids[i]
            };
            max_shift = max_shift.max((c - centroids[i]).abs());
            new_centroids.push(c);
        }
        centroids = new_centroids;
        if max_shift < 1e-12 {
            break;
        }
    }
    centroids
}

/// Run Lloyd-Max on the Beta distribution for given dimension and bit-width.
fn lloyd_max_beta(dim: u32, bit_width: u8) -> Vec<f64> {
    let n = 1usize << bit_width;
    let pdf = |x: f64| beta_pdf(x, dim);

    // Initialize centroids from quantiles
    let mut centroids: Vec<f64> = (0..n)
        .map(|i| {
            // Approximate: uniform in [-1, 1]
            -1.0 + (2.0 * (i as f64 + 0.5)) / n as f64
        })
        .collect();

    for _ in 0..200 {
        let mut boundaries = Vec::with_capacity(n + 1);
        boundaries.push(-1.0);
        for w in centroids.windows(2) {
            boundaries.push((w[0] + w[1]) / 2.0);
        }
        boundaries.push(1.0);

        let mut new_centroids = Vec::with_capacity(n);
        let mut max_shift = 0.0_f64;
        for i in 0..n {
            let a = boundaries[i];
            let b = boundaries[i + 1];
            let numer = integrate(&|x| x * pdf(x), a, b);
            let denom = integrate(&pdf, a, b);
            let c = if denom.abs() > 1e-15 {
                numer / denom
            } else {
                centroids[i]
            };
            max_shift = max_shift.max((c - centroids[i]).abs());
            new_centroids.push(c);
        }
        centroids = new_centroids;
        if max_shift < 1e-12 {
            break;
        }
    }
    centroids
}

/// Inverse CDF (quantile function) for N(0,1) using rational approximation.
fn quantile_normal(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    if p <= 0.0 {
        return -8.0;
    }
    if p >= 1.0 {
        return 8.0;
    }
    let p = p - 0.5;
    if p.abs() <= 0.425 {
        let r = 0.180625 - p * p;
        p * (((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r
            + 6.7265770927008700853e4)
            * r
            + 4.5921953931549871457e4)
            * r
            + 1.3731693765509461125e4)
            * r
            + 1.9715909503065514427e3)
            * r
            + 1.3314166764078226174e2)
            * r
            + 3.3871328727963666080e0)
            / (((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r
                + 3.9307895800092710610e4)
                * r
                + 2.1213794301586595867e4)
                * r
                + 5.3941960214247511077e3)
                * r
                + 6.8718700749205790830e2)
                * r
                + 4.2313330701600911252e1)
                * r
                + 1.0)
    } else {
        let r = if p < 0.0 { p + 0.5 } else { 0.5 - p };
        let r = (-r.ln()).sqrt();
        let result = if r <= 5.0 {
            let r = r - 1.6;
            (((((((7.7454501427834140764e-4 * r + 2.2723844989269184187e-2) * r
                + 7.2235882510988735035e-1)
                * r
                + 6.5435169056918379040e0)
                * r
                + 1.4895707413373488780e1)
                * r
                + 1.5518327524848219530e1)
                * r
                + 6.1651697744967400299e0)
                * r
                + 7.4559463367482098010e-1)
                / (((((((1.0507418520539770100e-4 * r + 1.0532057969648517637e-2) * r
                    + 2.5323074345003948015e-1)
                    * r
                    + 1.5927120968367805788e0)
                    * r
                    + 4.0779409909137100200e0)
                    * r
                    + 4.6699532449903198490e0)
                    * r
                    + 2.1449999936089607480e0)
                    * r
                    + 1.0)
        } else {
            let r = r - 5.0;
            (((((((2.0103438784972800974e-7 * r + 2.7115555687434552063e-5) * r
                + 1.2426609473880784386e-3)
                * r
                + 2.2736523482687150190e-2)
                * r
                + 1.8166921107749989849e-1)
                * r
                + 6.3918197316498267680e-1)
                * r
                + 8.8277431666079193860e-1)
                * r
                + 3.0838856104922207636e-1)
                / (((((((8.3687901617094846453e-8 * r + 1.2533297294681946182e-5) * r
                    + 6.5674816889992019995e-4)
                    * r
                    + 1.3927671891247491043e-2)
                    * r
                    + 1.3161109814498701100e-1)
                    * r
                    + 5.5907999469756138750e-1)
                    * r
                    + 9.4099459177392024320e-1)
                    * r
                    + 1.0)
        };
        if p < 0.0 { -result } else { result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_b1_gaussian() {
        let cb = Codebook::compute(1536, 1);
        assert_eq!(cb.centroids.len(), 2);
        let expected = 0.7978845608 / (1536.0_f64).sqrt();
        assert!((cb.centroids[0] as f64 + expected).abs() < 1e-5);
        assert!((cb.centroids[1] as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_codebook_b2_gaussian() {
        let cb = Codebook::compute(1536, 2);
        assert_eq!(cb.centroids.len(), 4);
        // Centroids should be symmetric around 0
        assert!((cb.centroids[0] + cb.centroids[3]).abs() < 1e-5);
        assert!((cb.centroids[1] + cb.centroids[2]).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_dequantize_scalar() {
        let cb = Codebook::compute(256, 2);
        // Quantize each centroid should return its own index
        for (i, &c) in cb.centroids.iter().enumerate() {
            assert_eq!(cb.quantize_scalar(c), i as u32);
        }
    }

    #[test]
    fn test_codebook_low_dim_beta() {
        // For low dim, should use exact Beta distribution
        let cb = Codebook::compute(8, 2);
        assert_eq!(cb.centroids.len(), 4);
        // Should be symmetric
        assert!((cb.centroids[0] + cb.centroids[3]).abs() < 1e-4);
    }

    #[test]
    fn test_lloyd_max_gaussian_b1() {
        let centroids = lloyd_max_gaussian(1);
        assert_eq!(centroids.len(), 2);
        // Should be ±sqrt(2/π)
        let expected = (2.0 / std::f64::consts::PI).sqrt();
        assert!((centroids[0] + expected).abs() < 1e-6);
        assert!((centroids[1] - expected).abs() < 1e-6);
    }
}
