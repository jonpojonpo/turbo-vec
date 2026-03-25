/// Pack a sequence of b-bit indices into a byte vector.
///
/// Each index must be in the range `0..2^bit_width`.
pub fn pack_indices(indices: &[u32], bit_width: u8) -> Vec<u8> {
    match bit_width {
        1 => pack_1bit(indices),
        2 => pack_2bit(indices),
        4 => pack_4bit(indices),
        8 => indices.iter().map(|&i| i as u8).collect(),
        _ => pack_general(indices, bit_width),
    }
}

/// Unpack b-bit indices from a packed byte vector.
pub fn unpack_indices(packed: &[u8], bit_width: u8, count: usize) -> Vec<u32> {
    match bit_width {
        1 => unpack_1bit(packed, count),
        2 => unpack_2bit(packed, count),
        4 => unpack_4bit(packed, count),
        8 => packed.iter().map(|&b| b as u32).collect(),
        _ => unpack_general(packed, bit_width, count),
    }
}

/// Pack sign bits: positive → 1, negative/zero → 0, into packed bytes.
pub fn pack_signs(values: &[f32]) -> Vec<u8> {
    let n_bytes = (values.len() + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    for (i, &v) in values.iter().enumerate() {
        if v >= 0.0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

/// Unpack sign bits to +1.0 / -1.0 values.
pub fn unpack_signs(packed: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bit = (packed[i / 8] >> (i % 8)) & 1;
        out.push(if bit == 1 { 1.0 } else { -1.0 });
    }
    out
}

// ---- 1-bit packing (8 indices per byte) ----

fn pack_1bit(indices: &[u32]) -> Vec<u8> {
    let n_bytes = (indices.len() + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    for (i, &idx) in indices.iter().enumerate() {
        if idx != 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

fn unpack_1bit(packed: &[u8], count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(((packed[i / 8] >> (i % 8)) & 1) as u32);
    }
    out
}

// ---- 2-bit packing (4 indices per byte) ----

fn pack_2bit(indices: &[u32]) -> Vec<u8> {
    let n_bytes = (indices.len() + 3) / 4;
    let mut packed = vec![0u8; n_bytes];
    for (i, &idx) in indices.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        packed[byte_idx] |= (idx as u8 & 0x03) << shift;
    }
    packed
}

fn unpack_2bit(packed: &[u8], count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        out.push(((packed[byte_idx] >> shift) & 0x03) as u32);
    }
    out
}

// ---- 4-bit packing (2 indices per byte, nibbles) ----

fn pack_4bit(indices: &[u32]) -> Vec<u8> {
    let n_bytes = (indices.len() + 1) / 2;
    let mut packed = vec![0u8; n_bytes];
    for (i, &idx) in indices.iter().enumerate() {
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] |= idx as u8 & 0x0F;
        } else {
            packed[byte_idx] |= (idx as u8 & 0x0F) << 4;
        }
    }
    packed
}

fn unpack_4bit(packed: &[u8], count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte_idx = i / 2;
        if i % 2 == 0 {
            out.push((packed[byte_idx] & 0x0F) as u32);
        } else {
            out.push(((packed[byte_idx] >> 4) & 0x0F) as u32);
        }
    }
    out
}

// ---- General bit-stream packing ----

fn pack_general(indices: &[u32], bit_width: u8) -> Vec<u8> {
    let total_bits = indices.len() * bit_width as usize;
    let n_bytes = (total_bits + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    let bw = bit_width as usize;
    let mask = (1u32 << bw) - 1;

    for (i, &idx) in indices.iter().enumerate() {
        let bit_offset = i * bw;
        let byte_offset = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let val = idx & mask;

        // May span up to 2 bytes
        packed[byte_offset] |= (val << bit_shift) as u8;
        if bit_shift + bw > 8 && byte_offset + 1 < n_bytes {
            packed[byte_offset + 1] |= (val >> (8 - bit_shift)) as u8;
        }
        if bit_shift + bw > 16 && byte_offset + 2 < n_bytes {
            packed[byte_offset + 2] |= (val >> (16 - bit_shift)) as u8;
        }
    }
    packed
}

fn unpack_general(packed: &[u8], bit_width: u8, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let bw = bit_width as usize;
    let mask = (1u32 << bw) - 1;

    for i in 0..count {
        let bit_offset = i * bw;
        let byte_offset = bit_offset / 8;
        let bit_shift = bit_offset % 8;

        let mut val = (packed[byte_offset] as u32) >> bit_shift;
        if bit_shift + bw > 8 && byte_offset + 1 < packed.len() {
            val |= (packed[byte_offset + 1] as u32) << (8 - bit_shift);
        }
        if bit_shift + bw > 16 && byte_offset + 2 < packed.len() {
            val |= (packed[byte_offset + 2] as u32) << (16 - bit_shift);
        }
        out.push(val & mask);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_1bit() {
        let indices: Vec<u32> = vec![0, 1, 1, 0, 1, 0, 0, 1, 1];
        let packed = pack_indices(&indices, 1);
        let unpacked = unpack_indices(&packed, 1, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_roundtrip_2bit() {
        let indices: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 0, 3, 1];
        let packed = pack_indices(&indices, 2);
        let unpacked = unpack_indices(&packed, 2, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_roundtrip_4bit() {
        let indices: Vec<u32> = (0..16).collect();
        let packed = pack_indices(&indices, 4);
        let unpacked = unpack_indices(&packed, 4, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_roundtrip_8bit() {
        let indices: Vec<u32> = (0..256).collect();
        let packed = pack_indices(&indices, 8);
        let unpacked = unpack_indices(&packed, 8, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_roundtrip_3bit() {
        let indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 5];
        let packed = pack_indices(&indices, 3);
        let unpacked = unpack_indices(&packed, 3, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_roundtrip_5bit() {
        let indices: Vec<u32> = (0..32).collect();
        let packed = pack_indices(&indices, 5);
        let unpacked = unpack_indices(&packed, 5, indices.len());
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_signs_roundtrip() {
        let values = vec![1.0, -2.0, 0.0, 3.5, -0.1, 0.001, -100.0, 42.0, -1.0];
        let packed = pack_signs(&values);
        let unpacked = unpack_signs(&packed, values.len());
        for (v, s) in values.iter().zip(unpacked.iter()) {
            if *v >= 0.0 {
                assert_eq!(*s, 1.0);
            } else {
                assert_eq!(*s, -1.0);
            }
        }
    }
}
