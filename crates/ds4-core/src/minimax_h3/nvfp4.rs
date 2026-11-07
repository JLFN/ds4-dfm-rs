//! The reference's NVFP4-AWQ Qwen3-VL encoder as a weight source.
//!
//! Sources read first-hand, and the only authority for this file:
//!
//! - ComfyUI v0.30.0, `b1693ecba9f5b65f8c80ab36b195ab963ec92413`, the revision
//!   `configs/dependencies.json:40-47` pins as `comfy_root`: `comfy/ops.py:
//!   1178-1183` maps the artifact's `weight_scale_2` to the per-tensor scale and
//!   `weight_scale` to the block scale; `comfy/ops.py:1350-1352` multiplies the
//!   *activation* by `pre_quant_scale`; `comfy/quant_ops.py:170-224` is the
//!   layout; `comfy/text_encoders/minimax.py:1-19` states the presentation is
//!   the unnormalized hidden state after layer 50.
//! - comfy-kitchen v0.2.26, `255a43879fe57bbcbecfdb273b46d772b00c5a90`, the
//!   version `requirements.txt:25` pins: `comfy_kitchen/backends/eager/
//!   quantization.py:118-222` is the quantize/dequantize pair, and
//!   `comfy_kitchen/float_utils.py` carries `pack_uint4` (high nibble first),
//!   `to_blocked` and `from_blocked`.
//!
//! What the artifact is: one file, 2054 tensors, layers 0..49 -- the same 50
//! this engine executes (`text.rs:26-39`) -- and no final norm and no `lm_head`
//! because the prompt path reads neither. Every projection is a
//! `{"format": "nvfp4", "full_precision_matrix_mult": true}` weight with a
//! per-tensor F32 scale, a per-16 E4M3 block scale, and, on the two
//! projections no layernorm feeds, an activation `pre_quant_scale`. The
//! embedding is `int8_tensorwise` with one F32 scale per row. The vision tower
//! is BF16 and is not read here.
//!
//! Two conventions are easy to get wrong, and both were measured against the
//! BF16 source before this was written. The codes are packed high nibble first
//! (`pack_uint4`). The block scales are stored in cuBLAS's 128x4 tiled blocked
//! layout, which preserves the shape and permutes the contents: `quantize`
//! ends in `to_blocked` and `dequantize` starts with `from_blocked`. Reading
//! them plainly leaves 0.40-0.56 relative residual against the BF16 weights;
//! unswizzling leaves 0.10, which is E2M1's own block-16 error (`nvfp4-verify.txt`
//! on the Spark, 2026-10-28).
//!
//! Deviation, named and bounded: the reference multiplies the BF16 activation
//! by `pre_quant_scale`, and its eager dequantizer rounds in BF16. This source
//! folds `pre_quant_scale` into the weight columns and accumulates in F32,
//! rounding once to BF16, because the forward it feeds is the existing BF16
//! one. The gate is the eleven-token embedding and the rendered clip.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::safetensors::{self, Dtype, Shard, Tensor};
use super::text::{
    Part, Role, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, QUERY_HEADS, VOCAB,
};

/// E2M1's sixteen codes (`E2M1_LUT`,
/// `comfy_kitchen/backends/eager/quantization.py:166-169`).
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The width one block scale covers: `block_size = 16` (`:136`, `:196`).
const BLOCK: usize = 16;

/// Rows per cuBLAS tile (`to_blocked`, `float_utils.py`).
const TILE_ROWS: usize = 128;

/// Scale columns per tile, the `4` of every tile (`to_blocked`).
const TILE_COLS: usize = 4;

/// Rows per swizzle group: the `32` of `SWIZZLE_32_4_4`
/// (`backends/eager/quantization.py:245`).
const GROUP_ROWS: usize = 32;

/// Elements per tile.
const TILE: usize = TILE_ROWS * TILE_COLS;

/// Stride between one tile's four groups (`blocks.reshape(-1, 4, 32, 4)`).
const TILE_STRIDE: usize = (TILE_ROWS / GROUP_ROWS) * TILE_COLS;

/// Query width, `heads * head_dim`.
const QUERY_DIM: u64 = QUERY_HEADS * HEAD_DIM;

/// Key and value width, `kv_heads * head_dim`.
const KV_DIM: u64 = KV_HEADS * HEAD_DIM;

/// The embedding, the one `int8_tensorwise` tensor (`comfy/ops.py:1178-1183`).
const EMBEDDING: &str = "model.embed_tokens.weight";

/// Why the artifact could not be read.
#[derive(Debug)]
pub enum Nvfp4Error {
    /// The file is not a readable safetensors store.
    Reader(String),
    /// A tensor the encoder needs is absent.
    Absent(String),
    /// A tensor's dtype, shape or marker is not this artifact's.
    Format { name: String, detail: String },
    /// A read failed.
    Io { name: String, detail: String },
}

impl Nvfp4Error {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            Nvfp4Error::Reader(detail) => format!("nvfp4 reader: {detail}"),
            Nvfp4Error::Absent(name) => format!("nvfp4: {name} is absent"),
            Nvfp4Error::Format { name, detail } => format!("nvfp4 {name}: {detail}"),
            Nvfp4Error::Io { name, detail } => format!("nvfp4 {name}: {detail}"),
        }
    }
}

impl std::fmt::Display for Nvfp4Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for Nvfp4Error {}

/// Which dequantization a role needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A plain BF16 tensor: the four norms, copied through.
    Bf16,
    /// `int8_tensorwise`: one F32 scale per row.
    Int8Rows,
    /// A packed NVFP4 weight with a per-tensor and a per-block scale.
    Quantized,
}

/// One role's tensors: the payload and its companions.
#[derive(Clone, Copy, Debug)]
struct Site {
    kind: Kind,
    weight: usize,
    scale: Option<usize>,
    per_tensor: Option<usize>,
    pre_quant: Option<usize>,
}

/// The reference's quantized prompt encoder.
pub struct Nvfp4 {
    path: PathBuf,
    shard: Shard,
    sites: BTreeMap<Role, Site>,
}

impl Nvfp4 {
    /// Open the artifact and resolve every role this encoder reads.
    pub fn open(path: &Path) -> Result<Nvfp4, Nvfp4Error> {
        let shard =
            safetensors::read_shard(path).map_err(|error| Nvfp4Error::Reader(error.token()))?;
        let index: BTreeMap<&str, usize> = shard
            .tensors
            .iter()
            .enumerate()
            .map(|(at, tensor)| (tensor.name.as_str(), at))
            .collect();

        let mut sites = BTreeMap::new();
        sites.insert(Role::Embedding, embedding_site(&shard, &index)?);

        for layer in 0..LAYERS {
            for part in ALL_PARTS {
                let site = layer_site(&shard, &index, layer, part)?;
                sites.insert(Role::Layer { layer, part }, site);
            }
        }

        Ok(Nvfp4 {
            path: path.to_path_buf(),
            shard,
            sites,
        })
    }

    /// The file this source reads.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Decode one role into BF16 bytes, the form the forward uploads.
    pub fn read_into(&self, role: Role, buffer: &mut Vec<u8>) -> Result<(), Nvfp4Error> {
        let site = *self
            .sites
            .get(&role)
            .ok_or_else(|| Nvfp4Error::Absent(format!("{role:?}")))?;

        match site.kind {
            Kind::Bf16 => self.copy_bf16(self.tensor(site.weight), buffer),
            Kind::Int8Rows => self.decode_embedding(site, buffer),
            Kind::Quantized => self.decode_weight(site, buffer),
        }
    }

    fn tensor(&self, at: usize) -> &Tensor {
        &self.shard.tensors[at]
    }

    fn read(&self, at: usize) -> Result<Vec<u8>, Nvfp4Error> {
        let tensor = self.tensor(at);
        let mut buffer = vec![0u8; tensor.bytes() as usize];
        self.shard
            .read_data(tensor, &mut buffer)
            .map_err(|error| Nvfp4Error::Io {
                name: tensor.name.clone(),
                detail: error.token(),
            })?;

        Ok(buffer)
    }

    fn copy_bf16(&self, tensor: &Tensor, buffer: &mut Vec<u8>) -> Result<(), Nvfp4Error> {
        buffer.clear();
        buffer.resize(tensor.bytes() as usize, 0);
        self.shard
            .read_data(tensor, buffer)
            .map_err(|error| Nvfp4Error::Io {
                name: tensor.name.clone(),
                detail: error.token(),
            })
    }

    /// `int8_tensorwise` (`comfy/ops.py:1560-1563`): one F32 scale per row.
    fn decode_embedding(&self, site: Site, buffer: &mut Vec<u8>) -> Result<(), Nvfp4Error> {
        let weight = self.tensor(site.weight);
        let rows = weight.shape[0] as usize;
        let columns = weight.shape[1] as usize;
        let codes = self.read(site.weight)?;
        let scales = self.read(site.scale.expect("an int8 site carries a scale"))?;

        buffer.clear();
        buffer.reserve(rows * columns * 2);

        for row in 0..rows {
            let scale = f32::from_le_bytes(
                scales[row * 4..row * 4 + 4]
                    .try_into()
                    .expect("four bytes of scale"),
            );
            let row_codes = &codes[row * columns..(row + 1) * columns];
            for &code in row_codes {
                push_bf16(buffer, f32::from(code as i8) * scale);
            }
        }

        Ok(())
    }

    /// One NVFP4 projection: codes times the per-tensor and per-block scales.
    fn decode_weight(&self, site: Site, buffer: &mut Vec<u8>) -> Result<(), Nvfp4Error> {
        let weight = self.tensor(site.weight);
        let rows = weight.shape[0] as usize;
        let packed = weight.shape[1] as usize;
        let columns = packed * 2;
        let blocks = columns / BLOCK;

        let codes = self.read(site.weight)?;
        let raw_scale = self.read(site.scale.expect("a quantized site carries a scale"))?;
        let scale = unswizzle(&raw_scale, rows, blocks);
        let per_tensor =
            self.scalar(site.per_tensor.expect("a quantized site carries a scalar"))?;
        let pre_quant = match site.pre_quant {
            Some(at) => Some(self.columns_bf16(at)?),
            None => None,
        };

        buffer.clear();
        buffer.reserve(rows * columns * 2);

        for row in 0..rows {
            let row_codes = &codes[row * packed..(row + 1) * packed];
            let row_scale = &scale[row * blocks..(row + 1) * blocks];

            for (block, &block_scale) in row_scale.iter().enumerate() {
                // `total_scale = per_tensor_scale * block_scale` (`:213-214`).
                let total = per_tensor * e4m3(block_scale);
                let start = block * (BLOCK / 2);

                for pair in 0..BLOCK / 2 {
                    let byte = row_codes[start + pair];
                    let column = block * BLOCK + pair * 2;
                    let (hi, lo) = match &pre_quant {
                        Some(gains) => (gains[column], gains[column + 1]),
                        None => (1.0, 1.0),
                    };

                    // `hi_first`: the even column is the high nibble
                    // (`float_utils.py:pack_uint4`).
                    push_bf16(buffer, E2M1[(byte >> 4) as usize] * total * hi);
                    push_bf16(buffer, E2M1[(byte & 0x0F) as usize] * total * lo);
                }
            }
        }

        Ok(())
    }

    /// One F32 scalar, the artifact's per-tensor scale.
    fn scalar(&self, at: usize) -> Result<f32, Nvfp4Error> {
        let bytes = self.read(at)?;
        let four: [u8; 4] = bytes
            .get(..4)
            .ok_or_else(|| Nvfp4Error::Format {
                name: self.tensor(at).name.clone(),
                detail: "a per-tensor scale is shorter than four bytes".to_string(),
            })?
            .try_into()
            .expect("four bytes");

        Ok(f32::from_le_bytes(four))
    }

    /// A BF16 column vector as F32, for the `pre_quant_scale` fold.
    fn columns_bf16(&self, at: usize) -> Result<Vec<f32>, Nvfp4Error> {
        let bytes = self.read(at)?;

        Ok(bytes
            .chunks_exact(2)
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect())
    }
}

/// Every part of a layer, in the order the forward reads them.
const ALL_PARTS: [Part; 11] = [
    Part::InputNorm,
    Part::Query,
    Part::Key,
    Part::Value,
    Part::QueryNorm,
    Part::KeyNorm,
    Part::Output,
    Part::PostNorm,
    Part::Gate,
    Part::Up,
    Part::Down,
];

/// The embedding: `I8 [vocab, hidden]` with an `F32 [vocab, 1]` row scale.
fn embedding_site(shard: &Shard, index: &BTreeMap<&str, usize>) -> Result<Site, Nvfp4Error> {
    let weight = find(index, EMBEDDING)?;
    expect(shard, weight, Dtype::I8, &[VOCAB, HIDDEN])?;

    let scale = find(index, "model.embed_tokens.weight_scale")?;
    expect(shard, scale, Dtype::F32, &[VOCAB, 1])?;

    Ok(Site {
        kind: Kind::Int8Rows,
        weight,
        scale: Some(scale),
        per_tensor: None,
        pre_quant: None,
    })
}

/// One layer's part: a BF16 norm, or a packed NVFP4 projection.
fn layer_site(
    shard: &Shard,
    index: &BTreeMap<&str, usize>,
    layer: u32,
    part: Part,
) -> Result<Site, Nvfp4Error> {
    let prefix = format!("model.layers.{layer}.");

    let norm = match part {
        Part::InputNorm => Some(("input_layernorm.weight", HIDDEN)),
        Part::PostNorm => Some(("post_attention_layernorm.weight", HIDDEN)),
        Part::QueryNorm => Some(("self_attn.q_norm.weight", HEAD_DIM)),
        Part::KeyNorm => Some(("self_attn.k_norm.weight", HEAD_DIM)),
        _ => None,
    };

    if let Some((suffix, width)) = norm {
        let name = format!("{prefix}{suffix}");
        let at = find(index, &name)?;
        expect(shard, at, Dtype::Bf16, &[width])?;

        return Ok(Site {
            kind: Kind::Bf16,
            weight: at,
            scale: None,
            per_tensor: None,
            pre_quant: None,
        });
    }

    let (suffix, out, width) = projection(part).expect("every part is a norm or a projection");
    let name = format!("{prefix}{suffix}.weight");
    let at = find(index, &name)?;
    expect(shard, at, Dtype::U8, &[out, width / 2])?;
    // [`unswizzle`] covers whole tiles only; the artifact's seven shapes all
    // divide TILE_ROWS rows and TILE_COLS scale columns exactly.
    if out % TILE_ROWS as u64 != 0 || (width / BLOCK as u64) % TILE_COLS as u64 != 0 {
        return Err(Nvfp4Error::Format {
            name: name.clone(),
            detail: format!(
                "{out} rows and {} blocks are not whole tiles",
                width / BLOCK as u64
            ),
        });
    }
    marker(shard, index, &format!("{prefix}{suffix}"))?;

    let scale = find(index, &format!("{prefix}{suffix}.weight_scale"))?;
    expect(shard, scale, Dtype::F8E4M3, &[out, width / BLOCK as u64])?;

    let per_tensor = find(index, &format!("{prefix}{suffix}.weight_scale_2"))?;
    expect(shard, per_tensor, Dtype::F32, &[])?;

    // `pre_quant_scale` is the fold's activation side on the two projections
    // no layernorm feeds (`comfy/ops.py:1350-1352`); the artifact carries it on
    // exactly `down_proj` and `o_proj` and on no other.
    let folded = matches!(part, Part::Down | Part::Output);
    let pre_quant = match find(index, &format!("{prefix}{suffix}.pre_quant_scale")) {
        Ok(at) => {
            if !folded {
                return Err(Nvfp4Error::Format {
                    name: name.clone(),
                    detail: "a layernorm-fed projection carries no pre_quant_scale".to_string(),
                });
            }
            expect(shard, at, Dtype::Bf16, &[width])?;
            Some(at)
        }
        Err(_) if folded => {
            return Err(Nvfp4Error::Format {
                name: name.clone(),
                detail: "no pre_quant_scale on a projection no layernorm feeds".to_string(),
            });
        }
        Err(_) => None,
    };

    Ok(Site {
        kind: Kind::Quantized,
        weight: at,
        scale: Some(scale),
        per_tensor: Some(per_tensor),
        pre_quant,
    })
}

/// `(suffix, output width, input width)` of the seven quantized projections.
fn projection(part: Part) -> Option<(&'static str, u64, u64)> {
    Some(match part {
        Part::Query => ("self_attn.q_proj", QUERY_DIM, HIDDEN),
        Part::Key => ("self_attn.k_proj", KV_DIM, HIDDEN),
        Part::Value => ("self_attn.v_proj", KV_DIM, HIDDEN),
        Part::Output => ("self_attn.o_proj", HIDDEN, QUERY_DIM),
        Part::Gate => ("mlp.gate_proj", INTERMEDIATE, HIDDEN),
        Part::Up => ("mlp.up_proj", INTERMEDIATE, HIDDEN),
        Part::Down => ("mlp.down_proj", HIDDEN, INTERMEDIATE),
        _ => return None,
    })
}

fn find(index: &BTreeMap<&str, usize>, name: &str) -> Result<usize, Nvfp4Error> {
    index
        .get(name)
        .copied()
        .ok_or_else(|| Nvfp4Error::Absent(name.to_string()))
}

/// Every dtype and dimension is checked before a byte is read (`h3_weights.c:144-197`).
fn expect(shard: &Shard, at: usize, dtype: Dtype, shape: &[u64]) -> Result<(), Nvfp4Error> {
    let tensor = &shard.tensors[at];

    if tensor.dtype != dtype {
        return Err(Nvfp4Error::Format {
            name: tensor.name.clone(),
            detail: format!(
                "dtype is {}, expected {}",
                tensor.dtype.name(),
                dtype.name()
            ),
        });
    }

    if tensor.shape != shape {
        return Err(Nvfp4Error::Format {
            name: tensor.name.clone(),
            detail: format!("shape is {:?}, expected {shape:?}", tensor.shape),
        });
    }

    Ok(())
}

/// `{"format": "nvfp4", ...}`, the artifact's own marker for a quantized weight
/// (`comfy/ops.py:1170-1183`); it sits beside the projection, not on the weight.
fn marker(
    shard: &Shard,
    index: &BTreeMap<&str, usize>,
    projection: &str,
) -> Result<(), Nvfp4Error> {
    let marker = format!("{projection}.comfy_quant");
    let at = find(index, &marker)?;
    let tensor = &shard.tensors[at];
    let mut bytes = vec![0u8; tensor.bytes() as usize];
    shard
        .read_data(tensor, &mut bytes)
        .map_err(|error| Nvfp4Error::Io {
            name: marker.clone(),
            detail: error.token(),
        })?;

    if !bytes.windows(5).any(|window| window == b"nvfp4") {
        return Err(Nvfp4Error::Format {
            name: marker,
            detail: "the marker does not name the nvfp4 format".to_string(),
        });
    }

    Ok(())
}

/// `from_blocked` (`comfy_kitchen/float_utils.py`): cuBLAS's 128x4 tiled
/// blocked layout back to `(rows, blocks)`.
///
/// The tiled layout keeps an array's shape and permutes its contents: tile
/// `(row / 128, column / 4)` occupies `TILE` consecutive elements, and inside
/// that tile the four 32-row groups interleave with the four columns. The
/// artifact stores every block scale this way, so a plain read is wrong.
///
/// Every shape the encoder meets divides 128 and 4 exactly -- 1024, 5120,
/// 8192 and 25600 rows, 320, 512 and 1600 scale columns -- so the padded case
/// the source code allows cannot arise here, and [`unswizzle`] refuses it
/// rather than copying a layout it has not measured.
fn unswizzle(blocked: &[u8], rows: usize, blocks: usize) -> Vec<u8> {
    debug_assert!(
        rows % TILE_ROWS == 0 && blocks % TILE_COLS == 0,
        "whole tiles only"
    );
    let mut out = vec![0u8; rows * blocks];

    for row in 0..rows {
        for column in 0..blocks {
            let index = blocked_offset(row, column, blocks);
            out[row * blocks + column] = blocked[index];
        }
    }

    out
}

/// Where `(row, column)`'s scale sits in the blocked layout: tile
/// `(row / TILE_ROWS, column / TILE_COLS)` first, then the group and lane
/// inside it.
fn blocked_offset(row: usize, column: usize, blocks: usize) -> usize {
    let column_blocks = blocks / TILE_COLS;
    let within = row % TILE_ROWS;
    let (group, lane) = (within / GROUP_ROWS, within % GROUP_ROWS);

    (row / TILE_ROWS * column_blocks + column / TILE_COLS) * TILE
        + lane * TILE_STRIDE
        + group * TILE_COLS
        + column % TILE_COLS
}

/// One stored block scale.
fn e4m3(code: u8) -> f32 {
    e4m3_table()[code as usize]
}

/// E4M3 (`1` sign, `4` exponent with bias `7`, `3` mantissa; the all-ones
/// pattern is not a number, and `448` is the largest finite value).
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();

    TABLE.get_or_init(|| {
        let mut table = [0.0f32; 256];

        for code in 0..256usize {
            let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
            let exponent = (code >> 3) & 0x0F;
            let mantissa = code & 0x07;

            if exponent == 0x0F && mantissa == 0x07 {
                table[code] = f32::NAN;
                continue;
            }

            let magnitude = if exponent == 0 {
                (mantissa as f32 / 8.0) * 2.0f32.powi(-6)
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
            };
            table[code] = sign * magnitude;
        }

        table
    })
}

/// Round to nearest even, the storage the forward uploads.
fn push_bf16(buffer: &mut Vec<u8>, value: f32) {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    buffer.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closed form against the source's own numpy `from_blocked`: offsets
    /// derived by running `comfy_kitchen/float_utils.py` on the Spark.
    #[test]
    fn blocked_offset_matches_the_source() {
        let blocks = 8;

        for (row, column, expected) in [
            (0usize, 0usize, 0usize),
            (0, 4, 512),
            (1, 0, 16),
            (1, 1, 17),
            (0, 7, 515),
            (32, 0, 4),
            (64, 3, 11),
            (96, 5, 525),
            (127, 7, 1023),
        ] {
            assert_eq!(
                blocked_offset(row, column, blocks),
                expected,
                "({row}, {column}) should read the blocked offset {expected}"
            );
        }
    }

    /// The tiled layout is a permutation: every stored scale is used once.
    ///
    /// Whole tiles only, which is the precondition the loader checks at open.
    #[test]
    fn blocked_offset_is_a_permutation() {
        for (rows, blocks) in [(128, 8), (256, 4), (1280, 20)] {
            let mut seen = vec![false; rows * blocks];

            for row in 0..rows {
                for column in 0..blocks {
                    let at = blocked_offset(row, column, blocks);
                    assert!(at < rows * blocks && !seen[at], "offset {at} repeated");
                    seen[at] = true;
                }
            }
        }
    }

    /// The copy loop reads the offsets [`blocked_offset`] names.
    #[test]
    fn unswizzle_copies_by_offset() {
        let (rows, blocks) = (128, 8);
        let blocked: Vec<u8> = (0..rows * blocks).map(|at| (at % 251) as u8).collect();
        let out = unswizzle(&blocked, rows, blocks);

        for (row, column) in [(0usize, 0usize), (1, 0), (64, 3), (127, 7)] {
            let expected = (blocked_offset(row, column, blocks) % 251) as u8;
            assert_eq!(out[row * blocks + column], expected);
        }
    }

    #[test]
    fn e4m3_decodes_its_ends() {
        let table = e4m3_table();

        assert_eq!(table[0x00], 0.0);
        assert_eq!(table[0x7E], 448.0);
        assert_eq!(table[0x38], 1.0);
        assert_eq!(table[0xFE], -448.0);
        assert!(table[0x7F].is_nan());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        let mut buffer = Vec::new();
        push_bf16(&mut buffer, 1.0);
        push_bf16(&mut buffer, -2.0);

        assert_eq!(u16::from_le_bytes([buffer[0], buffer[1]]), 0x3F80);
        assert_eq!(u16::from_le_bytes([buffer[2], buffer[3]]), 0xC000);
    }
}
