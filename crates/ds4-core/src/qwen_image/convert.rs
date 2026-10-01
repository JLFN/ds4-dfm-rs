//! VAE format tool: `qwen_image_2.1_vae_bf16.safetensors` -> a GGUF whose
//! layout is pinned by [`crate::qwen_image::vae_decode_contract`].
//!
//! Why a converter and not a second reader: this tree reads GGUF only
//! (`crate::gguf`), and a safetensors reader would put a second format and a
//! second tensor-resolution path in the runtime. Converting once keeps one
//! format, one loader and one layout contract.
//!
//! Two conversions happen here, and both are byte-level:
//!
//! 1. Scope. The exporter ships the encoder half too (238 tensors); decode-only
//!    serving needs 134 (`decoder.*` plus `conv2.*`). The reference gates the
//!    encoder and the top-level `conv1` on `decode_only`
//!    (`wan_vae.hpp:1092-1099`), so dropping them mirrors the reference.
//! 2. Layout. Torch stores a conv weight as `[OC, IC, kT, kH, kW]`; the
//!    reference's kernel derives `OC` from `w->ne[3] / IC`
//!    (`ggml_extend.cpp:452-475`), so the pinned layout is `{kW, kH, kT, IC*OC}`
//!    for conv3d, `{kW, kH, IC, OC}` for the conv2d resample, and a flat `{C}`
//!    for the norms and biases. Every tensor is re-indexed to that order.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{vae_decode_contract, ImageTensor};

/// GGUF alignment this writer emits and the reader assumes when no
/// `general.alignment` key is present (`crate::gguf`: default 32).
const ALIGNMENT: u64 = 32;
const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
const HEADER_BYTES: u64 = 24; // magic + version + n_tensors + n_kv
/// `ggml_type` ids written into the directory.
const TYPE_BF16: u32 = super::TYPE_BF16;
/// The safetensors header is a JSON tensor table; 64 MiB is far above any real
/// one and stops a corrupt length from being allocated.
const HEADER_CAP: u64 = 64 << 20;

#[derive(Debug)]
pub enum ConvertError {
    Io(std::io::Error),
    Header(String),
    /// The source does not match the pinned contract.
    Contract { tensor: String, why: String },
    Unsupported(String),
}

impl ConvertError {
    pub fn token(&self) -> String {
        match self {
            ConvertError::Io(e) => format!("io {e}"),
            ConvertError::Header(m) => format!("header {m}"),
            ConvertError::Contract { tensor, why } => format!("contract {tensor}: {why}"),
            ConvertError::Unsupported(m) => format!("unsupported {m}"),
        }
    }
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for ConvertError {}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        ConvertError::Io(e)
    }
}

/// One source tensor from the safetensors header.
#[derive(Clone, Debug)]
pub struct SourceTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}

/// The header, parsed. `data_start` is the first data byte, since safetensors
/// offsets are relative to the end of the header.
#[derive(Clone, Debug)]
pub struct Safetensors {
    pub tensors: Vec<SourceTensor>,
    pub data_start: u64,
}

impl Safetensors {
    pub fn find(&self, name: &str) -> Option<&SourceTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Reads and parses only the header, so a 675 MB file is not slurped.
pub fn read_header(path: &Path) -> Result<Safetensors, ConvertError> {
    let mut f = File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > HEADER_CAP {
        return Err(ConvertError::Header(format!("declared length {n} is not a header")));
    }
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    let json: serde_json::Value = serde_json::from_slice(&buf)
        .map_err(|e| ConvertError::Header(format!("json: {e}")))?;
    let map = json
        .as_object()
        .ok_or_else(|| ConvertError::Header("not an object".into()))?;

    let mut tensors = Vec::new();
    for (name, v) in map {
        if name == "__metadata__" {
            continue;
        }
        // `__metadata__` is the only non-tensor key in this exporter's files.
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| ConvertError::Header(format!("{name}: no dtype")))?
            .to_string();
        let shape: Vec<u64> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| ConvertError::Header(format!("{name}: no shape")))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0))
            .collect();
        let offs = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| ConvertError::Header(format!("{name}: no data_offsets")))?;
        let begin = offs.first().and_then(|x| x.as_u64()).unwrap_or(0);
        let end = offs.get(1).and_then(|x| x.as_u64()).unwrap_or(0);
        tensors.push(SourceTensor { name: name.clone(), dtype, shape, begin, end });
    }
    tensors.sort_by(|a, b| a.begin.cmp(&b.begin));
    Ok(Safetensors { tensors, data_start: 8 + n })
}

/// Torch `[OC, IC, kT, kH, kW]` -> ggml `{kW, kH, kT, IC*OC}`: the conv3d
/// weight order the reference's kernel consumes.
pub fn permute_conv3d(
    src: &[u8],
    oc_n: u64,
    ic_n: u64,
    kt_n: u64,
    kh_n: u64,
    kw_n: u64,
    esz: usize,
) -> Vec<u8> {
    let count = (oc_n * ic_n * kt_n * kh_n * kw_n) as usize;
    let mut dst = vec![0u8; count * esz];
    for oc in 0..oc_n {
        for ic in 0..ic_n {
            for kt in 0..kt_n {
                for kh in 0..kh_n {
                    for kw in 0..kw_n {
                        let s = (((oc * ic_n + ic) * kt_n + kt) * kh_n + kh) * kw_n + kw;
                        // IC is the fastest of the flattened `IC*OC` axis.
                        let d = kw + kw_n * (kh + kh_n * (kt + kt_n * (ic + ic_n * oc)));
                        dst[d as usize * esz..(d as usize + 1) * esz]
                            .copy_from_slice(&src[s as usize * esz..(s as usize + 1) * esz]);
                    }
                }
            }
        }
    }
    dst
}

/// Torch `[OC, IC, kH, kW]` -> ggml `{kW, kH, IC, OC}`: the conv2d order the
/// VAE's `resample` convs are stored in.
pub fn permute_conv2d(src: &[u8], oc_n: u64, ic_n: u64, kh_n: u64, kw_n: u64, esz: usize) -> Vec<u8> {
    let count = (oc_n * ic_n * kh_n * kw_n) as usize;
    let mut dst = vec![0u8; count * esz];
    for oc in 0..oc_n {
        for ic in 0..ic_n {
            for kh in 0..kh_n {
                for kw in 0..kw_n {
                    let s = ((oc * ic_n + ic) * kh_n + kh) * kw_n + kw;
                    let d = kw + kw_n * (kh + kh_n * (ic + ic_n * oc));
                    dst[d as usize * esz..(d as usize + 1) * esz]
                        .copy_from_slice(&src[s as usize * esz..(s as usize + 1) * esz]);
                }
            }
        }
    }
    dst
}

/// The pinned ggml dims a source shape must produce. Also the gate: a source
/// that would not reproduce the contract dims exactly is refused instead of
/// silently written in the exporter's order.
pub fn map_source_dims(shape: &[u64], want: &[u64]) -> Result<Vec<u64>, String> {
    match want.len() {
        1 => {
            let n: u64 = shape.iter().product();
            if n != want[0] {
                return Err(format!("{shape:?} has {n} values, contract wants {}", want[0]));
            }
            Ok(vec![n])
        }
        4 if shape.len() == 5 => {
            let (oc, ic, kt, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
            let got = vec![kw, kh, kt, ic * oc];
            if got != want {
                return Err(format!("conv3d {shape:?} -> {got:?}, contract wants {want:?}"));
            }
            Ok(got)
        }
        4 if shape.len() == 4 => {
            let (oc, ic, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
            let got = vec![kw, kh, ic, oc];
            if got != want {
                return Err(format!("conv2d {shape:?} -> {got:?}, contract wants {want:?}"));
            }
            Ok(got)
        }
        _ => Err(format!("rank {} cannot satisfy contract dims {want:?}", shape.len())),
    }
}

/// Re-indexes one source tensor into the pinned layout.
fn to_pinned(src: &[u8], st: &SourceTensor, want: &ImageTensor) -> Result<Vec<u8>, ConvertError> {
    map_source_dims(&st.shape, &want.dims)
        .map_err(|why| ConvertError::Contract { tensor: st.name.clone(), why })?;
    let out = if want.dims.len() == 1 {
        src.to_vec()
    } else if st.shape.len() == 5 {
        permute_conv3d(
            src, st.shape[0], st.shape[1], st.shape[2], st.shape[3], st.shape[4], 2,
        )
    } else {
        permute_conv2d(src, st.shape[0], st.shape[1], st.shape[2], st.shape[3], 2)
    };
    let want_bytes = want.bytes();
    if out.len() as u64 != want_bytes {
        return Err(ConvertError::Contract {
            tensor: st.name.clone(),
            why: format!("wrote {} bytes, contract pins {want_bytes}", out.len()),
        });
    }
    Ok(out)
}

/// What the conversion produced, for the evidence ledger.
#[derive(Clone, Debug)]
pub struct ConvertReport {
    pub tensors: usize,
    pub bytes: u64,
    pub dropped: usize,
    pub source_tensors: usize,
}

impl ConvertReport {
    pub fn line(&self, out: &Path) -> String {
        format!(
            "vae-convert out={} tensors={} bytes={} dropped={} source_tensors={}",
            out.display(),
            self.tensors,
            self.bytes,
            self.dropped,
            self.source_tensors
        )
    }
}

/// Converts `src` (safetensors) to `out` (GGUF, decode-only, pinned layout).
///
/// The bytes land in `<out>.partial` first and are renamed on success, so a
/// contract failure never leaves a half-written artifact behind.
pub fn convert_vae(src: &Path, out: &Path) -> Result<ConvertReport, ConvertError> {
    let tmp = std::path::PathBuf::from(format!("{}.partial", out.display()));
    match convert_into(src, &tmp) {
        Ok(report) => {
            std::fs::rename(&tmp, out)?;
            Ok(report)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn convert_into(src: &Path, out: &Path) -> Result<ConvertReport, ConvertError> {
    let st = read_header(src)?;
    let contract = vae_decode_contract();

    // Bytes of the directory: one entry per tensor: name, rank, dims, type,
    // offset.
    let mut dir_bytes = 0u64;
    for t in &contract {
        dir_bytes += 8 + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
    }
    let mut offsets = Vec::with_capacity(contract.len());
    let mut cursor = 0u64;
    for t in &contract {
        offsets.push(cursor);
        cursor += crate::tensors::align_up(t.bytes(), ALIGNMENT);
    }
    let data_pos = crate::tensors::align_up(HEADER_BYTES + dir_bytes, ALIGNMENT);

    let mut w = BufWriter::new(File::create(out)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(contract.len() as u64).to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // n_kv: identification is name-based
    for (t, off) in contract.iter().zip(&offsets) {
        w.write_all(&(t.name.len() as u64).to_le_bytes())?;
        w.write_all(t.name.as_bytes())?;
        w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
        for d in &t.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&TYPE_BF16.to_le_bytes())?;
        w.write_all(&off.to_le_bytes())?;
    }
    let written = HEADER_BYTES + dir_bytes;
    for _ in written..data_pos {
        w.write_all(&[0u8])?;
    }

    let mut f = File::open(src)?;
    let mut bytes = 0u64;
    for t in &contract {
        let st_t = st.find(&t.name).ok_or_else(|| ConvertError::Contract {
            tensor: t.name.clone(),
            why: "absent from the safetensors header".into(),
        })?;
        if st_t.dtype != "BF16" {
            return Err(ConvertError::Unsupported(format!(
                "{} is {}, the contract pins BF16",
                t.name, st_t.dtype
            )));
        }
        let len = st_t.end - st_t.begin;
        let mut raw = vec![0u8; len as usize];
        f.seek(SeekFrom::Start(st.data_start + st_t.begin))?;
        f.read_exact(&mut raw)?;
        let pinned = to_pinned(&raw, st_t, t)?;
        w.write_all(&pinned)?;
        for _ in pinned.len() as u64..crate::tensors::align_up(pinned.len() as u64, ALIGNMENT) {
            w.write_all(&[0u8])?;
        }
        debug_assert_eq!(t.bytes(), pinned.len() as u64);
        bytes += pinned.len() as u64;
    }
    w.flush()?;

    Ok(ConvertReport {
        tensors: contract.len(),
        bytes,
        dropped: st.tensors.len() - contract.len(),
        source_tensors: st.tensors.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permute is proven on a tensor whose value encodes its own index
    /// tuple, so a wrong axis order cannot pass.
    #[test]
    fn conv3d_permute_places_index_tuples_at_the_pinned_offsets() {
        let (oc_n, ic_n, kt_n, kh_n, kw_n) = (3u64, 2u64, 2u64, 2u64, 3u64);
        let n = (oc_n * ic_n * kt_n * kh_n * kw_n) as usize;
        let mut src = vec![0u8; n * 2];
        for oc in 0..oc_n {
            for ic in 0..ic_n {
                for kt in 0..kt_n {
                    for kh in 0..kh_n {
                        for kw in 0..kw_n {
                            let s = (((oc * ic_n + ic) * kt_n + kt) * kh_n + kh) * kw_n + kw;
                            let v = (s as u16).to_le_bytes();
                            src[s as usize * 2..s as usize * 2 + 2].copy_from_slice(&v);
                        }
                    }
                }
            }
        }
        let dst = permute_conv3d(&src, oc_n, ic_n, kt_n, kh_n, kw_n, 2);
        for oc in 0..oc_n {
            for ic in 0..ic_n {
                for kt in 0..kt_n {
                    for kh in 0..kh_n {
                        for kw in 0..kw_n {
                            let s = (((oc * ic_n + ic) * kt_n + kt) * kh_n + kh) * kw_n + kw;
                            let d = kw + kw_n * (kh + kh_n * (kt + kt_n * (ic + ic_n * oc)));
                            let got = u16::from_le_bytes([
                                dst[d as usize * 2],
                                dst[d as usize * 2 + 1],
                            ]);
                            assert_eq!(got, s as u16, "oc={oc} ic={ic} kt={kt} kh={kh} kw={kw}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn conv2d_permute_places_index_tuples_at_the_pinned_offsets() {
        let (oc_n, ic_n, kh_n, kw_n) = (2u64, 3u64, 2u64, 2u64);
        let n = (oc_n * ic_n * kh_n * kw_n) as usize;
        let mut src = vec![0u8; n * 2];
        for oc in 0..oc_n {
            for ic in 0..ic_n {
                for kh in 0..kh_n {
                    for kw in 0..kw_n {
                        let s = ((oc * ic_n + ic) * kh_n + kh) * kw_n + kw;
                        src[s as usize * 2..s as usize * 2 + 2]
                            .copy_from_slice(&(s as u16).to_le_bytes());
                    }
                }
            }
        }
        let dst = permute_conv2d(&src, oc_n, ic_n, kh_n, kw_n, 2);
        for oc in 0..oc_n {
            for ic in 0..ic_n {
                for kh in 0..kh_n {
                    for kw in 0..kw_n {
                        let s = ((oc * ic_n + ic) * kh_n + kh) * kw_n + kw;
                        let d = kw + kw_n * (kh + kh_n * (ic + ic_n * oc));
                        let got = u16::from_le_bytes([dst[d as usize * 2], dst[d as usize * 2 + 1]]);
                        assert_eq!(got, s as u16);
                    }
                }
            }
        }
    }

    #[test]
    fn source_dims_must_reproduce_the_contract() {
        // conv3d: torch [OC, IC, KT, KH, KW] -> {KW, KH, KT, IC*OC}
        assert_eq!(
            map_source_dims(&[4, 144, 1, 3, 3], &[3, 3, 1, 576]).unwrap(),
            vec![3, 3, 1, 576]
        );
        // conv2d: torch [OC, IC, KH, KW] -> {KW, KH, IC, OC}
        assert_eq!(
            map_source_dims(&[1152, 1152, 3, 3], &[3, 3, 1152, 1152]).unwrap(),
            vec![3, 3, 1152, 1152]
        );
        // gamma: any trailing singleton shape reduces to {C}
        assert_eq!(map_source_dims(&[1152, 1, 1, 1], &[1152]).unwrap(), vec![1152]);
        // a mismatched shape is refused, not written
        assert!(map_source_dims(&[4, 144, 1, 3, 3], &[3, 3, 1, 288]).is_err());
    }

    #[test]
    fn contract_covers_only_decoder_and_conv2() {
        let c = vae_decode_contract();
        assert!(c.iter().all(|t| t.name.starts_with("decoder.") || t.name.starts_with("conv2.")));
        assert!(c.iter().all(|t| !t.name.starts_with("encoder") && !t.name.starts_with("conv1.")));
    }
}
