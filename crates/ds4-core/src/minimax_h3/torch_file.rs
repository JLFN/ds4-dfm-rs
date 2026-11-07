//! The `torch.save` container and the narrow pickle it carries: the cache
//! file format the release's own `torch.load(..., weights_only=True)` reads
//! and `torch.save` writes.
//!
//! Source of truth, read first-hand for this unit:
//!
//! - `torch/serialization.py` @ v2.13.0: `save` and `_save` (`DEFAULT_PROTOCOL`
//!   2, the `PyTorchPickler` whose `persistent_id` returns
//!   `("storage", storage_type, storage_key, location, storage_numel)` for a
//!   storage, and the record set `data.pkl` + `.format_version` `"1"` +
//!   `.storage_alignment` `"64"` + `byteorder` + `data/<key>` per storage);
//!   `load` and `_load` (record lookup by name, the storage `persistent_load`
//!   tuple shape, `torch.Size`, and the `version` requirement); `_legacy_save`
//!   for the old tar format, which this port does not read.
//! - `torch/_utils.py`: `_rebuild_tensor_v2(storage, storage_offset, size,
//!   stride, requires_grad, backward_hooks, metadata=None)`.
//! - `torch/_weights_only_unpickler.py`: the exact opcode set and the global
//!   allowlist a `weights_only=True` load accepts.
//! - `torch/storage.py`: `_dtype_to_storage_type_map()` names the bfloat16
//!   storage `"BFloat16Storage"`; `_get_dtype_from_pickle_storage_type` reads
//!   it back.
//! - `caffe2/serialize/inline_container.{h,cc}` and `versions.h`: the writer
//!   is ZIP64, uncompressed, names every record `<basename>/<record>` (the
//!   reader takes the archive name from the first entry), pads each record to
//!   a 64-byte boundary through an `FB` extra field, writes a CRC32 per
//!   record, and stamps `version` with `kMinProducedFileFormatVersion` = 3 as
//!   `"3\n"`; the reader requires `version` within
//!   [`kMinSupportedFileFormatVersion` 1, `kMaxSupportedFileFormatVersion` 10].
//!
//! ```text
//! file          = zip64(zip32 eocd fields filled too)
//! entry         = <archive>/data.pkl            pickle protocol 2
//!                 <archive>/.format_version     "1"
//!                 <archive>/.storage_alignment  "64"
//!                 <archive>/byteorder           "little"
//!                 <archive>/data/<key>          raw storage bytes
//!                 <archive>/version             "3\n"
//! data.pkl      = dict(...) with one REDUCE per tensor:
//!                 torch._utils._rebuild_tensor_v2(
//!                     persistent_id("storage", torch.BFloat16Storage,
//!                                   key, "cpu", numel),
//!                     storage_offset, torch.Size(shape), torch.Size(stride),
//!                     requires_grad, collections.OrderedDict())
//! ```
//!
//! Deviations, named and bounded:
//!
//! - Only stored (method 0) records are read. `torch.save` and the reference
//!   writer both store uncompressed; a compressed record is refused by name
//!   rather than mishandled.
//! - The writer omits `.data/serialization_id`, which the reference writer
//!   derives from `c10::hash` of the record names and the combined record
//!   CRC32 (`inline_container.cc::writeSerializationId`). The reference reader
//!   treats it as optional (`hasRecord`), so a load is unaffected; it is
//!   metadata only and nothing in the port reads it.
//! - `_rebuild_tensor_v2` is written with six arguments (through
//!   `backward_hooks`); `metadata` defaults to `None` in the reference
//!   signature. The reference's C++ reducer may pass a seventh.
//! - The reader resolves a tensor only through `_rebuild_tensor_v2` (and the
//!   four-argument `_rebuild_tensor`); any other rebuild function is refused
//!   by name.
//! - A record of 4 GiB or more is refused: the entry headers are written with
//!   32-bit sizes.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

/// `inline_container.cc::detail::getPadding` alignment.
const ALIGNMENT: u64 = 64;
/// `PyTorchStreamWriter::version_` default, `versions.h`.
const PRODUCED_VERSION: &[u8] = b"3\n";
/// `versions.h`: `kMinSupportedFileFormatVersion`.
const MIN_VERSION: u64 = 1;
/// `versions.h`: `kMaxSupportedFileFormatVersion`.
const MAX_VERSION: u64 = 10;
/// The 32-bit "look in the zip64 extra field" sentinel.
const SENTINEL32: u64 = 0xFFFF_FFFF;

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const LOCATOR_SIG: u32 = 0x0706_4b50;

/// The zip polynomial, for the CRC32 the reader verifies and the writer emits.
static CRC32_TABLE: [u32; 256] = crc32_table();

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// CRC32 (IEEE 802.3), the checksum every zip record carries.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &byte in data {
        c = CRC32_TABLE[((c ^ byte as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// The dtypes this container carries. `torch/storage.py`'s legacy names are
/// kept because that is what the pickle references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorchDtype {
    Bool,
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
}

impl TorchDtype {
    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::UInt8 | Self::Int8 => 1,
            Self::Int16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 | Self::Float64 => 8,
        }
    }

    /// The `torch.dtype` repr, as `str(dtype)` prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "torch.bool",
            Self::UInt8 => "torch.uint8",
            Self::Int8 => "torch.int8",
            Self::Int16 => "torch.int16",
            Self::Int32 => "torch.int32",
            Self::Int64 => "torch.int64",
            Self::Float16 => "torch.float16",
            Self::BFloat16 => "torch.bfloat16",
            Self::Float32 => "torch.float32",
            Self::Float64 => "torch.float64",
        }
    }

    /// `_dtype_to_storage_type_map` (`torch/storage.py`).
    pub fn storage_name(self) -> &'static str {
        match self {
            Self::Bool => "BoolStorage",
            Self::UInt8 => "ByteStorage",
            Self::Int8 => "CharStorage",
            Self::Int16 => "ShortStorage",
            Self::Int32 => "IntStorage",
            Self::Int64 => "LongStorage",
            Self::Float16 => "HalfStorage",
            Self::BFloat16 => "BFloat16Storage",
            Self::Float32 => "FloatStorage",
            Self::Float64 => "DoubleStorage",
        }
    }

    /// `_get_dtype_from_pickle_storage_type` (`torch/storage.py`).
    pub fn from_storage_name(name: &str) -> Option<Self> {
        Some(match name {
            "BoolStorage" => Self::Bool,
            "ByteStorage" => Self::UInt8,
            "CharStorage" => Self::Int8,
            "ShortStorage" => Self::Int16,
            "IntStorage" => Self::Int32,
            "LongStorage" => Self::Int64,
            "HalfStorage" => Self::Float16,
            "BFloat16Storage" => Self::BFloat16,
            "FloatStorage" => Self::Float32,
            "DoubleStorage" => Self::Float64,
            _ => return None,
        })
    }
}

/// One tensor as the container stores it: a dtype, a shape, a stride, an
/// offset into `bytes`, and the storage bytes it views.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub dtype: TorchDtype,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub storage_offset: u64,
    pub bytes: Vec<u8>,
}

impl Tensor {
    /// A contiguous tensor: the row-major stride, no storage offset.
    pub fn contiguous(dtype: TorchDtype, shape: Vec<u64>, bytes: Vec<u8>) -> Self {
        let mut stride = vec![0u64; shape.len()];
        let mut running = 1u64;
        for axis in (0..shape.len()).rev() {
            stride[axis] = running;
            running *= shape[axis];
        }
        Self {
            dtype,
            shape,
            stride,
            storage_offset: 0,
            bytes,
        }
    }

    /// Elements the tensor's shape describes.
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The row-major stride for this shape.
    pub fn row_major(shape: &[u64]) -> Vec<u64> {
        let mut stride = vec![0u64; shape.len()];
        let mut running = 1u64;
        for axis in (0..shape.len()).rev() {
            stride[axis] = running;
            running *= shape[axis];
        }
        stride
    }

    /// The tensor's own bytes when it is a contiguous view from offset zero.
    /// `None` for a strided or offset view, which this port never produces.
    pub fn contiguous_bytes(&self) -> Option<&[u8]> {
        if self.storage_offset != 0 || self.stride != Self::row_major(&self.shape) {
            return None;
        }
        let len = self.numel() as usize * self.dtype.size();
        self.bytes.get(..len)
    }

    /// Whether every element is finite. For the float dtypes the check is the
    /// exponent field; the integer dtypes are always finite.
    pub fn all_finite(&self) -> bool {
        let bytes = match self.contiguous_bytes() {
            Some(bytes) => bytes,
            None => return false,
        };
        match self.dtype {
            TorchDtype::BFloat16 => bytes.chunks_exact(2).all(|v| {
                let bits = u16::from_le_bytes([v[0], v[1]]);
                bits & 0x7F80 != 0x7F80
            }),
            TorchDtype::Float16 => bytes.chunks_exact(2).all(|v| {
                let bits = u16::from_le_bytes([v[0], v[1]]);
                bits & 0x7C00 != 0x7C00
            }),
            TorchDtype::Float32 => bytes.chunks_exact(4).all(|v| {
                let bits = u32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                bits & 0x7F80_0000 != 0x7F80_0000
            }),
            TorchDtype::Float64 => bytes.chunks_exact(8).all(|v| {
                let bits = u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]);
                bits & 0x7FF0_0000_0000_0000 != 0x7FF0_0000_0000_0000
            }),
            _ => true,
        }
    }
}

/// A storage as `persistent_load` returns it, before a rebuild function turns
/// it into a tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Storage {
    pub dtype: TorchDtype,
    pub key: String,
    pub location: String,
    pub bytes: Vec<u8>,
}

/// The pickle object graph, restricted to what the cache payload uses.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    /// A dict or `OrderedDict`, in insertion order.
    Map(Vec<(Value, Value)>),
    /// An unresolved class or function reference (`GLOBAL`).
    Global {
        module: String,
        name: String,
    },
    /// `torch.Size`.
    Size(Vec<u64>),
    Tensor(Tensor),
    Storage(Storage),
}

impl Value {
    pub fn as_map(&self) -> Option<&Vec<(Value, Value)>> {
        match self {
            Self::Map(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// A map value by string key. Duplicate keys keep the first, as a dict
    /// literal would.
    pub fn get(&self, key: &str) -> Option<&Value> {
        let entries = self.as_map()?;
        entries
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v)
    }
}

/// Build a `Map` from string keys.
pub fn map(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::Str(key.to_string()), value))
            .collect(),
    )
}

#[derive(Debug)]
pub enum TorchError {
    Io { path: PathBuf, detail: String },
    Format(String),
    Pickle(String),
    Value(String),
    Unsupported(String),
}

impl TorchError {
    /// A one-token form for a gate line.
    pub fn token(&self) -> String {
        match self {
            Self::Io { path, detail } => format!("io {}: {detail}", path.display()),
            Self::Format(detail) => format!("format: {detail}"),
            Self::Pickle(detail) => format!("pickle: {detail}"),
            Self::Value(detail) => format!("value: {detail}"),
            Self::Unsupported(detail) => format!("unsupported: {detail}"),
        }
    }
}

impl fmt::Display for TorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.token())
    }
}

impl std::error::Error for TorchError {}

struct Entry {
    name: String,
    method: u16,
    crc: u32,
    comp_size: u64,
    uncomp_size: u64,
    local_offset: u64,
}

/// A parsed zip archive: the bytes, the archive name the writer used, and the
/// central directory.
pub struct Archive {
    pub path: PathBuf,
    pub archive_name: String,
    bytes: Vec<u8>,
    entries: Vec<Entry>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self, TorchError> {
        let bytes = std::fs::read(path).map_err(|error| TorchError::Io {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })?;
        let mut archive = Self {
            path: path.to_path_buf(),
            archive_name: String::new(),
            bytes,
            entries: Vec::new(),
        };
        archive.parse()?;
        Ok(archive)
    }

    /// The store's records, with the archive prefix stripped, in file order.
    pub fn records(&self) -> Vec<String> {
        let prefix = format!("{}/", self.archive_name);
        self.entries
            .iter()
            .map(|entry| {
                entry
                    .name
                    .strip_prefix(&prefix)
                    .unwrap_or(&entry.name)
                    .to_string()
            })
            .collect()
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.name == self.full_name(name))
    }

    /// The `version` record, parsed and checked against the reference bounds.
    pub fn version(&self) -> Result<u64, TorchError> {
        let name = if self.has(".data/version") {
            ".data/version"
        } else {
            "version"
        };
        let raw = self.record(name)?;
        let text = String::from_utf8(raw)
            .map_err(|_| TorchError::Format("version record is not utf-8".to_string()))?;
        let value: u64 = text
            .trim()
            .parse()
            .map_err(|_| TorchError::Format(format!("version {text:?} is not a number")))?;
        if !(MIN_VERSION..=MAX_VERSION).contains(&value) {
            return Err(TorchError::Format(format!(
                "version {value} outside the supported range {MIN_VERSION}..={MAX_VERSION}"
            )));
        }
        Ok(value)
    }

    /// One logical record's bytes, CRC-checked.
    pub fn record(&self, name: &str) -> Result<Vec<u8>, TorchError> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.name == self.full_name(name))
            .ok_or_else(|| TorchError::Format(format!("no record {name:?}")))?;
        if entry.method != 0 {
            return Err(TorchError::Unsupported(format!(
                "record {name:?} uses zip method {} (only stored)",
                entry.method
            )));
        }
        if entry.comp_size != entry.uncomp_size {
            return Err(TorchError::Format(format!(
                "record {name:?} stores {} bytes for {} uncompressed",
                entry.comp_size, entry.uncomp_size
            )));
        }
        let start = self.local_data_offset(entry)?;
        let end = start + entry.comp_size as usize;
        let data = self
            .bytes
            .get(start..end)
            .ok_or_else(|| TorchError::Format(format!("record {name:?} runs past the file")))?;
        let crc = crc32(data);
        if crc != entry.crc {
            return Err(TorchError::Format(format!(
                "record {name:?} crc {crc:08x} != stored {:08x}",
                entry.crc
            )));
        }
        Ok(data.to_vec())
    }

    /// Read and check `data.pkl`'s bytes, then run the unpickler.
    pub fn load(&self) -> Result<Value, TorchError> {
        let data = self.record("data.pkl")?;
        let mut unpickler = Unpickler {
            data: &data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: BTreeMap::new(),
            storages: BTreeMap::new(),
            archive: self,
        };
        unpickler.run()
    }

    fn full_name(&self, name: &str) -> String {
        format!("{}/{}", self.archive_name, name)
    }

    fn local_data_offset(&self, entry: &Entry) -> Result<usize, TorchError> {
        let base = entry.local_offset as usize;
        let header = self
            .bytes
            .get(base..base + 30)
            .ok_or_else(|| TorchError::Format("local header runs past the file".to_string()))?;
        if read_u32(header, 0) != LOCAL_SIG {
            return Err(TorchError::Format("bad local header signature".to_string()));
        }
        let name_len = read_u16(header, 26) as usize;
        let extra_len = read_u16(header, 28) as usize;
        Ok(base + 30 + name_len + extra_len)
    }

    fn parse(&mut self) -> Result<(), TorchError> {
        let eocd = self.find_eocd()?;
        let (cd_offset, cd_size, total) = self.directory_span(eocd)?;
        let mut cursor = cd_offset as usize;
        let end = cd_offset as usize + cd_size as usize;
        for _ in 0..total {
            if cursor + 46 > end {
                break;
            }
            let head = &self.bytes[cursor..cursor + 46];
            if read_u32(head, 0) != CENTRAL_SIG {
                return Err(TorchError::Format(
                    "bad central directory signature".to_string(),
                ));
            }
            let method = read_u16(head, 10);
            let crc = read_u32(head, 16);
            let comp_size = read_u32(head, 20) as u64;
            let uncomp_size = read_u32(head, 24) as u64;
            let name_len = read_u16(head, 28) as usize;
            let extra_len = read_u16(head, 30) as usize;
            let comment_len = read_u16(head, 32) as usize;
            let local_offset = read_u32(head, 42) as u64;
            let name_at = cursor + 46;
            let name = self
                .bytes
                .get(name_at..name_at + name_len)
                .ok_or_else(|| TorchError::Format("central name runs past the file".to_string()))?;
            let name = String::from_utf8_lossy(name).into_owned();
            let extra = self
                .bytes
                .get(name_at + name_len..name_at + name_len + extra_len)
                .unwrap_or(&[]);
            // A zip64 extra field replaces only the fields that the 32-bit
            // header sentinelled (0xFFFFFFFF), in a fixed order.
            let (mut comp_size, mut uncomp_size, mut local_offset) =
                (comp_size, uncomp_size, local_offset);
            if let Some((z_comp, z_uncomp, z_offset)) = zip64_extra(extra) {
                if comp_size == SENTINEL32 {
                    comp_size = z_comp;
                }
                if uncomp_size == SENTINEL32 {
                    uncomp_size = z_uncomp;
                }
                if local_offset == SENTINEL32 {
                    local_offset = z_offset;
                }
            }
            self.entries.push(Entry {
                name,
                method,
                crc,
                comp_size,
                uncomp_size,
                local_offset,
            });
            cursor = name_at + name_len + extra_len + comment_len;
        }
        if self.entries.is_empty() {
            return Err(TorchError::Format("archive holds no records".to_string()));
        }
        let first = &self.entries[0].name;
        let cut = first.find('/').ok_or_else(|| {
            TorchError::Format(format!("record {first:?} is not in a subdirectory"))
        })?;
        self.archive_name = first[..cut].to_string();
        Ok(())
    }

    fn find_eocd(&self) -> Result<usize, TorchError> {
        let len = self.bytes.len();
        if len < 22 {
            return Err(TorchError::Format(
                "file is shorter than an end record".to_string(),
            ));
        }
        let floor = len.saturating_sub(22 + 65535);
        for start in (floor..=len - 22).rev() {
            if read_u32(&self.bytes, start) == EOCD_SIG {
                return Ok(start);
            }
        }
        Err(TorchError::Format(
            "no end-of-central-directory record".to_string(),
        ))
    }

    /// The central directory's offset, size and entry count, preferring the
    /// zip64 record when the locator is present.
    fn directory_span(&self, eocd: usize) -> Result<(u64, u64, u64), TorchError> {
        let head = &self.bytes[eocd..eocd + 22];
        let mut total = read_u16(head, 10) as u64;
        let mut cd_size = read_u32(head, 12) as u64;
        let mut cd_offset = read_u32(head, 16) as u64;
        if eocd >= 20 {
            let locator = &self.bytes[eocd - 20..eocd];
            if read_u32(locator, 0) == LOCATOR_SIG {
                let at = read_u64(locator, 8) as usize;
                let record = self.bytes.get(at..at + 56).ok_or_else(|| {
                    TorchError::Format("zip64 end record runs past the file".to_string())
                })?;
                if read_u32(record, 0) == EOCD64_SIG {
                    total = read_u64(record, 32);
                    cd_size = read_u64(record, 40);
                    cd_offset = read_u64(record, 48);
                }
            }
        }
        Ok((cd_offset, cd_size, total))
    }
}

fn zip64_extra(extra: &[u8]) -> Option<(u64, u64, u64)> {
    let mut at = 0usize;
    while at + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[at], extra[at + 1]]);
        let size = u16::from_le_bytes([extra[at + 2], extra[at + 3]]) as usize;
        let body = extra.get(at + 4..at + 4 + size)?;
        if id == 0x0001 {
            // Only the sentinelled fields are present, in the zip64 order:
            // uncompressed size, compressed size, local header offset, disk.
            let mut values = [SENTINEL32; 3];
            let mut cursor = 0usize;
            for slot in values.iter_mut() {
                if cursor + 8 > body.len() {
                    break;
                }
                *slot = read_u64(body, cursor);
                cursor += 8;
            }
            return Some((values[1], values[0], values[2]));
        }
        at += 4 + size;
    }
    None
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(raw)
}

/// The unpickler: protocol 2's opcode set, restricted to the globals
/// `_weights_only_unpickler._get_allowed_globals` allows that this container
/// uses.
struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<Vec<Value>>,
    memo: BTreeMap<u64, Value>,
    storages: BTreeMap<String, Storage>,
    archive: &'a Archive,
}

impl<'a> Unpickler<'a> {
    fn run(&mut self) -> Result<Value, TorchError> {
        loop {
            let opcode = self.byte()?;
            match opcode {
                0x80 => {
                    let protocol = self.byte()?;
                    if protocol > 2 {
                        return Err(TorchError::Unsupported(format!(
                            "pickle protocol {protocol} (only 2)"
                        )));
                    }
                }
                b'.' => {
                    return self
                        .stack
                        .pop()
                        .ok_or_else(|| TorchError::Pickle("STOP with an empty stack".to_string()));
                }
                b'(' => self.marks.push(std::mem::take(&mut self.stack)),
                b'}' => self.stack.push(Value::Map(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let value = self.byte()? as i64;
                    self.stack.push(Value::Int(value));
                }
                b'M' => {
                    let lo = self.byte()? as u16;
                    let hi = self.byte()? as u16;
                    self.stack.push(Value::Int((hi << 8 | lo) as i64));
                }
                b'J' => {
                    let raw = self.take(4)?;
                    self.stack.push(Value::Int(
                        i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64,
                    ));
                }
                0x8A => {
                    let len = self.byte()? as usize;
                    let raw = self.take(len)?;
                    self.stack.push(Value::Int(decode_long(raw)));
                }
                b'G' => {
                    let raw = self.take(8)?;
                    let mut fixed = [0u8; 8];
                    fixed.copy_from_slice(raw);
                    self.stack.push(Value::Float(f64::from_be_bytes(fixed)));
                }
                b'X' => {
                    let len = read_u32(self.take(4)?, 0) as usize;
                    let raw = self.take(len)?;
                    let text = String::from_utf8(raw.to_vec())
                        .map_err(|_| TorchError::Pickle("BINUNICODE is not utf-8".to_string()))?;
                    self.stack.push(Value::Str(text));
                }
                b'T' | b'U' => {
                    let len = if opcode == b'T' {
                        read_u32(self.take(4)?, 0) as usize
                    } else {
                        self.byte()? as usize
                    };
                    let raw = self.take(len)?;
                    self.stack.push(Value::Bytes(raw.to_vec()));
                }
                b'q' => {
                    let index = self.byte()? as u64;
                    self.memo.insert(index, self.peek()?.clone());
                }
                b'r' => {
                    let index = read_u32(self.take(4)?, 0) as u64;
                    self.memo.insert(index, self.peek()?.clone());
                }
                b'h' => {
                    let index = self.byte()? as u64;
                    let value = self.lookup(index)?;
                    self.stack.push(value);
                }
                b'j' => {
                    let index = read_u32(self.take(4)?, 0) as u64;
                    let value = self.lookup(index)?;
                    self.stack.push(value);
                }
                b'a' => {
                    let item = self.pop()?;
                    match self.peek_mut()? {
                        Value::List(items) => items.push(item),
                        _ => return Err(TorchError::Pickle("APPEND to a non-list".to_string())),
                    }
                }
                b'e' => {
                    let items = self.pop_mark();
                    match self.peek_mut()? {
                        Value::List(list) => list.extend(items),
                        _ => return Err(TorchError::Pickle("APPENDS to a non-list".to_string())),
                    }
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.peek_mut()? {
                        Value::Map(entries) => entries.push((key, value)),
                        _ => return Err(TorchError::Pickle("SETITEM to a non-dict".to_string())),
                    }
                }
                b'u' => {
                    let items = self.pop_mark();
                    let pairs: Vec<(Value, Value)> = items
                        .chunks(2)
                        .map(|pair| (pair[0].clone(), pair[1].clone()))
                        .collect();
                    match self.peek_mut()? {
                        Value::Map(entries) => entries.extend(pairs),
                        _ => return Err(TorchError::Pickle("SETITEMS to a non-dict".to_string())),
                    }
                }
                b't' => {
                    let items = self.pop_mark();
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => self.wrap_tuple(1)?,
                0x86 => self.wrap_tuple(2)?,
                0x87 => self.wrap_tuple(3)?,
                b'Q' => {
                    let id = self.pop()?;
                    let storage = self.persistent_load(id)?;
                    self.stack.push(storage);
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.check_global(&module, &name)?;
                    self.stack.push(Value::Global { module, name });
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let value = self.reduce(func, args)?;
                    self.stack.push(value);
                }
                b'b' => {
                    let state = self.pop()?;
                    let inst = self.peek()?.clone();
                    self.build(&inst, state)?;
                }
                other => {
                    return Err(TorchError::Unsupported(format!(
                        "pickle opcode {other:#04x} at {}",
                        self.pos - 1
                    )));
                }
            }
        }
    }

    fn wrap_tuple(&mut self, count: usize) -> Result<(), TorchError> {
        let at = self.stack.len() - count;
        let items = self.stack.split_off(at);
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    fn check_global(&self, module: &str, name: &str) -> Result<(), TorchError> {
        let known = match module {
            "torch" => {
                TorchDtype::from_storage_name(name).is_some()
                    || matches!(name, "Size" | "Tensor" | "device")
            }
            "torch._utils" => matches!(
                name,
                "_rebuild_tensor_v2" | "_rebuild_tensor" | "_rebuild_parameter"
            ),
            "collections" => name == "OrderedDict",
            "torch.nn.parameter" => name == "Parameter",
            _ => false,
        };
        if known {
            return Ok(());
        }
        Err(TorchError::Unsupported(format!(
            "global {module}.{name} is not one this container resolves"
        )))
    }

    fn reduce(&mut self, func: Value, args: Value) -> Result<Value, TorchError> {
        let Value::Global { module, name } = &func else {
            return Err(TorchError::Pickle("REDUCE of a non-global".to_string()));
        };
        let args = match args {
            Value::Tuple(items) => items,
            other => vec![other],
        };
        match (module.as_str(), name.as_str()) {
            ("torch", "Size") => Ok(Value::Size(as_u64s(args.first())?)),
            ("collections", "OrderedDict") => Ok(Value::Map(Vec::new())),
            ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
                self.rebuild_tensor(&args)
            }
            _ => Err(TorchError::Unsupported(format!(
                "REDUCE of {module}.{name}"
            ))),
        }
    }

    fn rebuild_tensor(&self, args: &[Value]) -> Result<Value, TorchError> {
        let storage = match args.first() {
            Some(Value::Storage(storage)) => storage,
            _ => {
                return Err(TorchError::Pickle(
                    "tensor rebuild takes a storage".to_string(),
                ))
            }
        };
        let storage_offset = args.get(1).and_then(Value::as_i64).unwrap_or(0).max(0) as u64;
        let shape = as_u64s(args.get(2))?;
        let stride = as_u64s(args.get(3))?;
        Ok(Value::Tensor(Tensor {
            dtype: storage.dtype,
            shape,
            stride,
            storage_offset,
            bytes: storage.bytes.clone(),
        }))
    }

    fn build(&mut self, inst: &Value, state: Value) -> Result<(), TorchError> {
        match (inst, state) {
            (Value::Map(_), Value::Map(state)) => {
                if let Some(Value::Map(entries)) = self.stack.last_mut() {
                    entries.extend(state);
                    return Ok(());
                }
                Err(TorchError::Pickle("BUILD mirrored a non-dict".to_string()))
            }
            _ => Err(TorchError::Unsupported("BUILD for a non-dict".to_string())),
        }
    }

    /// `_load`'s storage `persistent_load`: the tuple is
    /// `("storage", storage class, key, location, numel)`; the bytes come from
    /// the record `data/<key>`.
    fn persistent_load(&mut self, id: Value) -> Result<Value, TorchError> {
        let Value::Tuple(items) = &id else {
            return Err(TorchError::Pickle(
                "persistent id is not a tuple".to_string(),
            ));
        };
        let tag = items.first().and_then(Value::as_str);
        if tag != Some("storage") {
            return Err(TorchError::Unsupported(format!(
                "persistent id {tag:?} (only storage)"
            )));
        }
        let (module, name) = match items.get(1) {
            Some(Value::Global { module, name }) => (module.as_str(), name.as_str()),
            _ => {
                return Err(TorchError::Pickle(
                    "storage class is not a global".to_string(),
                ))
            }
        };
        if module != "torch" {
            return Err(TorchError::Unsupported(format!(
                "storage class {module}.{name}"
            )));
        }
        let dtype = if name == "UntypedStorage" {
            TorchDtype::UInt8
        } else {
            TorchDtype::from_storage_name(name)
                .ok_or_else(|| TorchError::Unsupported(format!("storage class {name}")))?
        };
        let key = items
            .get(2)
            .and_then(Value::as_str)
            .ok_or_else(|| TorchError::Pickle("storage key is not a string".to_string()))?
            .to_string();
        let location = items
            .get(3)
            .and_then(Value::as_str)
            .unwrap_or("cpu")
            .to_string();
        let numel = items
            .get(4)
            .and_then(Value::as_i64)
            .ok_or_else(|| TorchError::Pickle("storage numel is not an int".to_string()))?;
        if location != "cpu" {
            return Err(TorchError::Unsupported(format!(
                "storage location {location:?} (only cpu)"
            )));
        }
        if let Some(cached) = self.storages.get(&key) {
            return Ok(Value::Storage(cached.clone()));
        }
        let bytes = self.archive.record(&format!("data/{key}"))?;
        let expected = numel as usize * dtype.size();
        if bytes.len() != expected {
            return Err(TorchError::Format(format!(
                "storage {key} holds {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        let storage = Storage {
            dtype,
            key: key.clone(),
            location,
            bytes,
        };
        self.storages.insert(key, storage.clone());
        Ok(Value::Storage(storage))
    }

    fn byte(&mut self) -> Result<u8, TorchError> {
        let byte = *self
            .data
            .get(self.pos)
            .ok_or_else(|| TorchError::Pickle("truncated pickle".to_string()))?;
        self.pos += 1;
        Ok(byte)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], TorchError> {
        let slice = self
            .data
            .get(self.pos..self.pos + count)
            .ok_or_else(|| TorchError::Pickle("truncated pickle".to_string()))?;
        self.pos += count;
        Ok(slice)
    }

    /// A newline-terminated line, for `GLOBAL`.
    fn line(&mut self) -> Result<String, TorchError> {
        let rest = &self.data[self.pos..];
        let end = rest
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(|| TorchError::Pickle("unterminated GLOBAL".to_string()))?;
        let text = String::from_utf8(rest[..end].to_vec())
            .map_err(|_| TorchError::Pickle("GLOBAL is not utf-8".to_string()))?;
        self.pos += end + 1;
        Ok(text)
    }

    fn pop(&mut self) -> Result<Value, TorchError> {
        self.stack
            .pop()
            .ok_or_else(|| TorchError::Pickle("stack underflow".to_string()))
    }

    fn peek(&self) -> Result<&Value, TorchError> {
        self.stack
            .last()
            .ok_or_else(|| TorchError::Pickle("stack underflow".to_string()))
    }

    fn peek_mut(&mut self) -> Result<&mut Value, TorchError> {
        self.stack
            .last_mut()
            .ok_or_else(|| TorchError::Pickle("stack underflow".to_string()))
    }

    fn lookup(&self, index: u64) -> Result<Value, TorchError> {
        self.memo
            .get(&index)
            .cloned()
            .ok_or_else(|| TorchError::Pickle(format!("memo {index} is unset")))
    }

    fn pop_mark(&mut self) -> Vec<Value> {
        let items = std::mem::take(&mut self.stack);
        if let Some(mark) = self.marks.pop() {
            self.stack = mark;
        }
        items
    }
}

fn as_u64s(value: Option<&Value>) -> Result<Vec<u64>, TorchError> {
    match value {
        Some(Value::Size(items)) => Ok(items.clone()),
        Some(Value::Tuple(items)) | Some(Value::List(items)) => items
            .iter()
            .map(|item| {
                item.as_i64()
                    .filter(|value| *value >= 0)
                    .map(|value| value as u64)
                    .ok_or_else(|| TorchError::Pickle("size is not a non-negative int".to_string()))
            })
            .collect(),
        _ => Err(TorchError::Pickle("expected a size tuple".to_string())),
    }
}

/// Python's `pickle` `LONG1` little-endian two's-complement decode.
fn decode_long(raw: &[u8]) -> i64 {
    if raw.is_empty() {
        return 0;
    }
    let mut value: i128 = 0;
    for (shift, &byte) in raw.iter().enumerate() {
        value |= (byte as i128) << (8 * shift);
    }
    if raw[raw.len() - 1] & 0x80 != 0 {
        value -= 1i128 << (8 * raw.len());
    }
    value as i64
}

/// Write `value` as a `torch.save` file. The zip is ZIP64 with 32-bit entry
/// sizes (all records here are small), records are 64-byte aligned through an
/// `FB` extra field, and every record carries its CRC32.
pub fn write(path: &Path, value: &Value) -> Result<(), TorchError> {
    let archive_name = basename(path)?;
    let mut emitter = Emitter::default();
    // Protocol 2, the object, then STOP (`torch/serialization.py`,
    // `DEFAULT_PROTOCOL = 2`).
    emitter.byte(0x80);
    emitter.byte(2);
    emitter.emit(value)?;
    emitter.byte(b'.');
    let data_pkl = std::mem::take(&mut emitter.out);

    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    records.push(("data.pkl".to_string(), data_pkl));
    records.push((".format_version".to_string(), b"1".to_vec()));
    records.push((
        ".storage_alignment".to_string(),
        ALIGNMENT.to_string().into_bytes(),
    ));
    records.push(("byteorder".to_string(), b"little".to_vec()));
    for storage in &emitter.storages {
        records.push((format!("data/{}", storage.key), storage.bytes.clone()));
    }
    records.push(("version".to_string(), PRODUCED_VERSION.to_vec()));

    write_archive(path, &archive_name, &records)
}

/// The reference's `basename` (`inline_container.cc`): the file's final
/// component, cut at its last dot. That is the archive name every record is
/// prefixed with.
pub fn basename(path: &Path) -> Result<String, TorchError> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let start = name.rfind(['/', '\\']).map(|at| at + 1).unwrap_or(0);
    let tail = &name[start..];
    let end = tail.rfind('.').unwrap_or(tail.len());
    let stem = &tail[..end];
    if stem.is_empty() {
        return Err(TorchError::Value(format!(
            "{} has no archive name",
            path.display()
        )));
    }
    Ok(stem.to_string())
}

struct StorageOut {
    key: String,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct Emitter {
    out: Vec<u8>,
    storages: Vec<StorageOut>,
}

impl Emitter {
    fn emit(&mut self, value: &Value) -> Result<(), TorchError> {
        match value {
            Value::None => self.byte(b'N'),
            Value::Bool(true) => self.byte(0x88),
            Value::Bool(false) => self.byte(0x89),
            Value::Int(number) => self.int(*number),
            Value::Float(number) => {
                self.byte(b'G');
                self.out.extend_from_slice(&number.to_be_bytes());
            }
            Value::Str(text) => {
                self.byte(b'X');
                self.out
                    .extend_from_slice(&(text.len() as u32).to_le_bytes());
                self.out.extend_from_slice(text.as_bytes());
            }
            Value::Bytes(raw) => {
                self.byte(b'T');
                self.out
                    .extend_from_slice(&(raw.len() as u32).to_le_bytes());
                self.out.extend_from_slice(raw);
            }
            Value::List(items) => {
                self.byte(b']');
                if !items.is_empty() {
                    self.byte(b'(');
                    for item in items {
                        self.emit(item)?;
                    }
                    self.byte(b'e');
                }
            }
            Value::Tuple(items) => self.tuple(items)?,
            Value::Size(items) => {
                // `torch.Size.__reduce__` returns `(Size, (the_tuple,))`, so
                // the args tuple wraps one tuple of ints.
                self.global("torch", "Size");
                self.byte(b'(');
                self.byte(b'(');
                for item in items {
                    self.emit(&Value::Int(*item as i64))?;
                }
                self.byte(b't');
                self.byte(b't');
                self.byte(b'R');
            }
            Value::Map(entries) => {
                self.byte(b'}');
                if !entries.is_empty() {
                    self.byte(b'(');
                    for (key, item) in entries {
                        self.emit(key)?;
                        self.emit(item)?;
                    }
                    self.byte(b'u');
                }
            }
            Value::Global { module, name } => self.global(module, name),
            Value::Tensor(tensor) => self.tensor(tensor)?,
            Value::Storage(storage) => {
                return Err(TorchError::Value(format!(
                    "storage {} has no place outside a tensor",
                    storage.key
                )));
            }
        }
        Ok(())
    }

    fn tensor(&mut self, tensor: &Tensor) -> Result<(), TorchError> {
        let numel = tensor.bytes.len() / tensor.dtype.size();
        let key = self.storages.len().to_string();
        self.storages.push(StorageOut {
            key: key.clone(),
            bytes: tensor.bytes.clone(),
        });

        self.global("torch._utils", "_rebuild_tensor_v2");
        self.byte(b'(');
        // The storage reference: the persisted-id tuple, then BINPERSID.
        self.byte(b'(');
        self.emit(&Value::Str("storage".to_string()))?;
        self.global("torch", tensor.dtype.storage_name());
        self.emit(&Value::Str(key))?;
        self.emit(&Value::Str("cpu".to_string()))?;
        self.emit(&Value::Int(numel as i64))?;
        self.byte(b't');
        self.byte(b'Q');
        self.emit(&Value::Int(tensor.storage_offset as i64))?;
        self.emit(&Value::Size(tensor.shape.clone()))?;
        self.emit(&Value::Size(tensor.stride.clone()))?;
        self.byte(0x89); // requires_grad False
                         // backward_hooks: an empty OrderedDict, as the reference passes.
        self.global("collections", "OrderedDict");
        self.byte(b')');
        self.byte(b'R');
        self.byte(b't');
        self.byte(b'R');
        Ok(())
    }

    fn tuple(&mut self, items: &[Value]) -> Result<(), TorchError> {
        match items.len() {
            0 => self.byte(b')'),
            1 => {
                self.emit(&items[0])?;
                self.byte(0x85);
            }
            2 => {
                self.emit(&items[0])?;
                self.emit(&items[1])?;
                self.byte(0x86);
            }
            3 => {
                self.emit(&items[0])?;
                self.emit(&items[1])?;
                self.emit(&items[2])?;
                self.byte(0x87);
            }
            _ => {
                self.byte(b'(');
                for item in items {
                    self.emit(item)?;
                }
                self.byte(b't');
            }
        }
        Ok(())
    }

    fn int(&mut self, number: i64) {
        if (0..=0xFF).contains(&number) {
            self.byte(b'K');
            self.byte(number as u8);
        } else if (0..=0xFFFF).contains(&number) {
            self.byte(b'M');
            self.out.extend_from_slice(&(number as u16).to_le_bytes());
        } else if (i32::MIN as i64..=i32::MAX as i64).contains(&number) {
            self.byte(b'J');
            self.out.extend_from_slice(&(number as i32).to_le_bytes());
        } else {
            let raw = long_bytes(number);
            self.byte(0x8A);
            self.byte(raw.len() as u8);
            self.out.extend_from_slice(&raw);
        }
    }

    fn global(&mut self, module: &str, name: &str) {
        self.byte(b'c');
        self.out.extend_from_slice(module.as_bytes());
        self.byte(b'\n');
        self.out.extend_from_slice(name.as_bytes());
        self.byte(b'\n');
    }

    fn byte(&mut self, byte: u8) {
        self.out.push(byte);
    }
}

/// The shortest `LONG1` body for `number`, little-endian two's complement.
fn long_bytes(number: i64) -> Vec<u8> {
    let raw = number.to_le_bytes();
    let mut end = raw.len();
    let fill = if number < 0 { 0xFF } else { 0x00 };
    while end > 1 {
        let last = raw[end - 1];
        let next = raw[end - 2];
        let redundant = if fill == 0xFF {
            last == 0xFF && next & 0x80 != 0
        } else {
            last == 0x00 && next & 0x80 == 0
        };
        if !redundant {
            break;
        }
        end -= 1;
    }
    raw[..end].to_vec()
}

fn write_archive(
    path: &Path,
    archive_name: &str,
    records: &[(String, Vec<u8>)],
) -> Result<(), TorchError> {
    struct Cd {
        name: String,
        crc: u32,
        size: u64,
        offset: u64,
    }
    let mut out: Vec<u8> = Vec::new();
    let mut directory: Vec<Cd> = Vec::new();

    for (logical, data) in records {
        let full = format!("{archive_name}/{logical}");
        if data.len() as u64 >= 0xFFFF_FFFF {
            return Err(TorchError::Unsupported(format!(
                "record {logical} is 4 GiB or more"
            )));
        }
        let cursor = out.len() as u64;
        let start = cursor + 30 + full.len() as u64 + 4;
        let padding = (ALIGNMENT - start % ALIGNMENT) % ALIGNMENT;
        let extra_len = padding + 4;
        let crc = crc32(data);

        out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        out.extend_from_slice(&45u16.to_le_bytes()); // needs zip64
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // stored
        out.extend_from_slice(&0u16.to_le_bytes()); // time
        out.extend_from_slice(&0x0021u16.to_le_bytes()); // date, 1980-01-01
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(full.len() as u16).to_le_bytes());
        out.extend_from_slice(&(extra_len as u16).to_le_bytes());
        out.extend_from_slice(full.as_bytes());
        // The `FB` extra: 4 header bytes, then `padding` filler, which lands
        // the record's data on the next 64-byte boundary.
        out.extend_from_slice(b"FB");
        out.extend_from_slice(&(padding as u16).to_le_bytes());
        out.extend(std::iter::repeat(b'Z').take(padding as usize));
        out.extend_from_slice(data);

        directory.push(Cd {
            name: full,
            crc,
            size: data.len() as u64,
            offset: cursor,
        });
    }

    let cd_offset = out.len() as u64;
    for entry in &directory {
        out.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        out.extend_from_slice(&0x031Eu16.to_le_bytes()); // made by 3.30
        out.extend_from_slice(&45u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0x0021u16.to_le_bytes());
        out.extend_from_slice(&entry.crc.to_le_bytes());
        out.extend_from_slice(&(entry.size as u32).to_le_bytes());
        out.extend_from_slice(&(entry.size as u32).to_le_bytes());
        out.extend_from_slice(&(entry.name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra
        out.extend_from_slice(&0u16.to_le_bytes()); // comment
        out.extend_from_slice(&0u16.to_le_bytes()); // disk
        out.extend_from_slice(&0u16.to_le_bytes()); // internal
        out.extend_from_slice(&0u32.to_le_bytes()); // external
        out.extend_from_slice(&(entry.offset as u32).to_le_bytes());
        out.extend_from_slice(entry.name.as_bytes());
    }
    let cd_size = out.len() as u64 - cd_offset;
    let count = directory.len() as u64;

    // ZIP64 end record, then its locator, then the classic end record with the
    // real counts so a zip32-only reader also works.
    let eocd64 = out.len() as u64;
    out.extend_from_slice(&EOCD64_SIG.to_le_bytes());
    out.extend_from_slice(&44u64.to_le_bytes());
    out.extend_from_slice(&0x031Eu16.to_le_bytes());
    out.extend_from_slice(&45u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());

    out.extend_from_slice(&LOCATOR_SIG.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&eocd64.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());

    out.extend_from_slice(&EOCD_SIG.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(count as u16).to_le_bytes());
    out.extend_from_slice(&(count as u16).to_le_bytes());
    out.extend_from_slice(&(cd_size as u32).to_le_bytes());
    out.extend_from_slice(&(cd_offset as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    let mut file = std::fs::File::create(path).map_err(|error| TorchError::Io {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    file.write_all(&out).map_err(|error| TorchError::Io {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        let video = Tensor::contiguous(
            TorchDtype::BFloat16,
            vec![1, 2, 2],
            vec![0x00, 0x3F, 0x00, 0x40, 0x00, 0x41, 0x00, 0x42],
        );
        map(vec![
            (
                "contexts",
                map(vec![
                    ("video", Value::Tensor(video)),
                    ("schema", Value::Int(1)),
                ]),
            ),
            ("flag", Value::Bool(true)),
            ("missing", Value::None),
            ("ratio", Value::Float(0.5)),
            ("names", Value::List(vec![Value::Str("a".to_string())])),
        ])
    }

    #[test]
    fn the_crc_matches_the_zip_polynomial() {
        // IEEE 802.3 check value for "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn the_archive_name_is_the_basename_cut_at_its_last_dot() {
        assert_eq!(
            basename(Path::new("/x/prompt_cache.pt")).unwrap(),
            "prompt_cache"
        );
        assert_eq!(basename(Path::new("a.b.pt")).unwrap(), "a.b");
        assert_eq!(basename(Path::new("cache")).unwrap(), "cache");
    }

    #[test]
    fn the_long_encoding_is_the_shortest_two_complement_body() {
        // +2147483648 needs five bytes: four would read back negative.
        assert_eq!(long_bytes(0x8000_0000), vec![0x00, 0x00, 0x00, 0x80, 0x00]);
        assert_eq!(long_bytes(-0x8000_0001), vec![0xFF, 0xFF, 0xFF, 0x7F, 0xFF]);
        assert_eq!(long_bytes(-1), vec![0xFF]);
        assert_eq!(long_bytes(-0x80), vec![0x80]);
    }

    #[test]
    fn a_payload_round_trips_through_the_container() {
        let dir = std::env::temp_dir().join(format!("h3-torch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prompt_cache.pt");
        let value = sample();
        write(&path, &value).unwrap();

        let archive = Archive::open(&path).unwrap();
        assert_eq!(archive.archive_name, "prompt_cache");
        assert_eq!(
            archive.records(),
            vec![
                "data.pkl",
                ".format_version",
                ".storage_alignment",
                "byteorder",
                "data/0",
                "version"
            ]
        );
        assert_eq!(archive.version().unwrap(), 3);
        assert_eq!(archive.record("byteorder").unwrap(), b"little".to_vec());
        assert_eq!(archive.record("data/0").unwrap().len(), 8);
        assert_eq!(archive.load().unwrap(), value);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_unpickler_reads_a_memoized_torch_style_pickle() {
        // `torch.save`'s pickler is `pickle.Pickler`, so its `data.pkl` is
        // memoized: BINPUT/BINGET for the strings, LONG_BINPUT/LONG_BINGET for
        // a repeated class. The port's writer emits none of these, so the
        // golden path needs its own coverage. The payload is
        // `{"x": tensor, "y": tensor}` with one shared bfloat16 storage.
        let mut pkl: Vec<u8> = vec![0x80, 0x02];
        pkl.extend_from_slice(&[b'}', b'q', 0x00]); // EMPTY_DICT; BINPUT 0
        pkl.push(b'('); // MARK
        unicode(&mut pkl, "x");
        pkl.extend_from_slice(&[b'q', 0x01]); // BINPUT 1
        tensor_into(&mut pkl, 2);
        pkl.extend_from_slice(&[b'q', 0x03]); // BINPUT 3, the tensor
        unicode(&mut pkl, "y");
        pkl.extend_from_slice(&[b'q', 0x04]); // BINPUT 4
        pkl.extend_from_slice(&[b'h', 0x03]); // BINGET 3, the same tensor
        pkl.push(b'u'); // SETITEMS
        pkl.push(b'.'); // STOP

        let dir = std::env::temp_dir().join(format!("h3-torch-memo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.pt");
        let records = vec![
            ("data.pkl".to_string(), pkl),
            ("byteorder".to_string(), b"little".to_vec()),
            ("data/0".to_string(), vec![0x00, 0x3F, 0x00, 0x40]),
            ("version".to_string(), b"3\n".to_vec()),
        ];
        write_archive(&path, "c", &records).unwrap();

        let value = Archive::open(&path).unwrap().load().unwrap();
        let entries = value.as_map().unwrap();
        assert_eq!(entries.len(), 2);
        for (key, item) in entries {
            let name = key.as_str().unwrap();
            assert!(name == "x" || name == "y");
            let Value::Tensor(tensor) = item else {
                panic!("{name} is not a tensor");
            };
            assert_eq!(tensor.dtype, TorchDtype::BFloat16);
            assert_eq!(tensor.shape, vec![2]);
            assert_eq!(tensor.stride, vec![1]);
            assert_eq!(tensor.bytes, vec![0x00, 0x3F, 0x00, 0x40]);
        }
        // The two entries are the same memoized object.
        assert_eq!(entries[0].1, entries[1].1);

        std::fs::remove_dir_all(&dir).ok();
    }

    fn unicode(out: &mut Vec<u8>, text: &str) {
        out.push(b'X');
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    /// `torch._utils._rebuild_tensor_v2(storage, 0, Size((numel,)),
    /// Size((1,)), False, OrderedDict())`, with `torch.Size` memoized the way
    /// the second use can be a LONG_BINGET.
    fn tensor_into(out: &mut Vec<u8>, numel: u8) {
        out.push(b'c');
        out.extend_from_slice(b"torch._utils\n_rebuild_tensor_v2\n");
        out.push(b'(');
        out.push(b'('); // the persisted-id tuple
        unicode(out, "storage");
        out.extend_from_slice(b"ctorch\nBFloat16Storage\n");
        unicode(out, "0");
        unicode(out, "cpu");
        out.extend_from_slice(&[b'K', numel]);
        out.push(b't');
        out.push(b'Q');
        out.extend_from_slice(&[b'K', 0x00]);
        out.extend_from_slice(b"ctorch\nSize\n");
        out.extend_from_slice(&[b'r', 0x05, 0x00, 0x00, 0x00]); // LONG_BINPUT 5
        out.extend_from_slice(&[b'(', b'(', b'K', numel, b't', b't', b'R']);
        out.extend_from_slice(&[b'j', 0x05, 0x00, 0x00, 0x00]); // LONG_BINGET 5
        out.extend_from_slice(&[b'(', b'(', b'K', 0x01, b't', b't', b'R']);
        out.push(0x89); // NEWFALSE
        out.extend_from_slice(b"ccollections\nOrderedDict\n");
        out.push(b')');
        out.push(b'R');
        out.push(b't');
        out.push(b'R');
    }

    #[test]
    fn a_stored_crc_is_verified_on_read() {
        let dir = std::env::temp_dir().join(format!("h3-torch-crc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.pt");
        write(&path, &sample()).unwrap();

        // Flip one byte inside the data/0 record and expect the read to refuse.
        let mut bytes = std::fs::read(&path).unwrap();
        let archive = Archive::open(&path).unwrap();
        let entry = archive
            .entries
            .iter()
            .find(|entry| entry.name == "c/data/0")
            .unwrap();
        let at = archive.local_data_offset(entry).unwrap();
        bytes[at] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let error = Archive::open(&path).unwrap().record("data/0").unwrap_err();
        assert!(error.token().contains("crc"), "{}", error.token());

        std::fs::remove_dir_all(&dir).ok();
    }
}
