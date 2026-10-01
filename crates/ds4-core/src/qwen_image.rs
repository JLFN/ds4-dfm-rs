//! Qwen-Image-2.1: the image engine kind, its artifact contracts and its plan.
//!
//! Image generation is a sibling ENGINE beside the text families, not a
//! `ModelFamily`: the autoregressive contract (KV cache, banks, prefix reuse,
//! snapshots, MTP) has no meaning for a diffusion graph, so this module owns
//! its own catalogue instead of widening `shape.rs`.
//!
//! P0 scope is identification and contracts only: the two artifact layouts, the
//! quantization pinning, the plan/refusal skeleton and the placement model. No
//! kernel, graph or sampler is built here.
//!
//! The reference is `stable-diffusion.cpp` at `6dcb5bb`; every dimension below
//! was read from its source or from the artifacts on this host, never inferred.
//! `qwen_image_2_1.hpp:8-52` holds the DiT config and `wan_vae.hpp:1072-1090`
//! the VAE config; both files are indexed by `docs/qwen-image-2.1-recipe.md`.

pub mod convert;
pub mod dit;
pub mod oracle;

use crate::gguf::{GgufError, GgufFile};
use crate::tensors::{TensorError, TensorInventory};

/// GGML type ids pinned by the two contracts (`ggml.h` enum order).
pub const TYPE_F32: u32 = 0;
pub const TYPE_BF16: u32 = 30;
pub const TYPE_Q6_K: u32 = 14;

/// The engine's name in reports and refusals.
pub const ENGINE: &str = "qwen-image-2.1";

// ---------------------------------------------------------------------------
// DiT geometry (`qwen_image_2_1.hpp:8-52`)
// ---------------------------------------------------------------------------

/// Transformer blocks. One artifact variant exists; the count is pinned.
pub const DIT_LAYERS: u32 = 32;
pub const DIT_HIDDEN: u64 = 4096;
pub const DIT_HEAD_DIM: u64 = 128;
/// `hidden / head_dim` (`qwen_image_2_1.hpp:161`).
pub const DIT_HEADS: u64 = DIT_HIDDEN / DIT_HEAD_DIM;
pub const DIT_CONTEXT_DIM: u64 = 4096;
pub const DIT_INTERMEDIATE: u64 = 12288;
/// Latent channels in and out. There is no spatial patchify in 2.1: the image
/// goes through `DiT::patchify(image, 1, 1)`, so this is the raw channel count.
pub const DIT_IN_CHANNELS: u64 = 64;
pub const DIT_OUT_CHANNELS: u64 = 64;
/// Per-axis RoPE widths (`qwen_image_2_1.hpp:16`); 16 + 56 + 56 = 128.
pub const DIT_AXES_DIM: [u64; 3] = [16, 56, 56];
pub const DIT_TIME_EMBED_DIM: u64 = 256;
/// `4 * hidden`: the modulation parameter width before the row split.
pub const DIT_MODULATION: u64 = 4 * DIT_HIDDEN;
pub const DIT_ROPE_THETA: f32 = 10000.0;
/// `QwenImage21ZeroCenterRMSNorm` eps (`qwen_image_2_1.hpp:122-133`).
pub const DIT_NORM_EPS: f32 = 1e-6;

/// Tensors in the single target DiT artifact: 9 top-level plus 9 per block.
pub const DIT_TENSOR_COUNT: usize = 9 + (DIT_LAYERS as usize) * 9;
/// Q6_K matmul tensors in that artifact, measured on this host.
pub const DIT_Q6K_COUNT: usize = 229;
/// BF16 tensors in that artifact: the two boundary projections and the 64
/// per-head norm weights.
pub const DIT_BF16_COUNT: usize = 68;

// ---------------------------------------------------------------------------
// VAE geometry (`wan_vae.hpp:1072-1090`, `:1366-1389`)
// ---------------------------------------------------------------------------

pub const VAE_DEC_DIM: u64 = 144;
pub const VAE_Z_DIM: u64 = 64;
pub const VAE_OUT_CHANNELS: u64 = 4;
pub const VAE_DIM_MULT: [u64; 5] = [1, 2, 4, 8, 8];
/// `temperal_upsample`; the trailing `false` is why the last upsample level
/// carries no `time_conv` (`wan_vae.hpp:1086`).
pub const VAE_TEMPORAL_UPSAMPLE: [bool; 4] = [true, true, true, false];
/// `temperal_downsample`; the encoder is not part of the decode path.
pub const VAE_TEMPORAL_DOWNSAMPLE: [bool; 4] = [false, true, true, true];
/// `RMS_norm` eps (`wan_vae.hpp:118`).
pub const VAE_NORM_EPS: f32 = 1e-12;
/// `WanVAERunner::scale_factor` is `1.0f` and no version branch reassigns it
/// (`wan_vae.hpp:1293`); the run-time reload was verified, not assumed.
pub const VAE_SCALE_FACTOR: f32 = 1.0;

/// The 64-channel latent statistics (`wan_vae.hpp:1368-1375`). The two
/// conversions are
/// `vae -> diffusion: (latents - mean) * scale_factor / std` and
/// `diffusion -> vae: latents * std / scale_factor + mean` (`:1381-1389`).
pub const VAE_LATENT_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510,
    0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391,
    -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342,
    -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680,
    0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162,
    1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub const VAE_LATENT_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904,
    2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075,
    3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490,
    3.7662, 3.6746, 3.4563, 3.8161,
];

/// Decoder tensor count for the pinned decode-only layout: 132 `decoder.*` plus
/// `conv2.{weight,bias}`. The encoder and the top-level `conv1` are absent by
/// construction (`wan_vae.hpp:1092-1099` gates them on `decode_only`).
pub const VAE_DECODE_TENSOR_COUNT: usize = 134;

// ---------------------------------------------------------------------------
// Engine, modules, placement
// ---------------------------------------------------------------------------

/// The image engine kinds this tree knows. One variant exists; the enum is the
/// split point, not a flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageEngine {
    QwenImage21,
}

impl ImageEngine {
    pub fn name(self) -> &'static str {
        match self {
            ImageEngine::QwenImage21 => ENGINE,
        }
    }
}

/// The three modules the engine places independently (plan appendix A.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageModule {
    TextEncoder,
    Diffusion,
    Vae,
}

impl ImageModule {
    pub fn name(self) -> &'static str {
        match self {
            ImageModule::TextEncoder => "te",
            ImageModule::Diffusion => "diffusion",
            ImageModule::Vae => "vae",
        }
    }

    pub const ALL: [ImageModule; 3] = [
        ImageModule::TextEncoder,
        ImageModule::Diffusion,
        ImageModule::Vae,
    ];
}

/// The graph device a module's compute runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageDevice {
    Cpu,
    Cuda(u32),
}

impl ImageDevice {
    pub fn name(self) -> String {
        match self {
            ImageDevice::Cpu => "cpu".into(),
            ImageDevice::Cuda(i) => format!("cuda{i}"),
        }
    }
}

/// Where a module's parameters live. Three tiers, not two: the offload mode
/// keeps weights in host RAM and streams them into VRAM (plan A.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamTier {
    Vram,
    HostRam,
    Disk,
}

impl ParamTier {
    pub fn name(self) -> &'static str {
        match self {
            ParamTier::Vram => "vram",
            ParamTier::HostRam => "host",
            ParamTier::Disk => "disk",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModulePlacement {
    pub module: ImageModule,
    pub device: ImageDevice,
    pub tier: ParamTier,
}

/// The deployed assignment (`imagegen.toml:110-113`): the text encoder and VAE
/// on the CPU, the diffusion graph on device 0 with its weights in VRAM.
pub const DEFAULT_PLACEMENT: [ModulePlacement; 3] = [
    ModulePlacement {
        module: ImageModule::TextEncoder,
        device: ImageDevice::Cpu,
        tier: ParamTier::HostRam,
    },
    ModulePlacement {
        module: ImageModule::Diffusion,
        device: ImageDevice::Cuda(0),
        tier: ParamTier::Vram,
    },
    ModulePlacement {
        module: ImageModule::Vae,
        device: ImageDevice::Cpu,
        tier: ParamTier::HostRam,
    },
];

// ---------------------------------------------------------------------------
// Layout contracts
// ---------------------------------------------------------------------------

/// A pinned tensor: name, ggml dimensions (`ne`, fastest-varying first) and the
/// quantized type the layout pins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub typ: u32,
}

impl ImageTensor {
    fn new(name: String, dims: Vec<u64>, typ: u32) -> Self {
        Self { name, dims, typ }
    }

    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Bytes for the pinned type. Q6_K is 210 bytes per 256-element block
    /// (the block carries its own 2-byte scale); BF16 is two bytes per
    /// element. Blocks round up, as `ggml` types do.
    pub fn bytes(&self) -> u64 {
        match self.typ {
            TYPE_Q6_K => self.elements().div_ceil(256) * 210,
            TYPE_BF16 => self.elements() * 2,
            TYPE_F32 => self.elements() * 4,
            _ => 0,
        }
    }

    /// Matches a directory entry: same name, same rank, same dims, same type.
    pub fn matches(&self, t: &crate::tensors::TensorInfo) -> bool {
        self.typ == t.typ
            && self.dims.len() == t.ndim as usize
            && self.dims.iter().zip(t.dim.iter()).all(|(a, b)| a == b)
    }
}

/// The DiT layout. Q6_K binds to every matmul, BF16 to the boundary
/// projections and the per-head norms; that split is the artifact measured on
/// this host, not a preference.
pub fn dit_contract() -> Vec<ImageTensor> {
    let h = DIT_HIDDEN;
    let mut out = vec![
        ImageTensor::new("img_in.weight".into(), vec![DIT_IN_CHANNELS, h], TYPE_BF16),
        ImageTensor::new("txt_in.text_norm.weight".into(), vec![h], TYPE_BF16),
        ImageTensor::new("txt_in.in_layer.weight".into(), vec![h, h], TYPE_BF16),
        ImageTensor::new("txt_in.out_layer.weight".into(), vec![h, h], TYPE_BF16),
        ImageTensor::new(
            "modulation.1.weight".into(),
            vec![h, DIT_MODULATION],
            TYPE_Q6_K,
        ),
        ImageTensor::new(
            "time_text_embed.timestep_embedder.linear_1.weight".into(),
            vec![DIT_TIME_EMBED_DIM, h],
            TYPE_Q6_K,
        ),
        ImageTensor::new(
            "time_text_embed.timestep_embedder.linear_2.weight".into(),
            vec![h, h],
            TYPE_Q6_K,
        ),
        ImageTensor::new("norm_out.linear.weight".into(), vec![h, h], TYPE_Q6_K),
        ImageTensor::new("proj_out.weight".into(), vec![h, DIT_OUT_CHANNELS], TYPE_Q6_K),
    ];
    for il in 0..DIT_LAYERS {
        let b = format!("transformer_blocks.{il}");
        out.push(ImageTensor::new(
            format!("{b}.attn.norm_q.weight"),
            vec![DIT_HEAD_DIM],
            TYPE_BF16,
        ));
        out.push(ImageTensor::new(
            format!("{b}.attn.norm_k.weight"),
            vec![DIT_HEAD_DIM],
            TYPE_BF16,
        ));
        for suffix in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            out.push(ImageTensor::new(format!("{b}.{suffix}.weight"), vec![h, h], TYPE_Q6_K));
        }
        out.push(ImageTensor::new(
            format!("{b}.img_mlp.proj.weight"),
            vec![h, DIT_INTERMEDIATE],
            TYPE_Q6_K,
        ));
        out.push(ImageTensor::new(
            format!("{b}.img_mlp.gate_layer.weight"),
            vec![h, DIT_INTERMEDIATE],
            TYPE_Q6_K,
        ));
        out.push(ImageTensor::new(
            format!("{b}.img_mlp.out.weight"),
            vec![DIT_INTERMEDIATE, h],
            TYPE_Q6_K,
        ));
    }
    out
}

/// Conv3d weight in the layout the reference's conv kernel consumes:
/// `ne = {kW, kH, kT, IC*OC}` (`ggml_extend.cpp:452-475` derives `OC` from
/// `w->ne[3] / IC`). The exporter's torch shape is `[OC, IC, kT, kH, kW]`, and
/// every conv in the Qwen-Image-2.1 export has `kT == 1`.
pub fn conv3d_dims(kw: u64, kh: u64, kt: u64, ic: u64, oc: u64) -> Vec<u64> {
    vec![kw, kh, kt, ic * oc]
}

/// Conv2d weight in the ggml layout `{kW, kH, IC, OC}`; the exporter's torch
/// shape is `[OC, IC, kH, kW]` (the VAE's `resample` convs).
pub fn conv2d_dims(kw: u64, kh: u64, ic: u64, oc: u64) -> Vec<u64> {
    vec![kw, kh, ic, oc]
}

/// Decoder channel widths: `{dim_mult.back() * dim}` then `dim * dim_mult[i]`
/// in reverse (`wan_vae.hpp:851-854`), giving `{1152, 1152, 1152, 576, 288,
/// 144}` for this VAE.
pub fn vae_decoder_dims() -> Vec<u64> {
    let mut dims = vec![VAE_DIM_MULT[VAE_DIM_MULT.len() - 1] * VAE_DEC_DIM];
    for m in VAE_DIM_MULT.iter().rev() {
        dims.push(m * VAE_DEC_DIM);
    }
    dims
}

/// The decode-only VAE layout, generated from the config so the table cannot
/// drift from the reference's block structure.
pub fn vae_decode_contract() -> Vec<ImageTensor> {
    let dims = vae_decoder_dims();
    let mut out = Vec::new();
    let mut push = |name: String, d: Vec<u64>| {
        out.push(ImageTensor::new(name, d, TYPE_BF16));
    };

    // decoder.conv1: latent channels -> dims[0], spatial 3x3, singleton time.
    let d0 = dims[0];
    push("decoder.conv1.weight".into(), conv3d_dims(3, 3, 1, VAE_Z_DIM, d0));
    push("decoder.conv1.bias".into(), vec![d0]);

    // decoder.middle: ResidualBlock, AttentionBlock, ResidualBlock.
    for blk in ["0", "2"] {
        for slot in ["0", "3"] {
            push(format!("decoder.middle.{blk}.residual.{slot}.gamma"), vec![d0]);
        }
        for slot in ["2", "6"] {
            push(
                format!("decoder.middle.{blk}.residual.{slot}.weight"),
                conv3d_dims(3, 3, 1, d0, d0),
            );
            push(format!("decoder.middle.{blk}.residual.{slot}.bias"), vec![d0]);
        }
    }
    push("decoder.middle.1.norm.gamma".into(), vec![d0]);
    push(
        "decoder.middle.1.to_qkv.weight".into(),
        conv2d_dims(1, 1, d0, 3 * d0),
    );
    push("decoder.middle.1.to_qkv.bias".into(), vec![3 * d0]);
    push(
        "decoder.middle.1.proj.weight".into(),
        conv2d_dims(1, 1, d0, d0),
    );
    push("decoder.middle.1.proj.bias".into(), vec![d0]);

    // decoder.upsamples.{i}: three residual blocks, an optional channel
    // shortcut on the first, then the resample (and its time_conv where
    // temperal_upsample holds).
    for i in 0..dims.len() - 1 {
        let (cin, cout) = (dims[i], dims[i + 1]);
        let lvl = format!("decoder.upsamples.{i}");
        if cin != cout {
            push(format!("{lvl}.upsamples.0.shortcut.weight"), conv3d_dims(1, 1, 1, cin, cout));
            push(format!("{lvl}.upsamples.0.shortcut.bias"), vec![cout]);
        }
        for j in 0..3 {
            let (a, b) = if j == 0 { (cin, cout) } else { (cout, cout) };
            let r = format!("{lvl}.upsamples.{j}.residual");
            // ResidualBlock: RMS(cin), SiLU, conv(cin->cout), RMS(cout), SiLU,
            // conv(cout->cout). The leading norm spans the input width, which is
            // why it is the one channel-count exception in the block.
            push(format!("{r}.0.gamma"), vec![a]);
            push(format!("{r}.2.weight"), conv3d_dims(3, 3, 1, a, b));
            push(format!("{r}.2.bias"), vec![b]);
            push(format!("{r}.3.gamma"), vec![b]);
            push(format!("{r}.6.weight"), conv3d_dims(3, 3, 1, b, b));
            push(format!("{r}.6.bias"), vec![b]);
        }
        if i < VAE_TEMPORAL_UPSAMPLE.len() {
            push(
                format!("{lvl}.upsamples.3.resample.1.weight"),
                conv2d_dims(3, 3, cout, cout),
            );
            push(format!("{lvl}.upsamples.3.resample.1.bias"), vec![cout]);
            if VAE_TEMPORAL_UPSAMPLE[i] {
                // Singleton temporal kernel: this export carries no kT=3 conv
                // anywhere (measured: 62 weights at (1,3,3), 14 at (1,1,1)), and
                // the reference collapses the temporal kernel and its padding
                // for exactly that case (`wan_vae.hpp:30-34`).
                push(
                    format!("{lvl}.upsamples.3.time_conv.weight"),
                    conv3d_dims(1, 1, 1, cout, 2 * cout),
                );
                push(format!("{lvl}.upsamples.3.time_conv.bias"), vec![2 * cout]);
            }
        }
    }

    // decoder.head: RMS norm, SiLU, output conv to the 4 pixel channels.
    let last = dims[dims.len() - 1];
    push("decoder.head.0.gamma".into(), vec![last]);
    push(
        "decoder.head.2.weight".into(),
        conv3d_dims(3, 3, 1, last, VAE_OUT_CHANNELS),
    );
    push("decoder.head.2.bias".into(), vec![VAE_OUT_CHANNELS]);

    // conv2: the latent-width causal conv the decode path always ends on
    // (`wan_vae.hpp:1100-1105`, used at `:1220-1228`).
    push("conv2.weight".into(), conv3d_dims(1, 1, 1, VAE_Z_DIM, VAE_Z_DIM));
    push("conv2.bias".into(), vec![VAE_Z_DIM]);

    out
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ImageError {
    Gguf(GgufError),
    Tensor(TensorError),
    /// The artifact is not one of the pinned contracts.
    Contract {
        what: &'static str,
        problems: Vec<String>,
    },
    Unsupported(&'static str),
}

impl ImageError {
    pub fn token(&self) -> String {
        match self {
            ImageError::Gguf(e) => e.token(),
            ImageError::Tensor(e) => e.token(),
            ImageError::Contract { what, problems } => {
                format!("image-contract {what} {}", problems.join("; "))
            }
            ImageError::Unsupported(w) => format!("image-unsupported {w}"),
        }
    }
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for ImageError {}

impl From<GgufError> for ImageError {
    fn from(e: GgufError) -> Self {
        ImageError::Gguf(e)
    }
}

impl From<TensorError> for ImageError {
    fn from(e: TensorError) -> Self {
        ImageError::Tensor(e)
    }
}

/// Which of the two artifacts an identified file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageArtifactKind {
    DitQ6K,
    VaeDecodeBf16,
}

impl ImageArtifactKind {
    pub fn name(self) -> &'static str {
        match self {
            ImageArtifactKind::DitQ6K => "dit-q6_k",
            ImageArtifactKind::VaeDecodeBf16 => "vae-decode-bf16",
        }
    }
}

/// What identification established, in the shape the plan reports.
#[derive(Clone, Debug)]
pub struct ImageArtifactId {
    pub kind: ImageArtifactKind,
    pub tensors: usize,
    pub bytes: u64,
    /// Tensor count per GGML type id, so a re-quantized artifact is visible
    /// instead of silently accepted.
    pub quant_mix: Vec<(u32, usize)>,
}

impl ImageArtifactId {
    pub fn type_count(&self, typ: u32) -> usize {
        self.quant_mix.iter().find(|(t, _)| *t == typ).map(|(_, n)| *n).unwrap_or(0)
    }
}

/// Contract check against a directory: every pinned tensor present with the
/// pinned rank, dims and type, and nothing else. The reported problems are
/// capped so a wrong artifact fails with a readable reason.
const CONTRACT_PROBLEM_CAP: usize = 8;

fn check_contract(
    kind: ImageArtifactKind,
    contract: &[ImageTensor],
    inv: &TensorInventory,
) -> Result<ImageArtifactId, ImageError> {
    let mut problems = Vec::new();
    for want in contract {
        match inv.find(&want.name) {
            None => problems.push(format!("missing {}", want.name)),
            Some(got) if !want.matches(got) => problems.push(format!(
                "{} dims={:?} type={} expected dims={:?} type={}",
                want.name,
                &got.dim[..got.ndim as usize],
                crate::tensors::tensor_type_name(got.typ),
                want.dims,
                crate::tensors::tensor_type_name(want.typ),
            )),
            Some(_) => {}
        }
        if problems.len() >= CONTRACT_PROBLEM_CAP {
            break;
        }
    }
    if problems.is_empty() && inv.tensors.len() != contract.len() {
        problems.push(format!(
            "tensor count {} != {}",
            inv.tensors.len(),
            contract.len()
        ));
    }
    if !problems.is_empty() {
        return Err(ImageError::Contract { what: kind.name(), problems });
    }

    let mut mix: Vec<(u32, usize)> = Vec::new();
    for t in contract {
        match mix.iter_mut().find(|(q, _)| *q == t.typ) {
            Some((_, n)) => *n += 1,
            None => mix.push((t.typ, 1)),
        }
    }
    Ok(ImageArtifactId {
        kind,
        tensors: contract.len(),
        bytes: contract.iter().map(|t| t.bytes()).sum(),
        quant_mix: mix,
    })
}

/// Identifies the diffusion model. The artifact carries no
/// `general.architecture` (verified: zero metadata keys), so the tensor
/// directory is the only key there is.
pub fn identify_dit(path: &std::path::Path) -> Result<ImageArtifactId, ImageError> {
    let g = GgufFile::open(path)?;
    refuse_text_artifact(&g)?;
    let inv = TensorInventory::from_file(path, &g)?;
    check_contract(ImageArtifactKind::DitQ6K, &dit_contract(), &inv)
}

/// Identifies the converted decode-only VAE. Same rule: names, not metadata.
pub fn identify_vae(path: &std::path::Path) -> Result<ImageArtifactId, ImageError> {
    let g = GgufFile::open(path)?;
    refuse_text_artifact(&g)?;
    let inv = TensorInventory::from_file(path, &g)?;
    check_contract(ImageArtifactKind::VaeDecodeBf16, &vae_decode_contract(), &inv)
}

/// Refuses a text artifact early, by name, instead of routing it into the
/// autoregressive catalogue.
pub fn refuse_text_artifact(g: &GgufFile) -> Result<(), ImageError> {
    match g.get_string("general.architecture") {
        Some(a) => Err(ImageError::Unsupported(if a.starts_with(b"qwen_image") {
            "an image artifact carrying general.architecture is not one of the pinned layouts"
        } else {
            "a text artifact, not an image one"
        })),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Plan and refusals
// ---------------------------------------------------------------------------

/// An autoregressive serving control. The image engine refuses each by name:
/// there is no KV cache, no sequence batching, no speculation and no snapshot
/// to attach these to (`docs/serving-contract.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArControl {
    pub flag: &'static str,
    pub code: &'static str,
    pub why: &'static str,
}

pub const AR_CONTROLS: [ArControl; 11] = [
    ArControl {
        flag: "--ctx",
        code: "ar_ctx_unsupported",
        why: "the diffusion graph owns no KV cache; its token count follows the image size",
    },
    ArControl {
        flag: "--max-seqs",
        code: "ar_max_seqs_unsupported",
        why: "the runner asserts batch 1; there is no continuous batching",
    },
    ArControl {
        flag: "--prefix-reuse",
        code: "ar_prefix_reuse_unsupported",
        why: "there is no prompt/prefix cache to reuse",
    },
    ArControl {
        flag: "--mtp-mode",
        code: "ar_mtp_unsupported",
        why: "the artifact carries no draft head and the image engine has no speculation",
    },
    ArControl {
        // The server's flag for the drafter sidecar is `--mtp`; the AR plan's
        // field is `mtp_path`.
        flag: "--mtp",
        code: "ar_mtp_sidecar_unsupported",
        why: "no sidecar attaches to a diffusion graph",
    },
    ArControl {
        flag: "--mtp-draft",
        code: "ar_mtp_draft_unsupported",
        why: "no draft width exists for a diffusion graph",
    },
    ArControl {
        flag: "--mtp-margin",
        code: "ar_mtp_margin_unsupported",
        why: "there is no speculation to bound",
    },
    ArControl {
        flag: "--kv-disk-dir",
        code: "ar_kv_disk_unsupported",
        why: "there is no KV state to checkpoint",
    },
    ArControl {
        flag: "--kv-disk-space-mb",
        code: "ar_kv_disk_space_unsupported",
        why: "there is no KV state to checkpoint",
    },
    ArControl {
        flag: "--kv-cache-min-tokens",
        code: "ar_kv_min_tokens_unsupported",
        why: "there is no KV state to checkpoint",
    },
    ArControl {
        flag: "--cont-width",
        code: "ar_cont_width_unsupported",
        why: "there is no continuous-bank width",
    },
];

/// Looks a CLI flag up in the refusal table, with `--flag=value` accepted.
pub fn ar_control(flag: &str) -> Option<&'static ArControl> {
    let head = flag.split('=').next().unwrap_or(flag);
    AR_CONTROLS.iter().find(|c| c.flag == head)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLevel {
    Error,
    Warn,
}

#[derive(Clone, Debug)]
pub struct ImageIssue {
    pub level: ImageLevel,
    pub code: &'static str,
    pub message: String,
}

fn error(code: &'static str, message: String) -> ImageIssue {
    ImageIssue { level: ImageLevel::Error, code, message }
}

fn warn(code: &'static str, message: String) -> ImageIssue {
    ImageIssue { level: ImageLevel::Warn, code, message }
}

/// What the caller asked for.
#[derive(Clone, Debug, Default)]
pub struct ImageRequest {
    /// AR controls seen on the command line, by flag name.
    pub ar_controls: Vec<String>,
    /// Per-module assignment; empty means the engine default.
    pub placement: Vec<ModulePlacement>,
    /// The offload mode: weights in host RAM streamed into a fixed VRAM arena.
    pub offload: bool,
    /// The per-device VRAM budget in MiB (`--max-vram`), which is what the
    /// placement rules resolve against. `None` means the deployed card's own
    /// budget, so an existing invocation keeps the behaviour it had.
    pub vram_budget_mib: Option<u64>,
}

impl ImageRequest {
    /// The budget the placement resolves against: what the caller declared, else
    /// the deployed card's usable VRAM.
    fn budget_mib(&self) -> u64 {
        self.vram_budget_mib.unwrap_or(FOOTPRINT_DEFAULT_BUDGET_MIB)
    }
}

/// Device footprints in MiB, the unit the reference's own log prints as "MB"
/// (5876556448 bytes prints as 5604.32). Each one is a measured figure ROUNDED
/// UP, so a sum is an upper bound and a budget is never over-counted, and each
/// names the artifact or log line it came from.
///
/// The resolution matters and is not incidental: the DiT's compute buffer is
/// 34.66 MiB for a 256-square generation and 2317.45 MiB for the 1024-square one
/// whose figures are used here, because the rules these footprints drive were
/// measured at 1024 square ("cannot complete a 1024-square generation on a 12 GB
/// card"). A different qualified resolution needs its own buffer figure.
///
/// They are the budget's inputs, not engine invariants: a module's footprint is
/// the same everywhere, and what decides is the device's declared budget.
const FOOTPRINT_TEXT_ENCODER_MIB: u64 = 4303;
const FOOTPRINT_DIT_WEIGHTS_MIB: u64 = 5605;
const FOOTPRINT_DIT_COMPUTE_MIB: u64 = 2318;
/// The decode-only GGUF this engine loads, not the reference's safetensors: the
/// converted artifact holds 518096424 bytes of BF16 tensor data (494.10 MiB),
/// while the reference's own 128-tensor VAE reports 482.81 MiB.
const FOOTPRINT_VAE_WEIGHTS_MIB: u64 = 495;
/// What an untiled VAE decode needs BEYOND its own parameters. A measured
/// FLOOR, not a measurement of the decode: with the DiT pinned the measured
/// 7921 MiB of the card's 11894 leave 3973 MiB and the decode does not fit, so
/// it needs more than that (plan A.1).
const FOOTPRINT_VAE_DECODE_FLOOR_MIB: u64 = 3973;
/// The deployed card's usable VRAM, and therefore the default budget.
const FOOTPRINT_DEFAULT_BUDGET_MIB: u64 = 11894;

/// What one module's parameters and its runner buffer cost on a device.
fn module_footprint_mib(module: ImageModule) -> u64 {
    match module {
        ImageModule::TextEncoder => FOOTPRINT_TEXT_ENCODER_MIB,
        ImageModule::Diffusion => FOOTPRINT_DIT_WEIGHTS_MIB + FOOTPRINT_DIT_COMPUTE_MIB,
        ImageModule::Vae => FOOTPRINT_VAE_WEIGHTS_MIB + FOOTPRINT_VAE_DECODE_FLOOR_MIB,
    }
}

#[derive(Clone, Debug)]
pub struct ImageRequested {
    pub controls: Vec<String>,
    pub modules: Vec<ModulePlacement>,
    pub offload: bool,
}

#[derive(Clone, Debug)]
pub struct ImageEffective {
    pub modules: Vec<ModulePlacement>,
    pub offload: bool,
}

/// What the engine can guarantee today. This is the contract, not a
/// measurement: resolution ceilings, step time and VRAM peak are P6's ledger.
#[derive(Clone, Debug)]
pub struct ImageQualified {
    pub dit_tensors: usize,
    pub dit_q6k: usize,
    pub dit_bf16: usize,
    pub vae_tensors: usize,
    pub refusals: &'static [ArControl],
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct ImagePlan {
    pub engine: &'static str,
    pub requested: ImageRequested,
    pub effective: ImageEffective,
    pub qualified: ImageQualified,
    pub issues: Vec<ImageIssue>,
}

impl ImagePlan {
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.level == ImageLevel::Error)
    }

    /// The same gate the AR path uses: no Error issue may pass.
    pub fn may_load(&self) -> bool {
        !self.has_errors()
    }

    pub fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "engine: {}", self.engine);
        let _ = writeln!(
            s,
            "requested: modules={} offload={} controls={}",
            modules_str(&self.requested.modules),
            self.requested.offload,
            if self.requested.controls.is_empty() {
                "-".into()
            } else {
                self.requested.controls.join(",")
            }
        );
        let _ = writeln!(
            s,
            "effective: modules={} offload={}",
            modules_str(&self.effective.modules),
            self.effective.offload
        );
        let _ = writeln!(
            s,
            "qualified: dit_tensors={} dit_q6_k={} dit_bf16={} vae_tensors={} refusals={}",
            self.qualified.dit_tensors,
            self.qualified.dit_q6k,
            self.qualified.dit_bf16,
            self.qualified.vae_tensors,
            self.qualified.refusals.len()
        );
        if !self.qualified.note.is_empty() {
            let _ = writeln!(s, "note: {}", self.qualified.note);
        }
        for i in &self.issues {
            let tag = match i.level {
                ImageLevel::Error => "error",
                ImageLevel::Warn => "warn",
            };
            let _ = writeln!(s, "{tag}: {} ({})", i.message, i.code);
        }
        s
    }

    pub fn to_json(&self) -> String {
        let modules = |m: &[ModulePlacement]| {
            let mut v = Vec::new();
            for p in m {
                v.push(format!(
                    "{{\"module\":\"{}\",\"device\":\"{}\",\"tier\":\"{}\"}}",
                    p.module.name(),
                    p.device.name(),
                    p.tier.name()
                ));
            }
            format!("[{}]", v.join(","))
        };
        let issues: Vec<String> = self
            .issues
            .iter()
            .map(|i| {
                format!(
                    "{{\"level\":\"{}\",\"code\":\"{}\",\"message\":{}}}",
                    match i.level {
                        ImageLevel::Error => "error",
                        ImageLevel::Warn => "warn",
                    },
                    i.code,
                    json_string(&i.message)
                )
            })
            .collect();
        format!(
            "{{\"engine\":\"{}\",\"requested\":{{\"modules\":{},\"offload\":{},\"controls\":[{}]}},\"effective\":{{\"modules\":{},\"offload\":{}}},\"qualified\":{{\"dit_tensors\":{},\"dit_q6_k\":{},\"dit_bf16\":{},\"vae_tensors\":{},\"refusals\":[{}]}},\"issues\":[{}]}}",
            self.engine,
            modules(&self.requested.modules),
            self.requested.offload,
            self.requested
                .controls
                .iter()
                .map(|c| json_string(c))
                .collect::<Vec<_>>()
                .join(","),
            modules(&self.effective.modules),
            self.effective.offload,
            self.qualified.dit_tensors,
            self.qualified.dit_q6k,
            self.qualified.dit_bf16,
            self.qualified.vae_tensors,
            self.qualified
                .refusals
                .iter()
                .map(|r| json_string(r.flag))
                .collect::<Vec<_>>()
                .join(","),
            issues.join(",")
        )
    }
}

fn modules_str(m: &[ModulePlacement]) -> String {
    m.iter()
        .map(|p| format!("{}={},{}", p.module.name(), p.device.name(), p.tier.name()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Fills the engine default for every module the request left unassigned.
fn resolve_modules(req: &ImageRequest) -> Vec<ModulePlacement> {
    let mut out = Vec::new();
    for default in DEFAULT_PLACEMENT {
        let chosen = req
            .placement
            .iter()
            .find(|p| p.module == default.module)
            .copied()
            .unwrap_or(default);
        out.push(chosen);
    }
    out
}

/// Placement rules that follow from the device budget, not from the engine
/// (plan A.1).
///
/// The two configurations below were measured to fail on a 12 GB card, and they
/// still fail there because the footprint and the default budget are that card's
/// measurements. What they are not is an engine invariant: a device whose budget
/// is large enough may pin the text encoder, or the DiT and the VAE together,
/// which is how the whole stack becomes resident in VRAM on a DGX Spark.
fn check_placement(req: &ImageRequest, eff: &[ModulePlacement], issues: &mut Vec<ImageIssue>) {
    let mut devices: Vec<u32> = eff
        .iter()
        .filter(|p| p.tier == ParamTier::Vram)
        .filter_map(|p| match p.device {
            ImageDevice::Cuda(index) => Some(index),
            // A VRAM tier on a CPU graph is not a device pin; there is nothing
            // to budget.
            ImageDevice::Cpu => None,
        })
        .collect();
    devices.sort_unstable();
    devices.dedup();

    for device in devices {
        let here: Vec<&ModulePlacement> = eff
            .iter()
            .filter(|p| p.tier == ParamTier::Vram && p.device == ImageDevice::Cuda(device))
            .collect();
        let pinned_mib: u64 = here.iter().map(|p| module_footprint_mib(p.module)).sum();
        let budget_mib = req.budget_mib();
        if pinned_mib <= budget_mib {
            continue;
        }

        let pinned = |module: ImageModule| here.iter().any(|p| p.module == module);
        let over = format!(
            "pinning {} on cuda{device} needs about {pinned_mib} MiB and the budget is {budget_mib} MiB",
            modules_str(&here.iter().map(|p| **p).collect::<Vec<_>>())
        );
        let mut named = false;
        if pinned(ImageModule::TextEncoder) {
            issues.push(error(
                "image_te_vram_unsupported",
                format!(
                    "{over}: the text encoder's parameters leave too little room for a 1024-square \
                     generation (measured on the deployed 12 GB card, plan A.1). Raise --max-vram on \
                     a device with more memory, or keep te on host or disk"
                ),
            ));
            named = true;
        }
        if pinned(ImageModule::Diffusion) && pinned(ImageModule::Vae) {
            issues.push(error(
                "image_double_pin_unsupported",
                format!(
                    "{over}: the untiled VAE decode is left less than the {floor} MiB it needs beyond \
                     its own parameters (measured, plan A.1). Move the VAE to host, disk or the \
                     offload mode, or raise --max-vram on a device with more memory",
                    floor = FOOTPRINT_VAE_DECODE_FLOOR_MIB
                ),
            ));
            named = true;
        }
        if !named {
            issues.push(error(
                "image_vram_budget_exceeded",
                format!("{over}: lower the pins, or raise --max-vram on a device with more memory"),
            ));
        }
    }
    if req.offload {
        issues.push(warn(
            "image_offload_staged",
            "offload mode streams staged weights; it cost 8.0% of a step when the whole DiT \
             was staged (phase-S M3)"
                .into(),
        ));
    }
}

/// Resolves the image plan: the refusal set, the placement and the contract
/// summary. `ids` are the artifacts identification established, if any.
pub fn resolve_image_plan(req: &ImageRequest, ids: &[ImageArtifactId]) -> ImagePlan {
    let mut issues = Vec::new();
    let modules = resolve_modules(req);

    for flag in &req.ar_controls {
        match ar_control(flag) {
            Some(c) => issues.push(error(
                c.code,
                format!("{} is an autoregressive control: {}", c.flag, c.why),
            )),
            None => issues.push(error(
                "image_unknown_control",
                format!("{flag} is not an image-engine control"),
            )),
        }
    }
    check_placement(req, &modules, &mut issues);

    let dit_id = ids.iter().find(|i| i.kind == ImageArtifactKind::DitQ6K);
    let vae_id = ids.iter().find(|i| i.kind == ImageArtifactKind::VaeDecodeBf16);
    let (dit_q6k, dit_bf16) = match dit_id {
        Some(id) => (id.type_count(TYPE_Q6_K), id.type_count(TYPE_BF16)),
        None => (0, 0),
    };

    ImagePlan {
        engine: ENGINE,
        requested: ImageRequested {
            controls: req.ar_controls.clone(),
            modules: modules.clone(),
            offload: req.offload,
        },
        effective: ImageEffective {
            modules,
            offload: req.offload,
        },
        qualified: ImageQualified {
            dit_tensors: dit_id.map(|i| i.tensors).unwrap_or(0),
            dit_q6k,
            dit_bf16,
            vae_tensors: vae_id.map(|i| i.tensors).unwrap_or(0),
            refusals: &AR_CONTROLS,
            note: "contract only: step time, resolution ceiling and VRAM peak are P6's ledger"
                .into(),
        },
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dit_contract_is_297_tensors_with_the_measured_quant_split() {
        let c = dit_contract();
        assert_eq!(c.len(), DIT_TENSOR_COUNT);
        let q6k = c.iter().filter(|t| t.typ == TYPE_Q6_K).count();
        let bf16 = c.iter().filter(|t| t.typ == TYPE_BF16).count();
        assert_eq!(q6k, DIT_Q6K_COUNT);
        assert_eq!(bf16, DIT_BF16_COUNT);
        let mut names: Vec<&str> = c.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), c.len());
    }

    #[test]
    fn vae_contract_is_134_tensors() {
        let c = vae_decode_contract();
        assert_eq!(c.len(), VAE_DECODE_TENSOR_COUNT);
        assert!(c.iter().all(|t| t.typ == TYPE_BF16));
        let mut names: Vec<&str> = c.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), c.len());
    }

    #[test]
    fn vae_decoder_dims_follow_the_config() {
        assert_eq!(vae_decoder_dims(), vec![1152, 1152, 1152, 576, 288, 144]);
    }

    #[test]
    fn ar_controls_refuse_by_name() {
        let req = ImageRequest {
            ar_controls: AR_CONTROLS.iter().map(|c| c.flag.to_string()).collect(),
            placement: Vec::new(),
            offload: false,
            vram_budget_mib: None,
        };
        let plan = resolve_image_plan(&req, &[]);
        assert!(!plan.may_load());
        for c in AR_CONTROLS {
            assert!(
                plan.issues.iter().any(|i| i.code == c.code),
                "{} not refused",
                c.flag
            );
        }
        assert_eq!(plan.issues.len(), AR_CONTROLS.len());
    }

    #[test]
    fn placement_defaults_are_the_deployed_assignment() {
        let plan = resolve_image_plan(&ImageRequest::default(), &[]);
        assert!(plan.may_load());
        assert_eq!(plan.effective.modules, DEFAULT_PLACEMENT.to_vec());
    }

    #[test]
    fn double_pin_is_refused() {
        let req = ImageRequest {
            placement: vec![ModulePlacement {
                module: ImageModule::Vae,
                device: ImageDevice::Cuda(0),
                tier: ParamTier::Vram,
            }],
            ..Default::default()
        };
        let plan = resolve_image_plan(&req, &[]);
        assert!(plan.issues.iter().any(|i| i.code == "image_double_pin_unsupported"));
    }

    #[test]
    fn te_vram_is_refused() {
        let req = ImageRequest {
            placement: vec![ModulePlacement {
                module: ImageModule::TextEncoder,
                device: ImageDevice::Cuda(0),
                tier: ParamTier::Vram,
            }],
            ..Default::default()
        };
        let plan = resolve_image_plan(&req, &[]);
        assert!(plan.issues.iter().any(|i| i.code == "image_te_vram_unsupported"));
    }

    fn pinned(module: ImageModule) -> ModulePlacement {
        ModulePlacement { module, device: ImageDevice::Cuda(0), tier: ParamTier::Vram }
    }

    /// With no budget declared the deployed card's own is used, so both measured
    /// refusals stand exactly as they did before the budget existed.
    #[test]
    fn the_deployed_budget_still_refuses_both_measured_pins() {
        let te = resolve_image_plan(&ImageRequest { placement: vec![pinned(ImageModule::TextEncoder)], ..Default::default() }, &[]);
        assert!(te.issues.iter().any(|i| i.code == "image_te_vram_unsupported"), "{}", te.report());

        let vae = resolve_image_plan(&ImageRequest { placement: vec![pinned(ImageModule::Vae)], ..Default::default() }, &[]);
        assert!(vae.issues.iter().any(|i| i.code == "image_double_pin_unsupported"), "{}", vae.report());
    }

    /// A DGX-class budget makes the whole stack resident in VRAM. Nothing about
    /// the configuration changed, only the device's budget: that is what it means
    /// for the rule to be budget-derived rather than an engine invariant.
    #[test]
    fn a_large_budget_admits_the_whole_stack_pinned() {
        let req = ImageRequest {
            placement: vec![
                pinned(ImageModule::TextEncoder),
                pinned(ImageModule::Diffusion),
                pinned(ImageModule::Vae),
            ],
            vram_budget_mib: Some(140 * 1024),
            ..Default::default()
        };
        let plan = resolve_image_plan(&req, &[]);
        assert!(plan.may_load(), "{}", plan.report());
        assert_eq!(plan.effective.modules.len(), 3);
        assert_eq!(plan.effective.modules[0].tier, ParamTier::Vram);
    }

    /// The rule is arithmetic on the declared budget, so a mid budget refuses
    /// only what overflows it: here the DiT and the VAE together, not the VAE
    /// with the DiT moved off the device.
    #[test]
    fn a_mid_budget_refuses_only_what_overflows_it() {
        let on_host = ModulePlacement {
            module: ImageModule::Diffusion,
            device: ImageDevice::Cuda(0),
            tier: ParamTier::HostRam,
        };

        let double = ImageRequest {
            placement: vec![pinned(ImageModule::Diffusion), pinned(ImageModule::Vae)],
            vram_budget_mib: Some(9000),
            ..Default::default()
        };
        let plan = resolve_image_plan(&double, &[]);
        assert!(plan.issues.iter().any(|i| i.code == "image_double_pin_unsupported"), "{}", plan.report());

        let vae_only = ImageRequest {
            placement: vec![on_host, pinned(ImageModule::Vae)],
            vram_budget_mib: Some(9000),
            ..Default::default()
        };
        let plan = resolve_image_plan(&vae_only, &[]);
        assert!(plan.may_load(), "{}", plan.report());

        // Below the VAE's own footprint the refusal is the generic budget one.
        let tiny = ImageRequest {
            placement: vec![on_host, pinned(ImageModule::Vae)],
            vram_budget_mib: Some(1000),
            ..Default::default()
        };
        let plan = resolve_image_plan(&tiny, &[]);
        assert!(plan.issues.iter().any(|i| i.code == "image_vram_budget_exceeded"), "{}", plan.report());
    }

    #[test]
    fn plan_report_and_json_name_the_engine() {
        let plan = resolve_image_plan(&ImageRequest::default(), &[]);
        assert!(plan.report().contains("engine: qwen-image-2.1"));
        assert!(plan.to_json().contains("\"engine\":\"qwen-image-2.1\""));
        assert!(plan.to_json().contains("\"module\":\"diffusion\""));
    }

    #[test]
    fn contract_carries_latent_statistics() {
        assert_eq!(VAE_LATENT_MEAN.len(), 64);
        assert_eq!(VAE_LATENT_STD.len(), 64);
        assert_eq!(VAE_SCALE_FACTOR, 1.0);
        assert_eq!(VAE_LATENT_MEAN[0], 0.5126);
        assert_eq!(VAE_LATENT_STD[63], 3.8161);
    }
}
