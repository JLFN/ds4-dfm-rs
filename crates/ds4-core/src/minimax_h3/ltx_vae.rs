//! The LTX-2.5 video VAE decoder: the release's `ConvVideoDecoder`, ported.
//!
//! Source of truth: LTX's own `ltx-core/src/ltx_core/model/video_vae/`, read
//! first-hand at `/data/ltx2-pin/packages` (the pin the release runs,
//! `d151147788a9284cca791edc6ce898007e727fe6` = tag v1.2.0), and the artifact's
//! own header (`ltx-2.5-video-vae-conv-bf16.safetensors`, 1,452,269,922 B,
//! `__metadata__.config.vae`). The release reaches it through
//! `runtime/stage2_ops/models.py:116-129`: `compat.VideoDecoder(path, ...)` and
//! `_decoder_builder.build(...)`, refusing any type but `ConvVideoDecoder`.
//!
//! The graph (`conv_video_decoder.py:176-357`):
//!
//! ```text
//! latent (1, 128, f, h, w) BF16
//!   -> per_channel_statistics.un_normalize   (ops.py:74-78)
//!   -> conv_in  128 -> 1024, 3x3x3           (:209-218)
//!   -> nine up_blocks (:222-238), the config's own list reversed:
//!        res_x @1024, compress_all x2 @512, res_x @512, compress_all x1,
//!        res_x @512, compress_time x2 @256, res_x @256, compress_space x2
//!        @128, res_x @128
//!   -> conv_norm_out (PixelNorm) -> SiLU -> conv_out 128 -> 48  (:240-254)
//!   -> unpatchify 4x4 (48 -> 3)                                 (:351-357)
//!   -> to_rgb: (x + 1) / 2 clamped, [c f h w] -> [f h w c]      (:497-499)
//! ```
//!
//! Four facts this port must not lose:
//!
//! - **Every convolution is `CausalConv3d` with `causal=False`**
//!   (`convolution.py:302-312`): one *replicated* frame each side in time and
//!   one *zero* frame each side in space (the artifact's
//!   `spatial_padding_mode: zeros`, `:299`). Replicating the spatial halo
//!   instead would halve the error at the borders and still be wrong.
//! - **Torch rounds BF16 elementwise ops one at a time.** `PixelNorm` is five
//!   ops and five roundings (`normalization.py:31-38`), `un_normalize` is two
//!   (`ops.py:75-78`), `to_rgb` is two (`conv_video_decoder.py:498-499`). The
//!   kernels round where the expression does; a fused single-rounding form is
//!   a different function.
//! - **The upsamplers drop the first output frame of every temporal doubling**
//!   (`sampling.py:119-120`), which is what turns `f` latents into `8(f-1)+1`
//!   frames.
//! - **`unpatchify` reads channel `c*p^2 + r*p + q` for row `h*p + q` and
//!   column `w*p + r`** (`ops.py:56-58`), with `p = patch_size_t = 1`.
//!
//! Named deviations, both measured in `docs/minimax-h3/ltx-vae-decoder-*.md`:
//!
//! - **The decode is untiled.** The release decodes through the explicit tile
//!   plan `TILES` (`models.py:20-22`: 128/24 frames, 448/64 and 768/64 pixels)
//!   resolved by `_prepare_tiles`/`tiled_decode`
//!   (`conv_video_decoder.py:359-482`), which at `(121, 768, 1344)` is four
//!   tiles blended by trapezoidal masks. This port computes the same
//!   convolution stack over the whole latent: the arithmetic inside every tile
//!   is identical, the values in the overlap zones are not (they are a
//!   weighted average there), and the peak footprint is the whole frame's.
//! - **The release's memory-efficient decode is not ported**
//!   (`memory_efficient_decode.py`, `MEMORY_EFFICIENT_DECODE` +
//!   `CHANNELS_LAST_3D_WEIGHTS` in `blocks.py:1103-1110`). It is a layout and
//!   in-place policy over the same expression, so values may differ in the
//!   last ulp; the port keeps the reference's own channel-major expression.

use std::fmt;
use std::path::Path;

use ds4_sys::h3_dit::{Dtype, Gpu, Tensor};

use super::safetensors::{self, Dtype as ShardDtype, Shard};

/// The artifact's `latent_channels`.
pub const LATENT_CHANNELS: u32 = 128;
/// The artifact's `decoder_base_channels`.
pub const BASE_CHANNELS: u32 = 128;
/// The artifact's `out_channels`: the decoder's RGB output.
pub const PIXEL_CHANNELS: u32 = 3;
/// The artifact's `patch_size`, the final 4x4 spatial expansion.
pub const PATCH_SIZE: u32 = 4;
/// `PixelNorm`'s epsilon (`common/normalization.py:23`).
pub const PIXEL_NORM_EPSILON: f32 = 1e-8;
/// Every convolution's kernel (`make_conv_nd(..., kernel_size=3, padding=1)`).
const KERNEL: [u32; 3] = [3, 3, 3];
/// The spatial and temporal halo, `1` on every axis of a 3x3x3.
const PADDING: [u32; 3] = [1, 1, 1];

/// One entry of the artifact's `decoder_blocks`, in the order the config
/// writes them (`__metadata__.config.vae`). The decoder builds them reversed
/// (`conv_video_decoder.py:221`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// `res_x`: a `UNetMidBlock3D` of that many resnets at the current width.
    Res { layers: usize },
    /// `compress_time` / `compress_space` / `compress_all`: one
    /// `DepthToSpaceUpsample` with this stride and channel reduction
    /// (`sampling.py:66-90`).
    Upsample { stride: [u32; 3], reduction: u32 },
}

/// The artifact's own list, read out of its header (12 tensors per res block
/// group, 2 per upsampler). `compress_*`'s `multiplier` is the reduction.
pub const DECODER_BLOCKS: [Block; 9] = [
    Block::Res { layers: 4 },
    Block::Upsample {
        stride: [1, 2, 2],
        reduction: 2,
    },
    Block::Res { layers: 6 },
    Block::Upsample {
        stride: [2, 1, 1],
        reduction: 2,
    },
    Block::Res { layers: 4 },
    Block::Upsample {
        stride: [2, 2, 2],
        reduction: 1,
    },
    Block::Res { layers: 2 },
    Block::Upsample {
        stride: [2, 2, 2],
        reduction: 2,
    },
    Block::Res { layers: 2 },
];

/// `SpatioTemporalScaleFactors::from_blocks` (`types.py:41-57`): one halving
/// per `compress_*` on the axis it touches, and the patch size on both spatial
/// axes.
pub fn scale_factors() -> [u32; 3] {
    let (mut spatial, mut temporal) = (0u32, 0u32);
    for block in DECODER_BLOCKS.iter() {
        if let Block::Upsample { stride, .. } = block {
            if stride[1] == 2 || stride[2] == 2 {
                spatial += 1;
            }
            if stride[0] == 2 {
                temporal += 1;
            }
        }
    }

    [
        2u32.pow(temporal),
        PATCH_SIZE * 2u32.pow(spatial),
        PATCH_SIZE * 2u32.pow(spatial),
    ]
}

/// `_decoder_bottleneck_channels` (`conv_video_decoder.py:43-58`): the width at
/// the latent end, `base_channels` times every upsampler's reduction.
pub fn bottleneck_channels() -> u32 {
    let mut multiplier = 1u32;
    for block in DECODER_BLOCKS.iter() {
        if let Block::Upsample { reduction, .. } = block {
            multiplier *= reduction;
        }
    }

    BASE_CHANNELS * multiplier
}

/// A channel-major `[channel][frame][height][width]` activation, which is the
/// layout every kernel in this module reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub channels: u32,
    pub frames: u32,
    pub height: u32,
    pub width: u32,
}

impl Shape {
    /// The elements one channel holds: the channel stride, and the position
    /// count a normalization reduces over.
    fn plane(&self) -> usize {
        self.frames as usize * self.height as usize * self.width as usize
    }

    /// The elements the whole activation holds: what a reader sizes a buffer
    /// with, and what a gate compares two tensors by.
    pub fn elements(&self) -> usize {
        self.channels as usize * self.plane()
    }

    /// The shape one upsampler of this block turns this shape into
    /// (`sampling.py:111-120`): each stride factor multiplies its axis, the
    /// temporal one drops a frame, and the channels divide by the reduction.
    fn upsampled(&self, stride: [u32; 3], reduction: u32) -> Shape {
        Shape {
            channels: self.channels / reduction,
            frames: self.frames * stride[0] - u32::from(stride[0] == 2),
            height: self.height * stride[1],
            width: self.width * stride[2],
        }
    }

    /// The shape of the convolution that precedes an upsampler: the same
    /// spatial extent, `prod(stride)` times as many channels.
    fn conv_output(&self, stride: [u32; 3], reduction: u32) -> Shape {
        Shape {
            channels: self.channels * stride[0] * stride[1] * stride[2] / reduction,
            ..*self
        }
    }
}

/// `VideoLatentShape::from_pixel_shape` (`types.py:110-125`): the latent grid a
/// `(frames, height, width)` pixel target decodes from.
pub fn latent_shape(pixel: [u32; 3]) -> [u32; 3] {
    let factors = scale_factors();

    [
        (pixel[0] - 1) / factors[0] + 1,
        pixel[1] / factors[1],
        pixel[2] / factors[2],
    ]
}

/// `VideoLatentShape::upscale` (`types.py:127-134`): the pixel target a latent
/// grid decodes to, `8(f - 1) + 1` frames.
pub fn pixel_shape(latent: [u32; 3]) -> [u32; 3] {
    let factors = scale_factors();

    [
        (latent[0] - 1) * factors[0] + 1,
        latent[1] * factors[1],
        latent[2] * factors[2],
    ]
}

/// The release's own target: `VideoPixelShape(batch=1, frames=121, height=768,
/// width=1344, fps=24)` (`refiner_encoder_ablation_single_gpu.py:FRAME_COUNT`,
/// `WIDTH`, `HEIGHT`; `models.py:20-22`).
pub const RELEASE_PIXELS: [u32; 3] = [121, 768, 1344];
/// The latent that target decodes from, which is the shape the refiner's stage
/// 2 emits (`ltx_refiner::VIDEO_TOKENS` = 16 x 24 x 42).
pub const RELEASE_LATENT: [u32; 3] = [16, 24, 42];

#[derive(Debug)]
pub enum VaeError {
    Io(String),
    Missing(String),
    Shape(String),
    Reader(String),
    Run(String),
}

impl VaeError {
    fn run(label: &str) -> impl Fn(String) -> VaeError + '_ {
        move |detail| VaeError::Run(format!("{label}: {detail}"))
    }
}

impl fmt::Display for VaeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VaeError::Io(detail) => write!(formatter, "vae io: {detail}"),
            VaeError::Missing(name) => write!(formatter, "vae checkpoint has no {name}"),
            VaeError::Shape(detail) => write!(formatter, "vae shape: {detail}"),
            VaeError::Reader(detail) => write!(formatter, "vae reader: {detail}"),
            VaeError::Run(detail) => write!(formatter, "vae run: {detail}"),
        }
    }
}

impl std::error::Error for VaeError {}

/// One convolution, in the checkpoint's own `[out][in][kd][kh][kw]` order plus
/// the `[kt*kh + ky][out][s*in + c]` pack the planar path's matmuls read
/// (`h3b_weight_pack`; same values, different order).
struct Conv {
    packed: Tensor,
    bias: Tensor,
    input_channels: u32,
    output_channels: u32,
}

/// A resnet block (`resnet.py:12-186`) at a constant width: `PixelNorm`, SiLU,
/// a 3x3x3, `PixelNorm`, SiLU, a 3x3x3, added to the input. No noise injection
/// and no timestep conditioning in this artifact (`inject_noise` and
/// `timestep_conditioning` are absent from its block dicts).
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
}

enum UpBlock {
    Res(Vec<ResBlock>),
    Upsample {
        conv: Conv,
        stride: [u32; 3],
        reduction: u32,
    },
}

/// The decoder's buffers, sized once from the release's geometry so the
/// untiled peak is a number the gate can report rather than a surprise. Every
/// activation is BF16 and channel-major; only the convolution accumulator is
/// F32, and it holds one frame.
struct Scratch {
    /// The graph's state and its two working buffers.
    state: Tensor,
    aux: Tensor,
    work: Tensor,
    /// One convolution's channel-major output before the pixel shuffle.
    conv_out: Tensor,
    /// The planar path's padded activation, `kw` copies wide.
    planar: Tensor,
    /// One convolution frame's F32 accumulator.
    temp: Tensor,
    /// The decoded RGB, `[frame][height][width][channel]` BF16.
    rgb: Tensor,
}

impl Scratch {
    fn new(gpu: &Gpu, latent: Shape) -> Result<Scratch, VaeError> {
        let max_activation = largest_activation(latent);
        let max_channels = largest_conv_output(latent);
        let planar = largest_planar(latent);
        let temp = largest_temp(latent);

        Ok(Scratch {
            state: gpu
                .alloc(Dtype::Bf16, max_activation)
                .map_err(VaeError::run("state"))?,
            aux: gpu
                .alloc(Dtype::Bf16, max_activation)
                .map_err(VaeError::run("aux"))?,
            work: gpu
                .alloc(Dtype::Bf16, max_activation)
                .map_err(VaeError::run("work"))?,
            conv_out: gpu
                .alloc(Dtype::Bf16, max_channels)
                .map_err(VaeError::run("conv_out"))?,
            planar: gpu
                .alloc(Dtype::Bf16, planar)
                .map_err(VaeError::run("planar"))?,
            temp: gpu.alloc(Dtype::F32, temp).map_err(VaeError::run("temp"))?,
            rgb: gpu
                .alloc(Dtype::Bf16, pixel_elements(latent))
                .map_err(VaeError::run("rgb"))?,
        })
    }

    fn bytes(&self) -> u64 {
        [
            &self.state,
            &self.aux,
            &self.work,
            &self.conv_out,
            &self.planar,
        ]
        .iter()
        .map(|tensor| tensor.bytes() as u64)
        .sum::<u64>()
            + self.temp.bytes() as u64
            + self.rgb.bytes() as u64
    }
}

/// The release's own latent geometry, for a caller that sizes or reports the
/// untiled plan without a checkpoint in hand.
pub fn release_latent() -> Shape {
    Shape {
        channels: LATENT_CHANNELS,
        frames: RELEASE_LATENT[0],
        height: RELEASE_LATENT[1],
        width: RELEASE_LATENT[2],
    }
}

/// The largest activation the graph carries: the state after the last
/// upsampler, which is `base_channels` at the full latent-time resolution.
fn largest_activation(latent: Shape) -> usize {
    let mut shape = Shape {
        channels: bottleneck_channels(),
        ..latent
    };
    let mut largest = shape.elements();

    for block in DECODER_BLOCKS.iter().rev() {
        if let Block::Upsample {
            stride, reduction, ..
        } = block
        {
            shape = shape.upsampled(*stride, *reduction);
            largest = largest.max(shape.elements());
        }
    }

    largest.max(latent.elements())
}

/// The largest buffer one convolution writes before its pixel shuffle: the
/// `compress_all` blocks carry eight times the input channels for one frame.
fn largest_conv_output(latent: Shape) -> usize {
    let mut shape = Shape {
        channels: bottleneck_channels(),
        ..latent
    };
    let mut largest = 0usize;

    for block in DECODER_BLOCKS.iter().rev() {
        if let Block::Upsample {
            stride, reduction, ..
        } = block
        {
            largest = largest.max(shape.conv_output(*stride, *reduction).elements());
            shape = shape.upsampled(*stride, *reduction);
        }
    }

    largest
}

/// The planar buffer's own element count for a convolution that reads
/// `(channels, frames, height, ·)`: the padded activation in channel-last form,
/// once per column tap (`h3b_planar`).
fn planar_elements(channels: u32, frames: u32, height: u32, out_width: u32) -> usize {
    channels as usize
        * (frames + KERNEL[0] - 1) as usize
        * (height + KERNEL[1] - 1) as usize
        * out_width as usize
        * KERNEL[2] as usize
}

fn padded_elements(channels: u32, frames: u32, height: u32, width: u32) -> usize {
    channels as usize * (frames + 2) as usize * (height + 2) as usize * (width + 2) as usize
}

/// The largest planar buffer the graph needs: the widest convolution's input,
/// which at the release geometry is the last resnet group's 128 channels at
/// `121 x 192 x 336`.
fn largest_planar(latent: Shape) -> usize {
    let mut shape = Shape {
        channels: bottleneck_channels(),
        ..latent
    };
    let mut largest = planar_elements(shape.channels, shape.frames, shape.height, shape.width).max(
        padded_elements(shape.channels, shape.frames, shape.height, shape.width),
    );

    for block in DECODER_BLOCKS.iter().rev() {
        if let Block::Upsample {
            stride, reduction, ..
        } = block
        {
            let conv = shape.conv_output(*stride, *reduction);
            largest = largest.max(planar_elements(
                conv.channels,
                conv.frames,
                conv.height,
                conv.width,
            ));
            shape = shape.upsampled(*stride, *reduction);
            largest = largest.max(planar_elements(
                shape.channels,
                shape.frames,
                shape.height,
                shape.width,
            ));
        }
    }

    largest
}

/// The widest convolution's per-frame F32 accumulator: `out_channels` rows of
/// one output plane.
fn largest_temp(latent: Shape) -> usize {
    let mut shape = Shape {
        channels: bottleneck_channels(),
        ..latent
    };
    let mut largest = shape.channels as usize * shape.height as usize * shape.width as usize;

    for block in DECODER_BLOCKS.iter().rev() {
        if let Block::Upsample {
            stride, reduction, ..
        } = block
        {
            let conv = shape.conv_output(*stride, *reduction);
            largest =
                largest.max(conv.channels as usize * conv.height as usize * conv.width as usize);
            shape = shape.upsampled(*stride, *reduction);
            largest =
                largest.max(shape.channels as usize * shape.height as usize * shape.width as usize);
        }
    }

    largest
}

fn pixel_elements(latent: Shape) -> usize {
    let pixel = pixel_shape([latent.frames, latent.height, latent.width]);

    PIXEL_CHANNELS as usize * pixel[0] as usize * pixel[1] as usize * pixel[2] as usize
}

/// What the untiled decode costs, buffer by buffer, in bytes. This is the
/// deviation's own price and it is computable without a device, which is how
/// the release geometry gets a number on a host that cannot hold it.
///
/// Three activation buffers (the state, the resnet's normed input and the
/// convolution's own output), one planar buffer per convolution and the RGB
/// frame are all resident at once; the F32 accumulator holds one frame.
pub fn scratch_plan(latent: Shape) -> [(&'static str, u64); 7] {
    let activation = largest_activation(latent) as u64 * 2;

    [
        ("state", activation),
        ("aux", activation),
        ("work", activation),
        ("conv_out", largest_conv_output(latent) as u64 * 2),
        ("planar", largest_planar(latent) as u64 * 2),
        ("temp", largest_temp(latent) as u64 * 4),
        ("rgb", pixel_elements(latent) as u64 * 2),
    ]
}

/// The release's video VAE decoder, its weights and its working set.
pub struct LtxVae {
    conv_in: Conv,
    up_blocks: Vec<UpBlock>,
    conv_out: Conv,
    /// `per_channel_statistics.{mean,std}-of-means` (`ops.py:70-79`).
    mean: Tensor,
    std: Tensor,
    scratch: Scratch,
    tensors: usize,
    bytes: u64,
}

impl LtxVae {
    /// Load the artifact's 84 decoder tensors and its two per-channel
    /// statistics against the plan its own config describes, sizing the
    /// working set for `latent`, the shape the caller will decode.
    pub fn open(gpu: &Gpu, path: &Path, latent: Shape) -> Result<LtxVae, VaeError> {
        let shard = safetensors::read_shard(path)
            .map_err(|error| VaeError::Io(format!("{}: {error:?}", path.display())))?;
        assert_config(&shard)?;

        let mut reader = Reader {
            gpu,
            buffer: Vec::new(),
            tensors: 0,
            bytes: 0,
        };

        let mut width = bottleneck_channels();
        let conv_in = reader.conv(&shard, "decoder.conv_in.conv", LATENT_CHANNELS, width)?;
        let mut up_blocks = Vec::with_capacity(DECODER_BLOCKS.len());

        for (index, block) in DECODER_BLOCKS.iter().rev().enumerate() {
            let name = format!("decoder.up_blocks.{index}");

            match block {
                Block::Res { layers } => {
                    let mut blocks = Vec::with_capacity(*layers);
                    for layer in 0..*layers {
                        blocks.push(ResBlock {
                            conv1: reader.conv(
                                &shard,
                                &format!("{name}.res_blocks.{layer}.conv1.conv"),
                                width,
                                width,
                            )?,
                            conv2: reader.conv(
                                &shard,
                                &format!("{name}.res_blocks.{layer}.conv2.conv"),
                                width,
                                width,
                            )?,
                        });
                    }
                    up_blocks.push(UpBlock::Res(blocks));
                }
                Block::Upsample {
                    stride, reduction, ..
                } => {
                    let channels = width * stride[0] * stride[1] * stride[2] / reduction;
                    up_blocks.push(UpBlock::Upsample {
                        conv: reader.conv(&shard, &format!("{name}.conv.conv"), width, channels)?,
                        stride: *stride,
                        reduction: *reduction,
                    });
                    width /= reduction;
                }
            }
        }

        let conv_out = reader.conv(
            &shard,
            "decoder.conv_out.conv",
            width,
            PIXEL_CHANNELS * PATCH_SIZE * PATCH_SIZE,
        )?;
        if width != BASE_CHANNELS {
            return Err(VaeError::Shape(format!(
                "the block plan ends at {width} channels, not {BASE_CHANNELS}"
            )));
        }

        // The statistics describe the latent, so they are `LATENT_CHANNELS`
        // wide, not the bottleneck's.
        let mean = reader.pull(
            &shard,
            "per_channel_statistics.mean-of-means",
            &[LATENT_CHANNELS as usize],
        )?;
        let std = reader.pull(
            &shard,
            "per_channel_statistics.std-of-means",
            &[LATENT_CHANNELS as usize],
        )?;

        if latent.channels != LATENT_CHANNELS {
            return Err(VaeError::Shape(format!(
                "a latent carries {LATENT_CHANNELS} channels, not {}",
                latent.channels
            )));
        }
        let scratch = Scratch::new(gpu, latent)?;

        Ok(LtxVae {
            conv_in,
            up_blocks,
            conv_out,
            mean,
            std,
            scratch,
            tensors: reader.tensors,
            bytes: reader.bytes,
        })
    }

    /// The tensors the load read and their byte count, for the gate's report.
    pub fn loaded(&self) -> (usize, u64) {
        (self.tensors, self.bytes)
    }

    /// The working set's own size: the untiled peak the deviation costs.
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch.bytes()
    }

    /// `ConvVideoDecoder.forward` (`conv_video_decoder.py:263-357`) over the
    /// whole latent: the graph above, one frame's convolution at a time.
    ///
    /// `latent` is `[channel][frame][height][width]` BF16, which is the layout
    /// the refiner's `latent_grid` emits. The decoded frames stay in this
    /// decoder's own buffer and come back as `[frame][height][width][channel]`
    /// in `[0, 1]`.
    pub fn decode(&self, gpu: &Gpu, latent: &Tensor, shape: Shape) -> Result<&Tensor, VaeError> {
        let expected = pixel_elements(shape);
        if expected > self.scratch.rgb.elements() {
            return Err(VaeError::Shape(format!(
                "the decode's {} values do not fit the {} this decoder was sized for",
                expected,
                self.scratch.rgb.elements()
            )));
        }

        let scratch = &self.scratch;
        // `sample.to(weights_dtype)` and `per_channel_statistics.un_normalize`
        // (`conv_video_decoder.py:284`, `:301`).
        gpu.video_vae_denormalize(
            &scratch.state,
            latent,
            &self.mean,
            &self.std,
            Dtype::Bf16,
            shape.channels,
            shape.plane(),
        )
        .map_err(VaeError::run("denormalize"))?;

        // `conv_in` widens the latents to the bottleneck (`:307`).
        let mut state = Shape {
            channels: bottleneck_channels(),
            ..shape
        };
        self.conv(gpu, &scratch.work, &scratch.state, &self.conv_in, shape)?;
        copy(gpu, &scratch.state, &scratch.work, state.elements())?;

        // The nine up blocks in the config's reversed order, then the output
        // head (`conv_video_decoder.py:222-238`, `:328-351`).
        for block in self.up_blocks.iter() {
            match block {
                UpBlock::Res(blocks) => {
                    for resnet in blocks {
                        self.resnet(gpu, state, resnet)?;
                    }
                }
                UpBlock::Upsample {
                    conv,
                    stride,
                    reduction,
                } => {
                    self.conv(gpu, &scratch.conv_out, &scratch.state, conv, state)?;
                    // The shuffle writes straight over the state: the
                    // convolution has already read it, and the reference's
                    // `x = up_block(x)` needs no copy of its own.
                    let next = state.upsampled(*stride, *reduction);
                    gpu.video_vae_depth_to_space(
                        &scratch.state,
                        &scratch.conv_out,
                        Dtype::Bf16,
                        state.conv_output(*stride, *reduction).channels,
                        state.frames,
                        state.height,
                        state.width,
                        stride[0],
                        stride[1],
                        stride[2],
                        u32::from(stride[0] == 2),
                    )
                    .map_err(VaeError::run("depth to space"))?;
                    state = next;
                }
            }
        }

        // `conv_norm_out` (a `PixelNorm` with no parameters), `conv_act`, and
        // `conv_out` (`conv_video_decoder.py:328`, `:349-351`).
        pixel_norm(gpu, &scratch.work, &scratch.state, state)?;
        silu(gpu, &scratch.work)?;
        self.conv(gpu, &scratch.conv_out, &scratch.work, &self.conv_out, state)?;

        let patchified = Shape {
            channels: PIXEL_CHANNELS * PATCH_SIZE * PATCH_SIZE,
            ..state
        };
        gpu.video_vae_unpatchify(
            &scratch.work,
            &scratch.conv_out,
            Dtype::Bf16,
            PIXEL_CHANNELS,
            PATCH_SIZE,
            patchified.frames,
            patchified.height,
            patchified.width,
        )
        .map_err(VaeError::run("unpatchify"))?;

        // The unpatchify already applied the last `patch_size` of spatial
        // expansion (`:351-357`), so the state is at the pixel resolution over
        // the patch size at this point - not `pixel_shape`'s, which is the
        // latent's own upscale.
        let pixel = [
            state.frames,
            state.height * PATCH_SIZE,
            state.width * PATCH_SIZE,
        ];
        gpu.video_vae_to_rgb(
            &scratch.rgb,
            &scratch.work,
            Dtype::Bf16,
            pixel[0],
            pixel[1],
            pixel[2],
        )
        .map_err(VaeError::run("to rgb"))?;

        Ok(&scratch.rgb)
    }

    /// One `ResnetBlock3D` (`resnet.py:122-186`) at a constant shape.
    fn resnet(&self, gpu: &Gpu, shape: Shape, resnet: &ResBlock) -> Result<(), VaeError> {
        let scratch = &self.scratch;

        pixel_norm(gpu, &scratch.aux, &scratch.state, shape)?;
        silu(gpu, &scratch.aux)?;
        self.conv(gpu, &scratch.work, &scratch.aux, &resnet.conv1, shape)?;
        pixel_norm(gpu, &scratch.aux, &scratch.work, shape)?;
        silu(gpu, &scratch.aux)?;
        self.conv(gpu, &scratch.work, &scratch.aux, &resnet.conv2, shape)?;
        add(gpu, &scratch.state, &scratch.work, shape.elements())
    }

    /// One convolution (`h3b_planar` + one dense matmul per `(kt, ky)` tap),
    /// with the artifact's padding: 3x3x3, stride 1, one zero frame of spatial
    /// halo and one replicated frame each side in time.
    fn conv(
        &self,
        gpu: &Gpu,
        out: &Tensor,
        input: &Tensor,
        conv: &Conv,
        shape: Shape,
    ) -> Result<(), VaeError> {
        let scratch = &self.scratch;
        let out_width = shape.width + 2 * PADDING[2] - KERNEL[2] + 1;
        if out_width != shape.width {
            return Err(VaeError::Shape(format!(
                "the planar path is a same-shape convolution only, {out_width} != {}",
                shape.width
            )));
        }

        // One frame's output plane: the A window `h3b_matmul_planar` takes is
        // one image of the padded buffer, so `rows` is a frame and the
        // epilogue places each frame inside the channel-major output.
        let out_plane = shape.height as usize * shape.width as usize;
        let columns = conv.input_channels as usize * KERNEL[2] as usize;
        let padded_height = shape.height + 2 * PADDING[1];

        gpu.latent_planar(
            &scratch.planar,
            input,
            Dtype::Bf16,
            conv.input_channels,
            shape.frames,
            shape.height,
            shape.width,
            KERNEL[2],
            PADDING[0],
            PADDING[1],
            PADDING[2],
            out_width,
            true,
        )
        .map_err(VaeError::run("planar"))?;

        for frame in 0..shape.frames as usize {
            for tap_t in 0..KERNEL[0] as usize {
                for tap_h in 0..KERNEL[1] as usize {
                    let pass = tap_t * KERNEL[1] as usize + tap_h;
                    let offset = ((frame + tap_t) * padded_height as usize + tap_h)
                        * out_width as usize
                        * columns;
                    gpu.latent_matmul_planar(
                        &scratch.temp,
                        &scratch.planar,
                        offset,
                        &conv.packed,
                        pass * conv.output_channels as usize * columns,
                        0,
                        out_plane as u32,
                        columns as u32,
                        conv.output_channels,
                        tap_t != 0 || tap_h != 0,
                        Dtype::Bf16,
                    )
                    .map_err(VaeError::run("planar matmul"))?;
                }
            }

            gpu.latent_epilogue(
                out,
                &scratch.temp,
                Some(&conv.bias),
                Dtype::Bf16,
                conv.output_channels,
                out_plane as u32,
                shape.frames as usize * out_plane,
                frame * out_plane,
            )
            .map_err(VaeError::run("epilogue"))?;
        }

        Ok(())
    }
}

/// `PixelNorm` (`normalization.py:31-38`), five roundings and all.
fn pixel_norm(gpu: &Gpu, out: &Tensor, input: &Tensor, shape: Shape) -> Result<(), VaeError> {
    gpu.video_vae_pixel_norm(
        out,
        input,
        Dtype::Bf16,
        shape.channels,
        shape.plane(),
        PIXEL_NORM_EPSILON,
    )
    .map_err(VaeError::run("pixel norm"))
}

fn silu(gpu: &Gpu, tensor: &Tensor) -> Result<(), VaeError> {
    gpu.latent_silu(tensor, tensor, Dtype::Bf16, tensor.elements())
        .map_err(VaeError::run("silu"))
}

fn add(gpu: &Gpu, out: &Tensor, input: &Tensor, elements: usize) -> Result<(), VaeError> {
    gpu.latent_add(out, out, input, Dtype::Bf16, elements)
        .map_err(VaeError::run("add"))
}

fn copy(gpu: &Gpu, out: &Tensor, input: &Tensor, elements: usize) -> Result<(), VaeError> {
    gpu.copy(out, 0, input, 0, elements)
        .map_err(VaeError::run("copy"))
}

/// The shard's reader: one reusable host buffer, and the counters the gate
/// reports. Every read checks the checkpoint's own descriptor against the shape
/// the plan derived from the config before anything reaches the device.
struct Reader<'a> {
    gpu: &'a Gpu,
    buffer: Vec<u8>,
    tensors: usize,
    bytes: u64,
}

impl Reader<'_> {
    /// The artifact is BF16 throughout, so a read checks the shard's own dtype
    /// against that and against the shape the config's plan derived.
    fn pull(&mut self, shard: &Shard, name: &str, shape: &[usize]) -> Result<Tensor, VaeError> {
        let tensor = shard
            .find(name)
            .ok_or_else(|| VaeError::Missing(name.to_string()))?;
        let want: Vec<u64> = shape.iter().map(|dim| *dim as u64).collect();

        if tensor.dtype != ShardDtype::Bf16 || tensor.shape != want {
            return Err(VaeError::Shape(format!(
                "{name}: checkpoint has {} {:?}, the plan needs {} {:?}",
                tensor.dtype.name(),
                tensor.shape,
                ShardDtype::Bf16.name(),
                want
            )));
        }

        let elements: u64 = want.iter().product();
        if tensor.elements() != elements || tensor.bytes() != elements * ShardDtype::Bf16.size() {
            return Err(VaeError::Shape(format!(
                "{name}: {} elements in {} bytes do not match {} x {}",
                tensor.elements(),
                tensor.bytes(),
                elements,
                ShardDtype::Bf16.size()
            )));
        }

        self.buffer.clear();
        self.buffer.resize(tensor.bytes() as usize, 0);
        shard
            .read_data(tensor, &mut self.buffer)
            .map_err(|error| VaeError::Reader(error.token()))?;

        let device = self
            .gpu
            .alloc(Dtype::Bf16, elements as usize)
            .map_err(VaeError::run(name))?;
        self.gpu
            .upload(&device, &self.buffer)
            .map_err(VaeError::run(name))?;

        self.tensors += 1;
        self.bytes += tensor.bytes();

        Ok(device)
    }

    /// One convolution: `[out][in][kd][kh][kw]` and `[out]`, plus the pack the
    /// planar path reads.
    fn conv(
        &mut self,
        shard: &Shard,
        name: &str,
        input_channels: u32,
        output_channels: u32,
    ) -> Result<Conv, VaeError> {
        let shape = [
            output_channels as usize,
            input_channels as usize,
            KERNEL[0] as usize,
            KERNEL[1] as usize,
            KERNEL[2] as usize,
        ];
        let weight = self.pull(shard, &format!("{name}.weight"), &shape)?;
        let bias = self.pull(shard, &format!("{name}.bias"), &[output_channels as usize])?;
        let packed = self
            .gpu
            .alloc(Dtype::Bf16, weight.elements())
            .map_err(VaeError::run(name))?;
        self.gpu
            .latent_weight_pack(
                &packed,
                &weight,
                Dtype::Bf16,
                output_channels,
                input_channels,
                KERNEL[0] * KERNEL[1] * KERNEL[2],
                KERNEL[2],
            )
            .map_err(VaeError::run("weight pack"))?;

        Ok(Conv {
            packed,
            bias,
            input_channels,
            output_channels,
        })
    }
}

/// The artifact's own `config.vae`, which this graph is built from. A shard
/// whose config disagrees with [`DECODER_BLOCKS`] is refused by name rather
/// than silently producing a latent-plausible picture.
fn assert_config(shard: &Shard) -> Result<(), VaeError> {
    let text = shard
        .metadata
        .get("config")
        .ok_or_else(|| VaeError::Shape("checkpoint carries no __metadata__.config".into()))?;
    let config: serde_json::Value =
        serde_json::from_str(text).map_err(|error| VaeError::Shape(format!("config: {error}")))?;
    let vae = &config["vae"];

    let text_field = |name: &str| -> Result<&str, VaeError> {
        vae.get(name)
            .and_then(|value| value.as_str())
            .ok_or_else(|| VaeError::Shape(format!("config.vae has no {name}")))
    };
    let number_field = |name: &str| -> Result<u64, VaeError> {
        vae.get(name)
            .and_then(|value| value.as_u64())
            .ok_or_else(|| VaeError::Shape(format!("config.vae has no {name}")))
    };

    if text_field("_class_name")? != "CausalVideoAutoencoder" {
        return Err(VaeError::Shape(format!(
            "config.vae._class_name is {}, not the conv autoencoder",
            text_field("_class_name")?
        )));
    }
    // The decoder's own switches, each one this port's assumption: a conv
    // decoder with no timestep conditioning, a uniform latent log variance,
    // pixel norm, zero spatial padding and a non-causal (replicated temporal
    // halo) padding policy.
    for name in [
        "causal_decoder",
        "timestep_conditioning",
        "use_quant_conv",
        "normalize_latent_channels",
    ] {
        let value = vae
            .get(name)
            .ok_or_else(|| VaeError::Shape(format!("config.vae has no {name}")))?;
        if value.as_bool() != Some(false) {
            return Err(VaeError::Shape(format!(
                "config.vae.{name} is {value}, not false"
            )));
        }
    }
    if text_field("latent_log_var")? != "uniform" {
        return Err(VaeError::Shape(format!(
            "config.vae.latent_log_var is {}, not uniform",
            text_field("latent_log_var")?
        )));
    }
    if text_field("norm_layer")? != "pixel_norm" || text_field("spatial_padding_mode")? != "zeros" {
        return Err(VaeError::Shape(format!(
            "config.vae says {} / {}, the port assumes pixel_norm / zeros",
            text_field("norm_layer")?,
            text_field("spatial_padding_mode")?
        )));
    }
    if number_field("latent_channels")? != LATENT_CHANNELS as u64
        || number_field("patch_size")? != PATCH_SIZE as u64
        || number_field("decoder_base_channels")? != BASE_CHANNELS as u64
    {
        return Err(VaeError::Shape(
            "config.vae's latent_channels / patch_size / decoder_base_channels are not 128 / 4 / 128"
                .into(),
        ));
    }

    let blocks = vae
        .get("decoder_blocks")
        .and_then(|value| value.as_array())
        .ok_or_else(|| VaeError::Shape("config.vae has no decoder_blocks".into()))?;
    if blocks.len() != DECODER_BLOCKS.len() {
        return Err(VaeError::Shape(format!(
            "config.vae lists {} decoder blocks, the port's table has {}",
            blocks.len(),
            DECODER_BLOCKS.len()
        )));
    }
    for (index, (entry, block)) in blocks.iter().zip(DECODER_BLOCKS.iter()).enumerate() {
        let entry = entry.as_array().ok_or_else(|| {
            VaeError::Shape(format!(
                "decoder_blocks.{index} is not a [name, params] pair"
            ))
        })?;
        let name = entry
            .first()
            .and_then(|value| value.as_str())
            .ok_or_else(|| VaeError::Shape(format!("decoder_blocks.{index} has no name")))?;
        let params = entry.get(1).unwrap_or(&serde_json::Value::Null);

        match block {
            Block::Res { layers } => {
                if !name.starts_with("res_x") {
                    return Err(VaeError::Shape(format!(
                        "decoder_blocks.{index} is {name}, the table has a res group"
                    )));
                }
                if params.get("num_layers").and_then(|value| value.as_u64()) != Some(*layers as u64)
                {
                    return Err(VaeError::Shape(format!(
                        "decoder_blocks.{index} (res_x) has {} layers, the table has {layers}",
                        params.get("num_layers").unwrap_or(&serde_json::Value::Null)
                    )));
                }
            }
            Block::Upsample { stride, reduction } => {
                // The name is the block's own: `compress_space` doubles the two
                // spatial axes, `compress_time` the temporal one and
                // `compress_all` all three.
                let want_name = match stride {
                    [1, 2, 2] => "compress_space",
                    [2, 1, 1] => "compress_time",
                    _ => "compress_all",
                };
                if name != want_name {
                    return Err(VaeError::Shape(format!(
                        "decoder_blocks.{index} is {name}, the table has {want_name}"
                    )));
                }
                if params.get("multiplier").and_then(|value| value.as_u64())
                    != Some(*reduction as u64)
                {
                    return Err(VaeError::Shape(format!(
                        "decoder_blocks.{index} ({name}) has multiplier {}, the table has {reduction}",
                        params
                            .get("multiplier")
                            .unwrap_or(&serde_json::Value::Null)
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_factors_are_the_blocks_own() {
        // `types.py:41-57`: three spatial doublings and three temporal ones
        // (two `compress_all` and one `compress_time`), with the patch size on
        // the two spatial axes - which is `VIDEO_SCALE_FACTORS.default()`.
        assert_eq!(scale_factors(), [8, 32, 32]);
    }

    #[test]
    fn the_bottleneck_is_eight_times_the_base_width() {
        // `conv_video_decoder.py:43-58` with the config's multipliers
        // 2 * 2 * 1 * 2.
        assert_eq!(bottleneck_channels(), 1024);
        // And that is exactly what the artifact's `conv_in` declares.
        assert_eq!(BASE_CHANNELS * 8, bottleneck_channels());
    }

    #[test]
    fn the_release_geometry_round_trips() {
        // `types.py:110-134` both ways against `models.py`'s own target.
        assert_eq!(latent_shape(RELEASE_PIXELS), RELEASE_LATENT);
        assert_eq!(pixel_shape(RELEASE_LATENT), RELEASE_PIXELS);
        // The refiner's stage 2 emits the same grid: 16 x 24 x 42 tokens.
        assert_eq!(RELEASE_LATENT.iter().product::<u32>(), 16 * 24 * 42);
    }

    #[test]
    fn the_channel_plan_is_the_artifacts_own() {
        // Reversed construction, the widths each `up_blocks.N` carries. These
        // are the checkpoint's own shapes: 4096 = 8 x 1024 / 2 at
        // `up_blocks.1`, 4096 = 8 x 512 / 1 at `up_blocks.3`, 512 =
        // 2 x 512 / 2 at `up_blocks.5`, 512 = 4 x 256 / 2 at `up_blocks.7`.
        let mut widths = vec![bottleneck_channels()];
        let mut width = bottleneck_channels();
        let mut convs = Vec::new();

        for block in DECODER_BLOCKS.iter().rev() {
            if let Block::Upsample {
                stride, reduction, ..
            } = block
            {
                convs.push(width * stride[0] * stride[1] * stride[2] / reduction);
                width /= reduction;
                widths.push(width);
            }
        }

        assert_eq!(convs, vec![4096, 4096, 512, 512]);
        assert_eq!(widths, vec![1024, 512, 512, 256, 128]);
        assert_eq!(
            PIXEL_CHANNELS * PATCH_SIZE * PATCH_SIZE,
            48,
            "the artifact's conv_out carries 48 channels"
        );
    }

    #[test]
    fn the_res_groups_carry_the_artifacts_layer_counts() {
        // 8, 2, 8, 2, 16, 2, 24, 2, 16 tensors per `up_blocks.N`: 2 per
        // resnet (a weight and a bias each), 2 per upsampler.
        let counts: Vec<usize> = DECODER_BLOCKS
            .iter()
            .rev()
            .map(|block| match block {
                Block::Res { layers } => layers * 4,
                Block::Upsample { .. } => 2,
            })
            .collect();

        assert_eq!(counts, vec![8, 2, 8, 2, 16, 2, 24, 2, 16]);
        // Plus `conv_in` and `conv_out`, four tensors: the artifact's 84.
        assert_eq!(counts.iter().sum::<usize>() + 4, 84);
    }

    #[test]
    fn the_upsampler_shapes_are_the_shuffles_own() {
        // `sampling.py:111-120` from the latent end to the end of the graph,
        // with the temporal axis dropping the first output frame each time it
        // doubles: 16 -> 31 -> 61 -> 121, and the spatial axes at 24 -> 48 ->
        // 96 -> 192 and 42 -> 84 -> 168 -> 336, one patch short of the pixels.
        let mut shape = Shape {
            channels: bottleneck_channels(),
            frames: RELEASE_LATENT[0],
            height: RELEASE_LATENT[1],
            width: RELEASE_LATENT[2],
        };
        let mut frames = vec![shape.frames];

        for block in DECODER_BLOCKS.iter().rev() {
            if let Block::Upsample {
                stride, reduction, ..
            } = block
            {
                shape = shape.upsampled(*stride, *reduction);
                frames.push(shape.frames);
            }
        }

        assert_eq!(frames, vec![16, 31, 61, 121, 121]);
        assert_eq!([shape.height, shape.width], [192, 336]);
        assert_eq!(shape.channels, BASE_CHANNELS);
        assert_eq!(
            [shape.height, shape.width],
            [
                RELEASE_PIXELS[1] / PATCH_SIZE,
                RELEASE_PIXELS[2] / PATCH_SIZE
            ]
        );
    }

    #[test]
    fn the_scratch_covers_the_largest_stage() {
        // The sized buffers must hold the release geometry's largest
        // activation and its largest convolution output.
        let latent = Shape {
            channels: LATENT_CHANNELS,
            frames: RELEASE_LATENT[0],
            height: RELEASE_LATENT[1],
            width: RELEASE_LATENT[2],
        };

        // 128 channels at 121 x 192 x 336, which is also what `up_blocks.7`'s
        // convolution emits before its shuffle: 512 channels at 121 x 96 x 168.
        assert_eq!(largest_activation(latent), 128 * 121 * 192 * 336);
        assert_eq!(largest_conv_output(latent), 512 * 121 * 96 * 168);
        assert_eq!(
            largest_activation(latent),
            largest_conv_output(latent),
            "the shuffle trades four channels for four positions"
        );
        // The planar buffer is the same channel-last padded image, `kw` wide,
        // for the largest convolution's input: `up_blocks.7`'s 512 channels at
        // 121 x 96 x 168, just ahead of the last resnet group's 128 channels at
        // 121 x 192 x 336.
        assert_eq!(largest_planar(latent), 512 * 123 * 98 * 168 * 3);
        assert_eq!(largest_planar(latent), 3_110_510_592);
        // The widest accumulator is `up_blocks.3`'s 4096-channel convolution,
        // one frame at a time - its own input's plane, since the shuffle comes
        // after the convolution.
        assert_eq!(largest_temp(latent), 4096 * 48 * 84);
        assert_eq!(pixel_elements(latent), 3 * 121 * 768 * 1344);
    }
}
