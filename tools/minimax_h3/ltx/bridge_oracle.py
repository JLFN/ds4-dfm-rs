#!/usr/bin/env python3
"""Golden outputs for the H3 -> LTX bridge, from the pinned references themselves.

The bridge has no C reference: its authority is the Sol-H3-Spark profile's own
Python (`runtime/stage2_ops/h3_ltx_adapter/`) plus the author's ComfyUI node
(`nodes/minimax_h3_latent_upscaler_3d.py` at its pinned commit). This script
runs both halves, at a geometry the caller picks, against the real checkpoints,
and writes the tensors plus a manifest that pins them. The Rust port compares
against that; see docs/minimax-h3/ltx-reference-map-2026-10-28.md §2 and §8.

Run it in float32 on all cores. torch's CPU bf16 conv3d has no fast path: the
same tiny upscaler forward took 483.5 s at bf16 and 0.5 s at float32 across 20
threads (record §8). An unseeded run is not a reference, so the seed is hashed
into the manifest.

    python3 -m venv --system-site-packages /tmp/oracle-venv
    /tmp/oracle-venv/bin/pip install einops
    /tmp/oracle-venv/bin/python tools/minimax_h3/ltx/bridge_oracle.py \
        --frames 124 --draft-height 384 --draft-width 672 --out-dir /tmp/bridge-golden

--draft-height/--draft-width are the H3 stage's own canvas (384x672 in the
release's recipe, the same as this port's job A); the adapter is handed twice
that, 768x1344, which is what makes its 48x84 input grid the LTX latent grid.
A small geometry (for example --frames 17 --draft-height 128 --draft-width 128)
finishes in seconds.

Defaults point at the pinned references: the Spark profile inside
/data/Sol-h3-Spark and its artifact store over the gvfs mount. Override with
--profile-runtime, --node-source and --artifacts.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# The profile's own constants; the adapter package is imported for real below.
GB10_ARTIFACTS = (
    "/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/mnt/h3-artifacts"
)
PROFILE_RUNTIME = "/data/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark/runtime"
NODE_SOURCE = "/data/h3-upscaler-node/nodes/minimax_h3_latent_upscaler_3d.py"

# The node defines exactly these, in this order, and the profile asserts it.
NODE_SYMBOLS = (
    "normalization", "zero_module", "AttnBlock3D", "ResBlockEmb3D",
    "TemporalConv", "LatentResizer3D", "_extract_upscaler_sd", "_detect_arch",
)

UPSCALER_BYTES = 690592992
UPSCALER_SHA256 = "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
ADAPTER_SHA256 = "170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_node(node_source: Path) -> dict:
    """Exec only the pinned symbols, as the profile does, and return the namespace."""
    source = node_source.read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NODE_SYMBOLS]
    if tuple(node.name for node in nodes) != NODE_SYMBOLS:
        raise ValueError("pinned LBH model definitions are missing or reordered")

    namespace = {"torch": torch, "nn": nn, "F": F, "gc": gc, "re": re,
                 "rearrange": rearrange}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(node_source), "exec"),
         namespace)
    return namespace


def build_upscaler(namespace: dict, checkpoint: Path, dtype):
    from safetensors.torch import load_file

    state = namespace["_extract_upscaler_sd"](load_file(str(checkpoint), device="cpu"))
    config = namespace["_detect_arch"](state)
    model = namespace["LatentResizer3D"](**config)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(dtype), config


def normalization_statistics(node_source: Path) -> tuple[list, list]:
    """The node's own 24-channel LATENTS_MEAN / LATENTS_STD literals."""
    values = {}
    for node in ast.parse(node_source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"LATENTS_MEAN", "LATENTS_STD"}:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {"LATENTS_MEAN", "LATENTS_STD"}:
        raise ValueError("node normalization statistics are missing")
    return values["LATENTS_MEAN"], values["LATENTS_STD"]


def stats(tensor: torch.Tensor) -> dict:
    wide = tensor.float()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "mean": float(wide.mean()), "std": float(wide.std()),
            "min": float(wide.min()), "max": float(wide.max()),
            "sum": float(wide.sum())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=124,
                        help="pixel frames the draft was rendered at")
    parser.add_argument("--draft-height", type=int, default=384,
                        help="the H3 draft's pixel height (stage 1 canvas)")
    parser.add_argument("--draft-width", type=int, default=672,
                        help="the H3 draft's pixel width (stage 1 canvas)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--out-dir", default="/tmp/bridge-golden")
    parser.add_argument("--artifacts", default=GB10_ARTIFACTS)
    parser.add_argument("--profile-runtime", default=PROFILE_RUNTIME)
    parser.add_argument("--node-source", default=NODE_SOURCE)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(args.threads)
    artifacts = Path(args.artifacts)
    runtime = Path(args.profile_runtime).resolve(strict=True)
    node_source = Path(args.node_source).resolve(strict=True)

    sys.path.insert(0, str(runtime / "stage2_ops"))
    from h3_ltx_adapter.adapter import H3ToLTXAdapter, expected_shapes

    adapter_dir = artifacts / "H3-to-LTX-Latent-Adapter"
    upscaler_ckpt = artifacts / "H3-upscaler" / "minimax_h3_latent_upscaler_3d_bf16.safetensors"

    # The chain the profile runs (stage2.py:58-76): the draft's latent is
    # upscaled 2x, and the ADAPTER is then told the doubled canvas, 768x1344
    # here, because the upscaled grid 48x84 is that canvas at 16x. Its
    # expected_shapes() enforces exactly that, so passing the draft canvas
    # raises rather than silently mis-shaping.
    canvas_h, canvas_w = args.draft_height * 2, args.draft_width * 2
    expected = expected_shapes(args.frames, canvas_h, canvas_w)
    h3_shape = [24, expected["h3"][1], args.draft_height // 16, args.draft_width // 16]
    upscaled_shape = (1, 24, h3_shape[1], h3_shape[2] * 2, h3_shape[3] * 2)
    if [upscaled_shape[1], *upscaled_shape[2:]] != expected["h3"]:
        raise ValueError(f"upscaled {upscaled_shape} is not the adapter's H3 input {expected['h3']}")
    print(f"draft canvas {args.draft_height}x{args.draft_width} -> adapter canvas "
          f"{canvas_h}x{canvas_w}; H3 {expected['h3']} -> LTX {expected['ltx']}")

    print("=== upscaler ===")
    namespace = load_node(node_source)
    started = time.time()
    upscaler, upscaler_config = build_upscaler(namespace, upscaler_ckpt, dtype)
    print(f"load+detect in {time.time() - started:.1f}s -> {upscaler_config}")
    mean, std = normalization_statistics(node_source)
    mean = torch.tensor(mean, dtype=dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(std, dtype=dtype).view(1, -1, 1, 1, 1)

    torch.manual_seed(args.seed)
    h3_input = torch.randn(1, *h3_shape, dtype=dtype)
    # The profile's own order: the node's N, its forward at scale 2.0, then D.
    network_input = (h3_input - mean) / std
    started = time.time()
    with torch.inference_mode():
        upscaled = upscaler(network_input, scale=2.0, target_size=upscaled_shape[2:],
                            enable_chunking=True)
        upscaled = upscaled * std + mean
    print(f"upscale in {time.time() - started:.1f}s -> {tuple(upscaled.shape)} "
          f"(chunked: T={h3_shape[1]})")

    print("=== adapter ===")
    started = time.time()
    adapter = H3ToLTXAdapter.from_pretrained(adapter_dir, device="cpu", dtype=dtype)
    print(f"loaded in {time.time() - started:.1f}s")
    started = time.time()
    with torch.inference_mode():
        adapter_output = adapter.convert(
            upscaled, pixel_frames=args.frames,
            pixel_height=canvas_h, pixel_width=canvas_w,
            input_normalization="normalized")
        refiner_input = adapter_output[:, :, :16].contiguous()
    print(f"convert in {time.time() - started:.1f}s -> {tuple(adapter_output.shape)}, "
          f"crop 16 -> {tuple(refiner_input.shape)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"h3_input": h3_input.contiguous(),
               "h3_upscaled": upscaled.contiguous(),
               "adapter_output": adapter_output.contiguous(),
               "refiner_input": refiner_input.contiguous()},
              str(out_dir / "bridge.safetensors"))

    manifest = {
        "schema": "h3_ltx_bridge_oracle_v1",
        "seed": args.seed, "dtype": args.dtype, "threads": args.threads,
        "geometry": {"pixel_frames": args.frames,
                     "draft_canvas": [args.draft_height, args.draft_width],
                     "adapter_canvas": [canvas_h, canvas_w]},
        "torch": torch.__version__,
        "references": {
            "profile_runtime": str(runtime),
            "node_source": str(node_source),
            "node_source_sha256": file_sha256(node_source),
        },
        "checkpoints": {
            "adapter": {"path": str(adapter_dir / "model.safetensors"),
                        "sha256": file_sha256(adapter_dir / "model.safetensors")},
            "upscaler": {"path": str(upscaler_ckpt),
                         "bytes": upscaler_ckpt.stat().st_size,
                         "sha256": file_sha256(upscaler_ckpt)},
        },
        "expected_shapes": expected, "upscaler_config": upscaler_config,
        "tensors": {"h3_input": stats(h3_input), "h3_upscaled": stats(upscaled),
                    "adapter_output": stats(adapter_output),
                    "refiner_input": stats(refiner_input)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["tensors"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
