# MiniMax-H3 reference tools

These probes compare the Rust port with the pinned
[h3.c reference](../../docs/minimax-h3/README.md#source-and-ownership).
They are not linked into production hosts.

| Tool | Purpose |
|---|---|
| `st_probe.c`, `tok_probe.c` | Safetensors and tokenizer parity |
| `text_probe.c`, `vision_probe.c`, `multimodal_probe.c` | Prompt and vision encoder parity |
| `dit_probe.c` | DiT forward and sampler parity |
| `audio_encoder_probe.c`, `video_encoder_probe.c` | Conditioning VAE parity |
| `reference_canvas_probe.c` | Reference geometry selection |
| `ref_gate.sh` | Render supplied media cases with either engine and compare manifests |
| `ppm_diff.c` | Pixel differences, PSNR and spatial attribution |
| `standin-renderer.sh` | Model-free HTTP plumbing, not model qualification |

The H3-to-LTX latent path has no C reference: its authority is the Sol-H3-Spark
profile's Python plus the author's ComfyUI node, so its probe is Python.

| Tool | Purpose |
|---|---|
| `ltx/latent_oracle.py` | Golden tensors for the latent upscale, the adapter and the crop to 16 frames, run against the real checkpoints and pinned by sha256 in its manifest |

Build probes against a reference checkout supplied by the operator. For
example, after building its `libh3.a`:

```sh
REFERENCE_DIR=/path/to/h3.c
cc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -I "$REFERENCE_DIR" \
  tools/minimax_h3/dit_probe.c "$REFERENCE_DIR/libh3.a" \
  -L/usr/local/cuda/lib64 -lcudart -lcublasLt -lstdc++ -licui18n -licuuc -lm \
  -o scratch/h3-dit-probe
```

The media gate takes `--binary`, `--model`, `--out`, `--clip` and `--image`.
Use `--case NAME` to limit work and `--compare DIR` to compare a prior run.
It performs full 20-step renders; put the invocation under the
[memory guard](../../docs/host-memory-guard.md), and run the two engines
sequentially. Supply the same input bytes, seed, workload and build target.
Comparison gates decoded video/audio hashes and geometry; elapsed time and
MP4 bytes remain diagnostics. `python3 tests/h3/test_ref_gate.py` checks the
gate with tiny media and a stub renderer, without loading a model.

After building `h3-rs`, `python3 tests/h3/test_cli_conditioning.py` checks
the CLI's conditioning dimensions. It stops at a stub decoder before any
model weights or GPU work; `H3_BINARY` can select a different port binary.
