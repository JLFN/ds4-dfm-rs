#!/usr/bin/env python3
"""Render one shot through the resident h3 server's local v1 API.

The server (`./h3-server-rs`) opens the engine once at startup, so a render
submitted here does not pay the 40 GB checkpoint load - that is the whole point
of the served path over `h3-rs --jobs`, which loads per process. This client is
the stdlib-only way to use it:

  tools/h3-api.py --prompt "a red fox in snow" --out outputs/x.mp4
  tools/h3-api.py --prompt-file p.txt --short-edge 768 --aspect 9:16 \\
      --seconds 5 --out outputs/ipanema.mp4

Contract: `docs/minimax-h3/local-v1-contract.md` - `POST /v1/videos`,
`GET /v1/videos/{id}` until `status` is terminal, `GET /v1/videos/{id}/content`
for the bytes. The create body is checked strictly by the server: exactly
`task`, `prompt`, `conditions`, `target`, `seed` (plus optional `attention`,
`vae_attention`, `vae_gemm` and `profile`), and an unknown field is refused
rather than ignored. `--profile` names a recipe the host resolves: the same
`tools/h3-profiles.json` the CLI reads, so a served render and a local one mean
the same thing by the same name.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# The contract's four states (`local-v1-contract.md:39`); a client that
# invents its own terminal set polls a finished job forever.
TERMINAL = {"completed", "failed"}


def request(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"content-type": "application/json"}
    opening = urllib.request.Request(url, data=data, headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(opening, timeout=timeout) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8090")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--task", default="t2va", choices=["t2va", "fl2va", "ref2va"])
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention", choices=["tensor", "scalar"])
    parser.add_argument("--profile", help="a named recipe from tools/h3-profiles.json")
    parser.add_argument("--crf", type=int, help="the mux line's crf (1-51)")
    parser.add_argument("--preset", help="the mux line's x264 preset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll", type=float, default=2.0)
    arguments = parser.parse_args()

    prompt = arguments.prompt
    if arguments.prompt_file:
        with open(arguments.prompt_file, "r", encoding="utf-8") as handle:
            prompt = " ".join(handle.read().split())
    if not prompt:
        raise SystemExit("one of --prompt or --prompt-file is required")

    body = {
        "task": arguments.task,
        "prompt": prompt,
        "conditions": [],
        "target": {
            "short_edge": arguments.short_edge,
            "aspect_ratio": arguments.aspect,
            "duration_seconds": arguments.seconds,
        },
        "seed": arguments.seed,
    }
    if arguments.attention:
        body["attention"] = arguments.attention
    if arguments.profile:
        body["profile"] = arguments.profile
    if arguments.crf:
        body["crf"] = arguments.crf
    if arguments.preset:
        body["preset"] = arguments.preset

    started = time.time()
    status, payload = request(f"{arguments.base}/v1/videos", body)
    if status not in (200, 201, 202):
        raise SystemExit(f"create refused ({status}): {payload[:400]!r}")
    answer = json.loads(payload)
    identifier = answer.get("id") or answer.get("task_id")
    if not identifier:
        raise SystemExit(f"create returned no id: {payload[:400]!r}")
    print(f"POST {identifier} {status} at {time.time() - started:.3f}s", flush=True)

    seen = None
    while True:
        status, payload = request(f"{arguments.base}/v1/videos/{identifier}")
        record = json.loads(payload) if payload else {}
        state = record.get("status")
        if state != seen:
            print(f"  {state} at {time.time() - started:.1f}s", flush=True)
            seen = state
        if state in TERMINAL:
            break
        time.sleep(arguments.poll)
    if state != "completed":
        raise SystemExit(f"render {state}: {json.dumps(record)[:600]}")

    status, content = request(f"{arguments.base}/v1/videos/{identifier}/content")
    if status != 200:
        raise SystemExit(f"content refused ({status}): {content[:300]!r}")
    with open(arguments.out, "wb") as handle:
        handle.write(content)
    print(f"saved {arguments.out} {len(content)} bytes at {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
