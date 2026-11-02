#!/usr/bin/env python3
"""Read one named render recipe out of a profile file for tools/h3-rapid.sh.

The profile file is JSON with a `profiles` object, one entry per recipe
(`tools/h3-profiles.json` is the shipped set). This tool does one thing: turn a
profile into shell assignments the script can `eval`, so the recipes live in a
versioned file instead of in the script's defaults and a clip can always be
re-rendered from the name it was made with.

  tools/h3-profile.py --file tools/h3-profiles.json --list
  tools/h3-profile.py --file tools/h3-profiles.json --name ts_h2
  # steps=6
  # draft_scale=0.7
  # ...

Standard library only, so it runs anywhere the campaign runs. Values are
`shlex.quote`d, so a doc string with an apostrophe cannot break the `eval`.
Unknown keys are refused by name: a typo in a recipe is an error, never a
silently ignored line.
"""
import argparse
import json
import shlex
import sys

# The script's flag surface, and what a profile key becomes in the shell. The
# three arms are exported rather than passed on the command line, because the
# renderer reads them from the environment.
SHELL = {
    "steps": "STEPS",
    "draft_scale": "DRAFT_SCALE",
    "refine_sigma": "REFINE_SIGMA",
    "draft_attention": "DRAFT_ATTENTION",
    "seconds": "SECONDS_ARG",
    "width": "WIDTH",
    "height": "HEIGHT",
    "crf": "CRF",
    "preset": "PRESET",
    "seed": "SEED",
    "one_pass": "ONE_PASS",
    "dit_gemm": "H3_DIT_GEMM",
    "vae_gemm": "H3_VAE_GEMM",
    "vae_attention": "H3_VAE_ATTENTION",
}


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise SystemExit(f"{path}: no `profiles` object")
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default="tools/h3-profiles.json")
    parser.add_argument("--name")
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()
    profiles = load(arguments.file)

    if arguments.list or not arguments.name:
        width = max(len(name) for name in profiles)
        for name, profile in profiles.items():
            doc = profile.get("doc", "")
            print(f"{name.ljust(width)}  {doc}")
        return 0

    profile = profiles.get(arguments.name)
    if profile is None:
        raise SystemExit(
            f"{arguments.file}: no profile {arguments.name!r}; have "
            + ", ".join(sorted(profiles))
        )
    for key, value in profile.items():
        if key == "doc":
            continue
        variable = SHELL.get(key)
        if variable is None:
            raise SystemExit(
                f"{arguments.file}: profile {arguments.name!r}: unknown key {key!r}; "
                "have " + ", ".join(sorted(SHELL))
            )
        if isinstance(value, bool):
            value = 1 if value else 0
        print(f"{variable}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
