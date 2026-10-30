# Unit 5: the stage-2 prompt cache

The file the refiner loads instead of a text encoder. Unit 4 produces the two
bf16 contexts in memory; this unit makes them a file, and makes the port able
to read the file the release's own Python writes.

The operator settled this unit's first question - the on-disk format - as full
`torch.save` compatibility in both directions, so the port now reads and writes
the release's own container rather than a private one.

Everything below was read first-hand: the release's `runtime/prompt_cache.py`
and `runtime/cache_builder.py` in full, and PyTorch 2.13.0's own sources for the
container (`torch/serialization.py`, `torch/_utils.py`,
`torch/_weights_only_unpickler.py`, `torch/storage.py`,
`caffe2/serialize/inline_container.{h,cc}`, `caffe2/serialize/versions.h`). No
Python was executed; the container contract is read from those sources and
checked against a third-party zip reader.

## 1. The container the release writes

`torch.save` since 1.6 writes an uncompressed ZIP64 archive
(`caffe2/serialize/inline_container.h`, the layout comment;
`serialization.py::save` with `_use_new_zipfile_serialization=True`):

    <basename>/data.pkl             pickle protocol 2 of the payload
    <basename>/.format_version      "1"
    <basename>/.storage_alignment   "64"
    <basename>/byteorder            "little"
    <basename>/data/<key>           one uncompressed record per storage
    <basename>/version              "3\n"

Facts that decide a compatible writer, each from the source:

- **The records are inside a subdirectory named after the file.**
  `PyTorchStreamWriter::setup` sets `archive_name_ = basename(file_name)`, and
  `basename` cuts at the last dot; the reader takes that name from the **first
  entry** (`PyTorchStreamReader::init`). A file named `prompt_cache.pt` holds
  `prompt_cache/data.pkl`. Renaming the file afterwards is harmless: the reader
  reads the prefix out of the archive, not out of the path.
- **Every record is padded to a 64-byte boundary** through an `FB` extra field
  whose 4-byte header carries the padding length (`detail::getPadding`,
  `detail::getOffset`), and every record carries a CRC32 over the uncompressed
  data. miniz verifies that CRC when the record is read, so a wrong one is not
  a cosmetic difference.
- **The writer always writes ZIP64** (`MZ_ZIP_FLAG_WRITE_ZIP64`), and
  `writeEndOfFile` stamps `version` with `version_`, whose default is
  `kMinProducedFileFormatVersion` = 3 (`versions.h`). The reader requires that
  record to parse into `[kMinSupportedFileFormatVersion` 1,
  `kMaxSupportedFileFormatVersion` 10`]`, and falls back to `.data/version` when
  the version is 6 or more.
- **`.data/serialization_id` is optional on read** (`hasRecord` guards it in
  `PyTorchStreamReader::init`); it is metadata, derived by
  `writeSerializationId` from `c10::hash` of the record names and the combined
  record CRC.

`data.pkl` is pickle protocol 2 (`DEFAULT_PROTOCOL`). A storage is not pickled
as data: `_save`'s `persistent_id` returns the tuple

    ("storage", <storage class>, <key>, <location>, <numel>)

followed by `BINPERSID`, and `_load`'s `persistent_load` reads the record
`data/<key>`. The tensor around it is rebuilt by
`torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
requires_grad, backward_hooks, metadata=None)`, with `size` and `stride` as
`torch.Size` (`torch/_utils.py`). The bfloat16 storage class is named
`BFloat16Storage` (`torch/storage.py::_dtype_to_storage_type_map`), and
`_get_dtype_from_pickle_storage_type` reads it back.

A `weights_only=True` load accepts a fixed set of opcodes and globals
(`torch/_weights_only_unpickler.py`): the rebuild function, the storage classes
and dtypes, `torch.Size`, `collections.OrderedDict`, `torch.Tensor`, and the
primitive opcodes. Nothing else, which is what bounds both directions here.

## 2. The payload contract

`runtime/prompt_cache.py` is the whole contract:

- `RECIPE` (`:16-28`), eleven entries naming both artifacts and their byte
  lengths, the int8 line counts, and the boundary as
  `post_INT8_connector_video_and_audio_encoding`.
- `FIXED_PROMPT` (`:17`).
- `fingerprint` (`:32-33`): the sha256 of
  `json.dumps({"prompt": prompt, "recipe": RECIPE}, sort_keys=True,
  separators=(",", ":"))`. That single string ties a cache to a prompt **and**
  to both artifacts.
- `validate_payload` (`:38-61`): `schema_version` 1; the prompt; the recipe
  equal to `RECIPE`; the fingerprint equal to `fingerprint(prompt)`; both
  contexts present; each on CPU, `torch.bfloat16`, exactly
  `CONTEXT_SHAPES` = video `(1, 1024, 4096)` and audio `(1, 1024, 2048)`
  (`:29-30`), every size >= 1, at most `MAX_CACHE_BYTES` 64 MiB, finite; a
  matching `tensor_manifest` entry; and `prompt_stats` with
  `characters == len(prompt)` and `truncated is False`.
- `load_cache` (`:93-100`): the file must be at most `MAX_CACHE_BYTES`, then
  `torch.load(path, map_location="cpu", weights_only=True)`.

`validate_payload` checks shape, dtype, finiteness and the fingerprint but
**not values**, so a subtly wrong cache passes it. The value gate is the
operator's golden, and this unit does not have it.

## 3. What the port implements

`minimax_h3::torch_file` is the container and the pickle:

- a zip reader over the end records, the zip64 locator and the central
  directory, with the `FB`/zip64 extra fields parsed and each record's CRC32
  verified on read;
- a pickle protocol-2 interpreter covering the opcode set the reference load
  accepts, including memoization (`BINPUT`/`BINGET`/`LONG_BINPUT`/
  `LONG_BINGET`), `REDUCE` for `torch._utils._rebuild_tensor_v2`,
  `torch.Size` and `collections.OrderedDict`, and the storage
  `persistent_load` tuple;
- a writer that emits protocol 2 with no memoization, one `data/<key>` record
  per tensor, ZIP64 end records plus a zip32 classic end record with real
  counts, and the `FB` 64-byte alignment.

`minimax_h3::prompt_cache` is the payload: `RECIPE`, `FIXED_PROMPT`,
`canonical_json`/`fingerprint`, `validate` with every rule above, `load`,
`save` and `payload`.

Deviations, named:

- Only stored (method 0) records are read. `torch.save` and the reference
  writer both store uncompressed, so this refuses nothing a release file
  contains; a compressed record is refused by name rather than mishandled.
- The writer omits `.data/serialization_id`, which the reference writer
  derives from `c10::hash` of the record names and the combined CRC. The
  reference reader treats it as optional, and nothing in the port reads it.
- `_rebuild_tensor_v2` is written with six arguments (through
  `backward_hooks`); `metadata` defaults to `None` in the reference signature,
  and the reference's C++ reducer may pass a seventh.
- `save` refuses an existing destination, as `cache_builder.py:94-96` does.
- `characters` is compared in code points, which is Python's `len(str)`, and
  the canonical JSON escapes the way `json.dumps(ensure_ascii=True)` does.

## 4. Evidence

Unit tests (`cargo test -p ds4-core --lib`, 745 pass, 7 ignored): six in
`torch_file` and three in `prompt_cache`.

- The CRC32 is the zip polynomial's (`crc32("123456789") == 0xCBF43926`).
- `basename` cuts at the last dot, as the reference's does.
- The `LONG1` encoder is the shortest two's complement body: `+2147483648` is
  five bytes because four would read back negative.
- A payload round-trips byte-exactly, with the record list
  `data.pkl, .format_version, .storage_alignment, byteorder, data/0, version`
  and `version` = 3.
- A flipped byte inside `data/0` is refused by the CRC check on read.
- A **memoized** pickle, hand-built the way `torch.save`'s own pickler emits
  (`BINPUT`, `BINGET`, `LONG_BINPUT`, `LONG_BINGET`), is read correctly; the
  port's writer never emits those opcodes, so the golden path needed its own
  coverage.
- The canonical JSON equals the `json.dumps(..., sort_keys=True,
  separators=(",", ":"))` form byte for byte, and
  `fingerprint(FIXED_PROMPT)` is
  `d37b38a96aa8f851cdb2782a3ca403e1f7d1328762a97da53c43e46a2b1b1e0e`.

The fingerprint is cross-checked by an independent implementation:
`sha256sum` over the exact 594 canonical bytes returns the same digest, and the
same check over the Spark prompt's canonical bytes returns `79ca00d1…364e2`,
the value the Spark's run reported.

The archive is cross-checked by an independent reader: `unzip -t` reports
`No errors detected` on all seven records of a written cache, and `unzip -l`
lists them with the expected lengths. Our own reader is not the only thing that
accepts the file.

`tests/h3_prompt_cache_gate`, host-only, on **both** hosts (sm_89 workstation
and sm_121 Spark):

    FINGERPRINT d37b38a9…1e0e
    CONTAINER   archive=<name> bytes=12585868 records=7 version=3
    CONTEXT     video 1,1024,4096 torch.bfloat16 8388608 sha256 f60456a0…ba81
    CONTEXT     audio 1,1024,2048 torch.bfloat16 4194304 sha256 2b8a9b85…7d2d
    ROUNDTRIP   video=same audio=same fingerprint=same
    RULE        twelve mutations, each refused, one per validate_payload rule
    RULE        valid_payload accepted rules=12
    PROMPT-CACHE-OK

The two synthetic contexts hash identically on both hosts, which is the one
cross-host bit check the cache layer can carry: the container and the payload
rules are host-independent host code.

The Spark ran the real chain into the file (`--cache` on the connector gate):

    TOKENS    width=1024 vocabulary=262144 bos=2 padding=1013
    CONNECTOR rows=1024 padding=1013 tensors=262 bytes=6344455168 load=1.205 run=0.411
    STAGE     video fnv=dc2e26ddc23e46ab, audio fnv=76457d4d47d3785e
    CACHE     path=/home/leandro/ds4-test/prompt-cache.pt rows=1024
              fingerprint=79ca00d1518977c098075f01ee57e3650b731d63a055c6cab83aa45a76a364e2
    GEMMA-CONNECTOR-OK

and then read it back with the port's own reader:

    CONTAINER archive=prompt-cache bytes=12585478 records=7 version=3
    CONTEXT   video sha256=91816f40…e0af, audio sha256=5e1a67b8…8c41
    STATS     characters=48 truncated=false
    PROMPT-CACHE-OK

So a cache written by the port in one process is a valid cache to the port in
another, on a second host, with the recipe's own shapes and the recipe's own
fingerprint. The Spark's contexts are BF16-derived, which
`docs/setup.md:127-129` says is **not** interchangeable with the recipe's INT8
cache; the delivered cache still has to be built from
`ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors`, which the Spark
does not have.

## 5. What is still owed

1. **The one check this unit cannot run.** The writer's file is verified by
   `unzip`, by our own reader, and against the torch sources line by line, but
   it has **not** been fed to `torch.load(weights_only=True)`, because the
   project's rule is that Python is not used to port the project. That is the
   residual risk on the write direction: the acceptance argument is read from
   `serialization.py` and `inline_container.cc`, not observed. A three-line
   check is available on the Spark, which carries a torch 2.14.1+cu130 tree in
   `~/.cache/uv/archive-v0/JRot790VW0whqV6S/torch/`; running it is an operator
   decision, not a silent one.
2. **The golden.** The value half of the gate needs the operator's own cache
   (the fixed prompt, INT8 route) so the port's contexts can be compared
   against values rather than against its own round trip. `--inspect FILE`
   reads such a file with no model and no device and reports every rule plus a
   per-context digest.
3. **The INT8 connector file**, still absent: 21,504,034,224 bytes, sha256
   `2edbdb44…caf92b`. The `distilled` file of the same byte count on the Spark
   is not it.
4. **The decoder route** is unchanged and still the operator's open choice:
   unset (F32, the reference) 110.4 s of FRAMES on job A, `bf16` 30.4 s,
   `bf16i` 24.4 s.

## 6. Reproduction

    cd /data/ds4-dfm-rs
    export PATH=/data/goose/bin:$PATH
    cargo test -p ds4-core --lib torch_file prompt_cache 3<&-
    rm -f tests/h3_prompt_cache_gate && make tests/h3_prompt_cache_gate
    ./tests/h3_prompt_cache_gate
    ./tests/h3_prompt_cache_gate --write /tmp/cache.pt
    unzip -t /tmp/cache.pt
    ./tests/h3_prompt_cache_gate --inspect /tmp/cache.pt

On the Spark (`CUDA_ARCH=sm_121`, `rm -f native/h3/*.o` first), the emit half:

    tests/h3_gemma_connector_gate --gemma <INT8 Gemma4> --connector <bf16 LTX-2.5> \
      --prompt 'A golden retriever running on a beach at sunset.' --cache ~/ds4-test/prompt-cache.pt
    tests/h3_prompt_cache_gate --inspect ~/ds4-test/prompt-cache.pt \
      --prompt 'A golden retriever running on a beach at sunset.'
