# ds41 engram row cache — 2026-10-14

Branch `feat/ds41-engram-optimize`, commit `387763b` (cache) on top of
`a1e75c9` (dev). Host: DGX Spark / GB10, driver 580.178.04, ds4-server built
from that branch on the Spark. Workload: the 2252-token `promessi_sposi` slice,
128 greedy tokens, `--prefix-reuse off` so both requests re-prefill the whole
prompt, one server boot per arm, two identical requests per arm.

## Why

The engram feed reads a 264-byte row (256 B e4m3 weights + 8 B ue8m0 scales)
per position per engram layer, two preads per row, through a 4096-byte
O_DIRECT alignment. Row ids are n-gram content hashes, so a re-sent prefix asks
for exactly the same rows. Per 2252-token request the feed reads 115,152 rows,
about 30 MB of payload, against a 203 GB table.

## Result

| arm | run | prefill t/s | TTFT ms | decode t/s | row cache |
| --- | --- | --- | --- | --- | --- |
| off (`DS41_FEED_ROW_CACHE_MB=0`) | 1 | 932.6 | 2417.8 | 43.7 | — |
| off | 2 | 946.4 | 2382.6 | 44.0 | — |
| on (1024 MiB cap) | 1 | 898.9 | 2508.1 | 43.7 | 6055 hits / 115152 rows (5.3%) |
| on | 2 | 1092.1 | 2065.0 | 44.2 | 117063 hits / 230400 rows (cumulative 50.8%) |

Run 2 rows served from RAM: (117063 - 6055) / (230400 - 115152) = 96.3%.
Second-request prefill 946.4 -> 1092.1 t/s (+15.4%), TTFT 2382.6 -> 2065.0 ms
(-13.3%, 318 ms). Decode is unchanged within noise (43.7 / 44.0 off against
43.7 / 44.2 on), as expected: its rows are mostly fresh.

Correctness: the generated token stream is byte-identical in all four runs
(md5 of the four completions equal), so a hit reproduces the disk bytes and no
accumulation order moves.

Cost on the first, cold request: prefill 932.6 -> 898.9 t/s (-3.6%), i.e. the
fill of 109,097 rows, repaid by the second request.

## Limits and follow-ups

- Effective size is 544 MiB, not 1024: the set count is rounded down to a power
  of two under the cap. Using the full budget needs a modulo index.
- Conflict misses: 2554 evictions on the first run (2.2% of its rows). Two-way
  associativity should recover about a percent.
- The 264-byte strided payload means every fill touches two or three
  unaligned cache lines; an aligned slot would cut the fill cost.
- Memory: with the model resident, `free -m` showed 9.9 GB free and 11.0 GB
  available before and after the on-arm; the allocation commits pages as rows
  are written, so the resident footprint grows with use up to the cap.
- Untested here: an agent-session workload where KV prefix reuse is active
  (fewer rows are re-read, so the win is smaller) and a session that drops its
  KV checkpoint (the win is the full prefill delta).
