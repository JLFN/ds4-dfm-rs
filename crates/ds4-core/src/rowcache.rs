/* rowcache.rs — a bounded host-side row cache in front of the engram shards.
 *
 * The engram feed reads 264-byte rows (256 B e4m3 weights + 8 B ue8m0 scales)
 * from two 101 GB safetensors shards, one pread pair per row, up to 48 rows
 * per token per round.  Row ids come from an n-gram hash, so a re-sent prefix
 * asks for exactly the same rows: a session's unique rows are tens of MB
 * against a table of hundreds of GB, which makes a small RAM cache in front of
 * the disk the cheapest win on this path.
 *
 * Two-way set associative with a per-way age, a multiply-shift set index and
 * no lock of its own: every caller reaches it through the feed's single
 * `FeedCore` lock, which the worker and the request thread already serialize
 * on, so an entry is never written while another thread reads it.  Direct
 * mapped, this table lost 13 percent of a 16k-token request's rows to index
 * collisions at the same total size, which is why the second way is there.
 *
 * The payload is the row exactly as the preads land it in the destination
 * (`[weights head_dim][scales head_dim/32]`), so a hit is a byte-identical
 * memcpy of what the disk would have returned and no accumulation order moves.
 *
 * Opt-in (`DS41_FEED_ROW_CACHE_MB`, default 0 = off):
 *   DS41_FEED_ROW_CACHE_MB=256   about 20k tokens of unique context
 *   DS41_FEED_ROW_CACHE_MB=1024  about 80k tokens
 * and inserts pause when the host drops below
 * `DS41_FEED_ROW_CACHE_FLOOR_MB` (default 3072 MiB) so a cap set too high
 * degrades to no caching rather than competing with the model for RAM.
 */
use std::ptr;

/// Ways per set.
const WAYS: usize = 2;
/// Opt-in: `0` (the default) leaves the feed reading straight from the shards.
const DEFAULT_CAP_MB: usize = 0;

pub struct RowCache {
    /// `0` marks an empty way; a live key is stored as `key + 1`.
    keys: Vec<u64>,
    /// Per-way last-use tick, the eviction hint.  A `u8` is enough: it only
    /// has to order two ways inside one set.
    ages: Vec<u8>,
    data: Vec<u8>,
    mask: usize,
    stride: usize,
    tick: u8,
    hits: u64,
    misses: u64,
    evictions: u64,
    /// Insertions are refused while `MemAvailable` is below this floor.
    floor_mb: u64,
    /// Checked every this many insertions: `/proc/meminfo` is a syscall pair
    /// and a row insert is 264 bytes.
    check_every: u64,
    inserts: u64,
    stalls: u64,
    refused: bool,
}

/// `MemAvailable` in MiB, or `None` when it cannot be read.
fn mem_available_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

impl RowCache {
    /// `cap_mb` is a hard bound: the set count is rounded down to a power of
    /// two, so the payload never exceeds the request even by one set.
    pub fn new(cap_mb: usize, stride: usize) -> Option<Self> {
        if cap_mb == 0 || stride == 0 {
            return None;
        }
        let want = cap_mb.saturating_mul(1024 * 1024) / stride;
        if want < WAYS * 2 {
            return None;
        }
        let mut sets = 1usize;
        while sets * 2 * WAYS <= want {
            sets *= 2;
        }
        Some(Self {
            keys: vec![0u64; sets * WAYS],
            ages: vec![0u8; sets * WAYS],
            data: vec![0u8; sets * WAYS * stride],
            mask: sets - 1,
            stride,
            tick: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            floor_mb: std::env::var("DS41_FEED_ROW_CACHE_FLOOR_MB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(3072),
            check_every: 4096,
            inserts: 0,
            stalls: 0,
            refused: false,
        })
    }

    pub fn from_env(stride: usize) -> Option<Self> {
        let cap = std::env::var("DS41_FEED_ROW_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_CAP_MB);
        Self::new(cap, stride)
    }

    fn set_of(&self, key: u64) -> usize {
        /* Fibonacci-style multiply; the mask takes the mid bits, which the
         * multiply spreads the row id's low bits into. */
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) as usize & self.mask
    }

    fn touch(&mut self, way: usize) {
        self.tick = self.tick.wrapping_add(1);
        self.ages[way] = self.tick;
    }

    /// Copy one row's bytes into `dst` when the set holds it.
    pub fn get(&mut self, layer: u8, row: u32, dst: *mut u8) -> bool {
        let key = ((layer as u64) << 32) | row as u64;
        let base = self.set_of(key) * WAYS;
        for w in 0..WAYS {
            if self.keys[base + w] == key + 1 {
                unsafe {
                    ptr::copy_nonoverlapping(
                        self.data.as_ptr().add((base + w) * self.stride),
                        dst,
                        self.stride,
                    )
                };
                self.touch(base + w);
                self.hits += 1;
                return true;
            }
        }
        self.misses += 1;
        false
    }

    /// Store one row's bytes; `src` must be the round's destination slot, read
    /// back after the preads completed.  Refused outright while the host is
    /// below the memory floor, so the cache never competes with the model for
    /// the last of the box's RAM.
    pub fn put(&mut self, layer: u8, row: u32, src: *const u8) {
        self.inserts += 1;
        if self.floor_mb > 0 && self.inserts % self.check_every == 0 {
            let ok = mem_available_mb().map_or(true, |mb| mb >= self.floor_mb);
            if !ok && !self.refused {
                self.stalls += 1;
                eprintln!(
                    "ds4: [v41] engram row cache: MemAvailable under {} MiB, inserts paused",
                    self.floor_mb
                );
            } else if ok && self.refused {
                eprintln!("ds4: [v41] engram row cache: MemAvailable recovered, inserts resumed");
            }
            self.refused = !ok;
        }
        if self.refused {
            return;
        }
        let key = ((layer as u64) << 32) | row as u64;
        let base = self.set_of(key) * WAYS;
        /* An empty way first, otherwise the older of the two. */
        let mut victim = base;
        for w in 0..WAYS {
            if self.keys[base + w] == 0 {
                victim = base + w;
                break;
            }
            if self.ages[base + w] < self.ages[victim] {
                victim = base + w;
            }
        }
        if self.keys[victim] != 0 && self.keys[victim] != key + 1 {
            self.evictions += 1;
        }
        unsafe {
            ptr::copy_nonoverlapping(
                src,
                self.data.as_mut_ptr().add(victim * self.stride),
                self.stride,
            )
        };
        self.keys[victim] = key + 1;
        self.touch(victim);
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    pub fn stalls(&self) -> u64 {
        self.stalls
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + self.keys.len() * 8 + self.ages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two rows that land in the same set both survive: the second way is the
    /// whole point of the associativity (direct mapped, one evicted the other).
    #[test]
    fn two_colliding_rows_both_stay() {
        let mut c = RowCache::new(1, 8).expect("cache");
        c.floor_mb = 0;   /* no /proc probe in a unit test */
        let mut seen = std::collections::HashMap::new();
        let mut pair = None;
        for r in 0..200_000u32 {
            let set = c.set_of(r as u64);
            if let Some(prev) = seen.insert(set, r) {
                pair = Some((prev, r));
                break;
            }
        }
        let (a, b) = pair.expect("two rows sharing a set");
        let (va, vb) = ([1u8; 8], [2u8; 8]);
        c.put(0, a, va.as_ptr());
        c.put(0, b, vb.as_ptr());
        let mut out = [0u8; 8];
        assert!(c.get(0, a, out.as_mut_ptr()), "first key still resident");
        assert_eq!(out, va, "first key's bytes are unchanged");
        let mut out2 = [0u8; 8];
        assert!(c.get(0, b, out2.as_mut_ptr()), "second key resident");
        assert_eq!(out2, vb, "second key's bytes are unchanged");
        let (_, _, evictions) = c.stats();
        assert_eq!(evictions, 0, "both fit: no eviction");
    }

    /// The payload never exceeds the cap, rounded down to whole sets.
    #[test]
    fn cap_is_a_hard_bound() {
        let c = RowCache::new(1, 264).expect("cache");
        assert!(c.bytes() <= 1024 * 1024, "{} bytes over a 1 MiB cap", c.bytes());
    }
}
