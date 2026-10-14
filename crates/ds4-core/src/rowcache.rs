/* rowcache.rs — a bounded host-side row cache in front of the engram shards.
 *
 * The engram feed reads 264-byte rows (256 B e4m3 weights + 8 B ue8m0 scales)
 * from two 101 GB safetensors shards, one pread pair per row, up to 48 rows
 * per token per round.  Row ids come from an n-gram hash, so a re-sent prefix
 * asks for exactly the same rows: a session's unique rows are tens of MB
 * against a table of hundreds of GB, which makes a small RAM cache in front of
 * the disk the cheapest win on this path.
 *
 * Direct-mapped, one way per set, a multiply-shift hash, no lock of its own:
 * every caller reaches it through the feed's single `FeedCore` lock, which the
 * worker and the request thread already serialize on, so an entry is never
 * written while another thread reads it.
 *
 * The payload is the row exactly as the preads land it in the destination
 * (`[weights head_dim][scales head_dim/32]`), so a hit is a byte-identical
 * memcpy of what the disk would have returned and no accumulation order moves.
 */
use std::ptr;

/// `DS41_FEED_ROW_CACHE_MB` default: 1 GiB, which is about 4M rows, roughly
/// 80k tokens of unique content.  `0` disables the cache.
const DEFAULT_CAP_MB: usize = 1024;

pub struct RowCache {
    /// `0` marks an empty set; a live key is stored as `key + 1`.
    keys: Vec<u64>,
    data: Vec<u8>,
    mask: usize,
    stride: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl RowCache {
    /// `cap_mb` is a hard bound: the set count is rounded down to a power of
    /// two, so the payload never exceeds the request even by one set.
    pub fn new(cap_mb: usize, stride: usize) -> Option<Self> {
        if cap_mb == 0 || stride == 0 {
            return None;
        }
        let want = cap_mb.saturating_mul(1024 * 1024) / stride;
        if want < 2 {
            return None;
        }
        let mut sets = 1usize;
        while sets * 2 <= want {
            sets *= 2;
        }
        Some(Self {
            keys: vec![0u64; sets],
            data: vec![0u8; sets * stride],
            mask: sets - 1,
            stride,
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    pub fn from_env(stride: usize) -> Option<Self> {
        let cap = std::env::var("DS41_FEED_ROW_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_CAP_MB);
        Self::new(cap, stride)
    }

    fn index(&self, key: u64) -> usize {
        /* Fibonacci-style multiply; the mask takes the mid bits, which the
         * multiply spreads the row id's low bits into. */
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) as usize & self.mask
    }

    /// Copy one row's bytes into `dst` when the set holds it.
    pub fn get(&mut self, layer: u8, row: u32, dst: *mut u8) -> bool {
        let key = ((layer as u64) << 32) | row as u64;
        let idx = self.index(key);
        if self.keys[idx] == key + 1 {
            unsafe {
                ptr::copy_nonoverlapping(self.data.as_ptr().add(idx * self.stride), dst, self.stride)
            };
            self.hits += 1;
            return true;
        }
        self.misses += 1;
        false
    }

    /// Store one row's bytes; `src` must be the round's destination slot, read
    /// back after the preads completed.
    pub fn put(&mut self, layer: u8, row: u32, src: *const u8) {
        let key = ((layer as u64) << 32) | row as u64;
        let idx = self.index(key);
        if self.keys[idx] != 0 && self.keys[idx] != key + 1 {
            self.evictions += 1;
        }
        unsafe {
            ptr::copy_nonoverlapping(src, self.data.as_mut_ptr().add(idx * self.stride), self.stride)
        };
        self.keys[idx] = key + 1;
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.evictions)
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + self.keys.len() * 8
    }
}
