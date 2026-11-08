//! Opt-in request-lifecycle tracing for diagnosing silent streams.
//!
//! Enabled only when `DS4_SERVER_DEBUG=1`. Lines go to stderr, which the
//! launcher redirects into the server log, and carry both a monotonic
//! millisecond timestamp and a wall-clock epoch in milliseconds. The trace is
//! intentionally only request boundaries: accept, queue wait, prefill, first
//! SSE chunk, keep-alive and completion. It must not print prompt or output
//! content.

use std::ffi::OsStr;
use std::fmt;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static ENABLED: OnceLock<bool> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();

pub(crate) fn enabled() -> bool {
    *ENABLED
        .get_or_init(|| std::env::var_os("DS4_SERVER_DEBUG").as_deref() == Some(OsStr::new("1")))
}

pub(crate) fn log(args: fmt::Arguments<'_>) {
    if !enabled() {
        return;
    }
    let start = START.get_or_init(Instant::now);
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    eprintln!(
        "ds4-debug: t={}ms wall_ms={wall} {args}",
        start.elapsed().as_millis()
    );
}
