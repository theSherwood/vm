//! **Worked example: a cache / page-fault estimator over memory hooks** (HOOKS.md).
//!
//! This is the educational use case from HOOKS.md made concrete: score a guest program by
//! estimating how cache-friendly its memory-access pattern is. It builds a tiny CPU-memory model —
//! a direct-mapped cache plus first-touch page faults — feeds every guest access into it through
//! `Instance::with_mem_hooks`, and reports hit rate, page faults, and a composite cycle estimate.
//!
//! It runs the *same* number of loads (1024) two ways to show the model discriminates on pattern,
//! not work:
//!   * **sequential** — sum 1024 contiguous `i64`s (8 KiB). Spreads across the cache's sets and
//!     reuses each 64-byte line 8×, so ~7/8 of accesses hit; touches 2 pages.
//!   * **page-strided** — sum 1024 `i64`s each one 4 KiB page apart, wrapping the 64 KiB window.
//!     Every page-aligned address maps to the *same* direct-mapped set, so each access evicts the
//!     last: a textbook conflict-miss thrash — ~100% misses, and all 16 pages fault.
//!
//! The hooks feature adds **zero cost to programs that don't opt in** (the engines are untouched);
//! this consumer pays only on the instrumented run. Backend-independent by the §3 parity invariant —
//! swap `Backend::Bytecode` for `TreeWalk`/`Jit` and the score is identical.
//!
//! Run:  `cargo run -p svm-run --release --example mem_hooks_cache_model`

use svm_run::{instantiate, Backend, MemEvent, MemHookFn, RunConfig};
use svm_text::parse_module;

// ---- The model ----------------------------------------------------------------------------------

const LINE: u64 = 64; // cache line / block size (bytes)
const SETS: u64 = 64; // number of direct-mapped sets → a 4 KiB cache
const PAGE: u64 = 4096; // page size for the first-touch fault model

// Cost weights (arbitrary but ordered like a real memory hierarchy), for the composite estimate.
const CYCLES_HIT: u64 = 1;
const CYCLES_MISS: u64 = 100;
const CYCLES_FAULT: u64 = 10_000;

/// A direct-mapped cache + first-touch page-fault counter. One instance scores one run.
struct MemModel {
    /// `tags[set]` = the block tag currently resident in that set, or `None` if cold.
    tags: Vec<Option<u64>>,
    /// Pages touched so far (a first touch is a page fault).
    seen_pages: Vec<bool>,
    hits: u64,
    misses: u64,
    faults: u64,
}

impl MemModel {
    fn new(window_bytes: u64) -> MemModel {
        MemModel {
            tags: vec![None; SETS as usize],
            seen_pages: vec![false; (window_bytes / PAGE) as usize],
            hits: 0,
            misses: 0,
            faults: 0,
        }
    }

    /// Charge every 64-byte block the byte range `[addr, addr+len)` touches — the granularity a
    /// cache actually works in (one scalar access is usually one block; a bulk op is many).
    fn touch(&mut self, addr: u64, len: u64) {
        if len == 0 {
            return;
        }
        let first = addr / LINE;
        let last = (addr + len - 1) / LINE;
        for block in first..=last {
            let set = (block % SETS) as usize;
            let tag = block / SETS;
            if self.tags[set] == Some(tag) {
                self.hits += 1;
            } else {
                self.misses += 1;
                self.tags[set] = Some(tag);
            }
            // First-touch page fault (independent of the cache).
            let page = (block * LINE / PAGE) as usize;
            if let Some(seen) = self.seen_pages.get_mut(page) {
                if !*seen {
                    *seen = true;
                    self.faults += 1;
                }
            }
        }
    }

    /// Fold one guest memory event into the model. `Copy` touches both its source and destination
    /// spans; `Fill` its destination; every scalar/atomic op its single access.
    fn observe(&mut self, ev: MemEvent) {
        match ev {
            MemEvent::Load { addr, width }
            | MemEvent::Store { addr, width }
            | MemEvent::AtomicLoad { addr, width }
            | MemEvent::AtomicStore { addr, width }
            | MemEvent::AtomicRmw { addr, width }
            | MemEvent::AtomicCmpxchg { addr, width } => self.touch(addr, width as u64),
            MemEvent::Copy { dst, src, len } => {
                self.touch(src, len);
                self.touch(dst, len);
            }
            MemEvent::Fill { dst, len } => self.touch(dst, len),
        }
    }

    fn accesses(&self) -> u64 {
        self.hits + self.misses
    }
    fn hit_rate(&self) -> f64 {
        if self.accesses() == 0 {
            0.0
        } else {
            self.hits as f64 / self.accesses() as f64
        }
    }
    /// The composite "score": estimated cycles spent in the memory hierarchy (lower is better).
    fn estimated_cycles(&self) -> u64 {
        self.hits * CYCLES_HIT + self.misses * CYCLES_MISS + self.faults * CYCLES_FAULT
    }
}

// ---- The guests ---------------------------------------------------------------------------------

/// Sum 1024 contiguous `i64`s (offsets 0, 8, … 8184 — 8 KiB): cache-friendly.
const SEQUENTIAL: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (off: i64, acc: i64) {
  v2 = i64.load off
  v3 = i64.add acc v2
  v4 = i64.const 8
  v5 = i64.add off v4
  v6 = i64.const 8192
  v7 = i64.lt_u v5 v6
  br_if v7 1(v5, v3) 2(v3)
}
block 2 (v8: i64) {
  return v8
  }
}
"#;

/// Sum 1024 `i64`s one 4 KiB page apart, wrapping the 64 KiB window: every access hits the same
/// direct-mapped set → conflict-miss thrash, and all 16 pages fault.
const PAGE_STRIDED: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (off: i64, acc: i64, i: i64) {
  v3 = i64.load off
  v4 = i64.add acc v3
  v5 = i64.const 4096
  v6 = i64.add off v5
  v7 = i64.const 65535
  v8 = i64.and v6 v7
  v9 = i64.const 1
  v10 = i64.add i v9
  v11 = i64.const 1024
  v12 = i64.lt_u v10 v11
  br_if v12 1(v8, v4, v10) 2(v4)
}
block 2 (v13: i64) {
  return v13
  }
}
"#;

/// A finished run's score — the summary read out of the [`MemModel`] after the run.
struct Score {
    accesses: u64,
    hits: u64,
    misses: u64,
    faults: u64,
    hit_rate: f64,
    cycles: u64,
}

/// Instrument `src`, run it, and read the model's score.
fn score(name: &str, src: &str) -> Score {
    use std::sync::{Arc, Mutex};
    let inst = instantiate(parse_module(src).expect("parse")).expect("instantiate");
    // `with_mem_hooks` takes a `Send + Sync` factory (it may build one handler per host), so shared
    // consumer state lives behind `Arc<Mutex<..>>`. Under a multi-vCPU guest the run's host is shared
    // across vCPUs and hook calls are serialized under its lock, so this same shape stays correct —
    // only the *order* events arrive in would then be schedule-dependent (this guest is single-vCPU).
    let model = Arc::new(Mutex::new(MemModel::new(64 * 1024)));
    let for_hook = model.clone();
    let hooked = inst
        .with_mem_hooks(move || -> MemHookFn {
            let m = for_hook.clone();
            Box::new(move |ev| {
                m.lock().unwrap().observe(ev);
                Ok(())
            })
        })
        .expect("with_mem_hooks");

    // Any backend gives the identical event stream (§3 parity); Bytecode is a good default.
    let run = hooked.run(Backend::Bytecode, &RunConfig::default());
    assert!(run.is_ok(), "{name} guest ran: {run:?}");
    // Read the score out under the lock (the hooked instance still holds a handle to the model).
    let m = model.lock().unwrap();
    Score {
        accesses: m.accesses(),
        hits: m.hits,
        misses: m.misses,
        faults: m.faults,
        hit_rate: m.hit_rate(),
        cycles: m.estimated_cycles(),
    }
}

fn main() {
    println!("Cache/page-fault estimate over memory hooks");
    println!(
        "  model: {}-byte lines, {} direct-mapped sets ({} KiB cache), {}-byte pages\n",
        LINE,
        SETS,
        SETS * LINE / 1024,
        PAGE
    );
    println!(
        "{:>14}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}  {:>14}",
        "guest", "accesses", "hits", "misses", "hit-rate", "faults", "est. cycles"
    );
    for (name, src) in [("sequential", SEQUENTIAL), ("page-strided", PAGE_STRIDED)] {
        let m = score(name, src);
        println!(
            "{:>14}  {:>9}  {:>9}  {:>9}  {:>7.1}%  {:>7}  {:>14}",
            name,
            m.accesses,
            m.hits,
            m.misses,
            m.hit_rate * 100.0,
            m.faults,
            m.cycles,
        );
    }
    println!(
        "\nSame 1024 loads each — the estimate discriminates on access *pattern*, not work done.\n\
         A student's program is scored the same way: instrument, run, read the model."
    );
}
