//! The **`VcpuReactor`** — the resumable-`Vcpu`-driven reactor the browser wasm-JIT tier-up rides.
//!
//! It must be a faithful substitute for the one-shot [`bytecode::Reactor`]: the same persistent
//! window (a counter in guest memory survives frame-to-frame) and the same inline `cap.call` host I/O
//! (each frame writes the counter's low byte through a `Stream(Out)` capability). This test drives an
//! identical guest through both and asserts byte-identical stdout — the differential that lets the
//! reactor switch engines. It also exercises the **tier-up seam**: with func 2 (a pure `x+1`) marked
//! eligible, a direct `Call` to it surfaces as [`VcpuEvent::TierUp`]; servicing it on the interpreter
//! (what the browser does with emitted wasm) must produce the *same* output as interpreting it.

use std::sync::{Arc, Mutex};
use svm_interp::{bytecode, Host, Region, StreamRole, Value};
use svm_text::parse_module;

// func 0 `_start(out)` seeds the counter; func 1 `tick(out)` increments it via func 2 (a pure,
// tier-up-eligible `x+1`), stores it back (so it persists), and writes its low byte through `out`.
const SRC: &str = r#"
memory 16
func (i32) -> () {
block 0 (vout: i32) {
  v0 = i64.const 64
  v1 = i64.const 0
  i64.store v1 v0
  return
  }
}
func (i32) -> () {
block 0 (vout: i32) {
  vaddr = i64.const 0
  vc = i64.load vaddr
  vc2 = call 2 (vc)
  i64.store vaddr vc2
  vptr = i64.const 0
  vlen = i64.const 1
  vw = cap.call 0 1 (i64, i64) -> (i64) vout(vptr, vlen)
  return
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v1 = i64.const 1
  vr = i64.add vx v1
  return vr
  }
}
"#;

/// A fresh, zeroed, 8-aligned window `back` (leaked for the test — never freed, so the `Region` borrow
/// is sound). 1 MiB comfortably covers the guest's 64 KiB window.
fn window() -> Arc<Region> {
    let size = 1usize << 20;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked for the process, so the backing outlives every use.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes owned here and never freed.
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

/// Run `frames` ticks through the one-shot `Reactor` (the reference), returning its stdout.
fn run_plain(frames: usize) -> Vec<u8> {
    let m = parse_module(SRC).unwrap();
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    let mut r = bytecode::Reactor::open(&m).expect("open");
    let mut fuel = u64::MAX;
    r.call(0, &[Value::I32(out)], &mut fuel, &mut host)
        .expect("_start");
    for _ in 0..frames {
        r.call(1, &[Value::I32(out)], &mut fuel, &mut host)
            .expect("tick");
    }
    host.stdout
}

/// Run `frames` ticks through the `VcpuReactor`. If `tierup`, mark func 2 eligible and service each
/// TierUp on the interpreter (the native stand-in for the browser's emitted `f2`); returns
/// `(stdout, tierups)`.
fn run_vcpu(frames: usize, tierup: bool) -> (Vec<u8>, u32) {
    let m = parse_module(SRC).unwrap();
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let host = Mutex::new(h);
    let mut r = bytecode::VcpuReactor::open(&m, window(), &host, &[Value::I32(out)]).expect("open");
    if tierup {
        r = r.with_jit_eligible(Arc::from(vec![false, false, true]));
    }
    let mut tierups = 0u32;
    for _ in 0..frames {
        r.frame(1, &[Value::I32(out)], &host, |func, argv| {
            tierups += 1;
            // Emulate the emitted `f{func}(argv...)`: run the pure callee standalone.
            assert_eq!(func, 2, "only func 2 is eligible");
            let mut fuel = u64::MAX;
            let args: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
            match bytecode::compile_and_run(&m, func, &args, &mut fuel).expect("supported") {
                Ok(vals) => Ok(vals
                    .iter()
                    .map(|v| match v {
                        Value::I64(x) => *x,
                        Value::I32(x) => *x as i64,
                        _ => panic!("non-integer result"),
                    })
                    .collect()),
                Err(t) => Err(t),
            }
        })
        .expect("tick");
    }
    let stdout = host.lock().unwrap().stdout.clone();
    (stdout, tierups)
}

#[test]
fn vcpu_reactor_matches_plain_reactor() {
    // The counter persists across frames, so stdout is the increasing byte sequence 'A','B','C',…
    let want = run_plain(6);
    assert_eq!(
        want, b"ABCDEF",
        "reference reactor stdout (persisted counter)"
    );
    let (got, tierups) = run_vcpu(6, false);
    assert_eq!(
        got, want,
        "VcpuReactor (interpreter-only) diverged from Reactor"
    );
    assert_eq!(tierups, 0, "no tier-up when eligibility is unset");
}

#[test]
fn vcpu_reactor_tierup_matches() {
    let want = run_plain(6);
    let (got, tierups) = run_vcpu(6, true);
    assert_eq!(
        got, want,
        "tier-up run diverged from the interpreter reference"
    );
    assert_eq!(tierups, 6, "func 2 must tier up once per frame");
}
