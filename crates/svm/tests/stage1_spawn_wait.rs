//! Stage 1 (STAGE1.md) — the **`posix_spawn` + `wait`** core for the shell, differentially on the
//! interpreter and the JIT. A parent "shell" seeds an `argv` token into a child's carve, spawns the
//! child as a separate host-verified `Module` via `Instantiator.instantiate_module` (op 5), `join`s
//! (op 1), and the child's return — a function of the seeded bytes — is the parent's result. The
//! child also writes its output back into the carve, which the parent (seeing the §14 nested-carve
//! superset) can read: the "child produced output, parent collected it" path a shell needs.
//!
//! This pins the Stage-1 ABI without any shell yet: argv is delivered by seeding the child's window
//! before spawn (the carve is not zeroed — the child runs over the parent's shared backing, and a
//! command with no data segment at those offsets sees the seed), and the exit status is `join`'s
//! `i64`. Everything rides existing, fuzzed substrate — no confinement-path changes.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10; // parent window: 128 KiB
const CARVE: u64 = 64 << 10; // child carve at 64 KiB (its window is [64 KiB, 128 KiB))
const OUT_OFF: u64 = 8; // child writes its result here (carve-relative)

/// The child "command" module: a 64 KiB window, no data segments (so the parent's seed at low
/// offsets survives spawn). Its entry reads a 4-byte `argv` token the parent seeded at offset 0,
/// sums the bytes, writes the sum back at `OUT_OFF` (the parent reads it via the shared carve), and
/// returns the sum as its exit status. A pure function of the seed — proving argv was delivered.
fn child_src() -> &'static str {
    "memory 16
func (i64) -> (i64) {
block0(v0: i64):
  p0 = i64.const 0
  b0 = i32.load8_u p0
  p1 = i64.const 1
  b1 = i32.load8_u p1
  p2 = i64.const 2
  b2 = i32.load8_u p2
  p3 = i64.const 3
  b3 = i32.load8_u p3
  s01 = i32.add b0 b1
  s23 = i32.add b2 b3
  s = i32.add s01 s23
  po = i64.const 8
  i32.store po s
  sx = i64.extend_i32_u s
  return sx
}
"
}

/// Build a parent "shell" module that seeds the 4-byte `token` into the child's carve at offset 0
/// (window offset `CARVE`), spawns the child module (entry 0, carve at `CARVE`, size_log2 16), joins,
/// and returns the child's exit status. `(Instantiator, Module)` arrive as the entry's two args.
fn parent_src(token: &[u8; 4]) -> String {
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n",)
        })
        .collect();
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block0(vinst: i32, vmod: i32):
{seed}  me = i64.extend_i32_s vmod
  ent = i64.const 0
  off = i64.const {CARVE}
  sl = i64.const 16
  qz = i64.const 0
  ch = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, ent, off, sl, qz)
  r = cap.call 6 1 (i32) -> (i64) vinst (ch)
  return r
}}
"
    )
}

type BothOut = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// Run the parent (seeding `token`) on both backends with identical grants — an `Instantiator` over
/// the whole window and a `Module` capability for the child — and return each backend's result and
/// final window bytes.
fn both(token: &[u8; 4]) -> BothOut {
    let parent = parse_module(&parent_src(token)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src()).expect("parse child");
    verify_module(&child).expect("verify child");
    // A non-zero fill proves the child's read comes from the *seed*, not an incidentally-zero carve.
    let init: Vec<u8> = (0..WIN as u64)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ii = hi.grant_instantiator(0, WIN as u64);
    let mi = hi.grant_module(&child);
    let mut hj = Host::new();
    let ij = hj.grant_instantiator(0, WIN as u64);
    let mj = hj.grant_module(&child);
    assert_eq!(
        (ii, mi),
        (ij, mj),
        "grants must encode identically across backends"
    );

    let mut fuel = 5_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ii), Value::I32(mi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ij as i64, mj as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
        None,
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

/// The byte-sum a correct child returns for a token.
fn want(token: &[u8; 4]) -> i64 {
    token.iter().map(|&b| b as i64).sum()
}

/// Core spawn/wait: the child consumes the seeded `argv` token and returns its byte-sum as the exit
/// status, identically on both backends. The result varies with the seed (not a constant), proving
/// the parent→child argument delivery.
#[test]
fn spawned_child_consumes_seeded_argv_and_returns_status() {
    for token in [b"ABCD", b"wxyz"] {
        let w = want(token);
        let (ir, _im, jo, _jm) = both(token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(w)],
            "interp: child status = byte-sum of seeded argv {token:?}"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[w]),
            "jit: child status = byte-sum of seeded argv {token:?}, got {jo:?}"
        );
    }
}

/// The child's output is readable by the parent through the §14 nested-carve superset: the sum the
/// child wrote at its `OUT_OFF` shows up at `CARVE + OUT_OFF` in the final window on both backends —
/// the "child produced output, parent collected it" path a shell forwards to stdout.
#[test]
fn parent_reads_child_output_from_the_shared_carve() {
    let token = b"ABCD";
    let w = want(token) as u32;
    let (ir, imem, jo, jmem) = both(token);
    assert!(
        ir.is_ok() && matches!(jo, JitOutcome::Returned(_)),
        "both backends ran"
    );

    let at = (CARVE + OUT_OFF) as usize;
    let iout = u32::from_le_bytes(imem[at..at + 4].try_into().unwrap());
    let jout = u32::from_le_bytes(jmem[at..at + 4].try_into().unwrap());
    assert_eq!(iout, w, "interp: child wrote its result into the carve");
    assert_eq!(jout, w, "jit: child wrote its result into the carve");

    // Confinement sanity: nothing below the carve was disturbed by the child (only the parent's own
    // seed writes land there, and those are above CARVE).
    let init: Vec<u8> = (0..WIN as u64)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    for i in 0..CARVE as usize {
        assert_eq!(
            imem[i], init[i],
            "interp: child escaped below its carve at {i}"
        );
        assert_eq!(
            jmem[i], init[i],
            "jit: child escaped below its carve at {i}"
        );
    }
}
