//! JIT.md Phase 2b/2c: the guest-driven **`Jit` capability** (iface 11, Model A),
//! differentially tested — the same guest program, the same `Host` powerbox setup, and the
//! same submitted blob run on the reference interpreter (eval-loop `invoke` + dispatch
//! `compile`/`release`) and the Cranelift JIT (`cap_thunk` native intercept →
//! `define_extra`/`invoke_extra` on the live `CompiledModule`), asserting equal results,
//! errnos, traps, and final memory (the escape-oracle).
//!
//! The security hinge — `decode_module` + `verify_module` + the memory-match precondition +
//! data/concurrency rejection — is the **shared** [`svm_run::jit_blob_validator`], installed
//! identically on both backends by [`svm_run::grant_jit`], so accept/reject is identical by
//! construction and pinned here against garbage, mismatched, and trapping blobs.

use svm_encode::encode_module;
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{JitOutcome, TrapKind};
use svm_run::{grant_jit, jit_cap_run};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Where the test harness places the blob in guest memory.
const BLOB_OFF: usize = 4096;

/// Encode `src` as the binary blob a guest submits to `Jit.compile`.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// Run `guest_src`'s func 0 on both backends with the `Jit` cap granted (handle = arg 0) and
/// `blob_bytes` seeded at [`BLOB_OFF`]; assert the outcomes agree and return the JIT's view
/// `(outcome, final_mem)` for the caller's expectations.
fn diff_run(guest_src: &str, blob_bytes: &[u8], user_args: &[i64]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(guest_src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    let mut init = vec![0u8; BLOB_OFF + blob_bytes.len()];
    init[BLOB_OFF..].copy_from_slice(blob_bytes);

    // Interpreter run.
    let mut host_i = Host::new();
    let h_i = grant_jit(&mut host_i, &m);
    let mut iargs = vec![Value::I32(h_i)];
    iargs.extend(user_args.iter().map(|&a| Value::I32(a as i32)));
    let mut fuel = 50_000_000u64;
    let (ires, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &iargs,
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );

    // JIT run — a fresh Host configured identically (the grant is deterministic, so the
    // handle value the guest receives is identical too).
    let mut host_j = Host::new();
    let h_j = grant_jit(&mut host_j, &m);
    assert_eq!(
        h_i, h_j,
        "identical powerbox setup must mint identical handles"
    );
    let mut jargs = vec![h_j as i64];
    jargs.extend_from_slice(user_args);
    let (jout, jmem) =
        jit_cap_run(&m, 0, &jargs, &init, DEFAULT_RESERVED_LOG2, &mut host_j).expect("jit run");

    // Differential: result/trap equivalence…
    match (&ires, &jout) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            assert_eq!(vals.len(), slots.len(), "result arity");
            for (v, s) in vals.iter().zip(slots) {
                let iv = match v {
                    Value::I32(x) => *x as i64,
                    Value::I64(x) => *x,
                    other => panic!("scalar result expected, got {other:?}"),
                };
                assert_eq!(iv, *s, "interp {ires:?} != jit {jout:?}");
            }
        }
        (Err(Trap::Unreachable), JitOutcome::Trapped(TrapKind::Unreachable))
        | (Err(Trap::CapFault), JitOutcome::Trapped(TrapKind::CapFault))
        | (Err(Trap::MemoryFault), JitOutcome::Trapped(TrapKind::MemoryFault))
        | (Err(Trap::DivByZero), JitOutcome::Trapped(TrapKind::DivByZero)) => {}
        other => panic!("backends disagree: {other:?}"),
    }
    // …and the escape-oracle: byte-identical final memory.
    assert_eq!(imem, jmem, "final memory must be byte-identical");
    (jout, jmem)
}

/// A guest that compiles the blob then invokes it with `(a, b)`, returning the result:
/// `(jit_handle, a, b) -> invoke(compile(blob), a, b)`.
const COMPILE_INVOKE: &str = "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v6\n}\n";

fn with_len(src: &str, len: usize) -> String {
    src.replace("BLOBLEN", &len.to_string())
}

/// The full Model A loop, differentially: guest submits IR, both backends validate + compile
/// + invoke it over the live window; the invoked code's store is visible in both final
/// memories (byte-identical), and the result crosses back through the cap.call.
#[test]
fn compile_and_invoke_agree_across_backends() {
    // (a, b) -> a + b + 1000, plus a store of 0xAB at window offset 64.
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.const 64\n  v3 = i32.const 171\n  i32.store8 v2 v3\n  v4 = i32.add v0 v1\n  v5 = i32.const 1000\n  v6 = i32.add v4 v5\n  return v6\n}\n");
    let guest = with_len(COMPILE_INVOKE, b.len());
    let (out, mem) = diff_run(&guest, &b, &[7, 35]);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[1042]),
        "{out:?}"
    );
    assert_eq!(mem[64], 0xab, "invoked code writes the live window");
}

/// A guest that only compiles and returns the raw compile result (handle or -errno):
/// `(jit_handle) -> compile(blob)`.
const COMPILE_ONLY: &str = "memory 16\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 4096\n  v2 = i64.const BLOBLEN\n  v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  return v3\n}\n";

/// Garbage bytes are rejected fail-closed (-EINVAL) identically on both backends — the
/// decode/verify gate never lets them near Cranelift.
#[test]
fn garbage_blob_rejected_identically() {
    let garbage = b"not an svm module at all".to_vec();
    let guest = with_len(COMPILE_ONLY, garbage.len());
    let (out, _) = diff_run(&guest, &garbage, &[]);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "{out:?}"
    );
}

/// The memory-match precondition (JIT.md "Security argument"): a *valid, verified* blob whose
/// declared memory differs from the parent window is rejected — on both backends, before any
/// compilation.
#[test]
fn memory_mismatch_rejected_identically() {
    // Valid module, but declares memory 17 while the parent declares 16.
    let b = blob("memory 17\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let guest = with_len(COMPILE_ONLY, b.len());
    let (out, _) = diff_run(&guest, &b, &[]);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "{out:?}"
    );
}

/// A blob with data segments is rejected (it would overwrite live guest memory at define
/// time — JIT.md "Reject, don't apply").
#[test]
fn data_segment_blob_rejected_identically() {
    let b = blob("data 0 \"\\x01\"\nmemory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let guest = with_len(COMPILE_ONLY, b.len());
    let (out, _) = diff_run(&guest, &b, &[]);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "{out:?}"
    );
}

/// A trap inside invoked code is terminal for the domain — identically on both backends
/// (the interp's nested eval propagates the trap; the JIT's trampoline writes the live trap
/// cell and the guest's propagation check unwinds).
#[test]
fn trap_in_invoked_code_terminal_identically() {
    let b = blob(
        "memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  unreachable\n}\n",
    );
    let guest = with_len(COMPILE_INVOKE, b.len());
    let (out, _) = diff_run(&guest, &b, &[1, 2]);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::Unreachable)),
        "{out:?}"
    );
}

/// A *memory fault* inside invoked code (store past the backed extent) detect-and-kills
/// identically: the interp's nested eval faults on its paged Mem; the JIT's nested guard
/// catches the hardware fault at the invoke boundary.
#[test]
fn memory_fault_in_invoked_code_terminal_identically() {
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i64.const 1048584\n  v3 = i32.const 1\n  i32.store8 v2 v3\n  v4 = i32.const 0\n  return v4\n}\n");
    let guest = with_len(COMPILE_INVOKE, b.len());
    let (out, _) = diff_run(&guest, &b, &[1, 2]);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "{out:?}"
    );
}

/// Invoking a forged code handle is an inert-handle trap (`CapFault`), identically.
#[test]
fn forged_code_handle_capfaults_identically() {
    // (jit_handle, a, b) -> invoke(9999, a, b) — never compiled anything.
    let guest = "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 9999\n  v4 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v3, v1, v2)\n  return v4\n}\n";
    let (out, _) = diff_run(guest, &[], &[1, 2]);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::CapFault)),
        "{out:?}"
    );
}

/// `release` revokes the handle: a subsequent `invoke` of it is a `CapFault` — identically
/// (the generation/clear machinery is the same Host table on both backends).
#[test]
fn release_then_invoke_capfaults_identically() {
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let guest_src = "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = cap.call 11 2 (i64) -> (i64) v0 (v5)\n  v7 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v7\n}\n";
    let guest = with_len(guest_src, b.len());
    let (out, _) = diff_run(&guest, &b, &[1, 2]);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::CapFault)),
        "{out:?}"
    );
}

/// Fuzz the `compile` op (JIT.md "Verification approach"): random byte strings and bit-flipped
/// mutations of a *valid* blob, fed through the full guest-side `cap.call compile` on **both**
/// backends. Every input must either mint a handle or return `-EINVAL` — identically — and
/// nothing may crash the host. (Deterministic xorshift so failures reproduce.)
#[test]
fn fuzzed_blobs_fail_closed_identically() {
    let valid = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let mut s: u64 = 0x9e3779b97f4a7c15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for case in 0..64 {
        let bytes: Vec<u8> = if case % 2 == 0 {
            // Arbitrary garbage of varied length.
            (0..(next() % 200) as usize)
                .map(|_| (next() & 0xff) as u8)
                .collect()
        } else {
            // A valid blob with 1–4 bit flips — the near-miss corpus decode/verify must reject
            // (or, occasionally, accept identically if the flip lands in a don't-care).
            let mut b = valid.clone();
            for _ in 0..=(next() % 4) {
                let i = (next() as usize) % b.len();
                b[i] ^= 1 << (next() % 8);
            }
            b
        };
        let guest = with_len(COMPILE_ONLY, bytes.len());
        let (out, _) = diff_run(&guest, &bytes, &[]);
        // diff_run already asserted backend agreement; additionally pin the result shape:
        // a handle (> 0) or -EINVAL, never anything else, never a trap.
        match out {
            JitOutcome::Returned(ref s) if s.len() == 1 && (s[0] > 0 || s[0] == -22) => {}
            other => panic!("case {case}: unexpected outcome {other:?}"),
        }
    }
}

/// Two units compiled in one run, invoked alternately — the accumulating-definitions REPL
/// shape: `(jit, a, b) -> invoke(u1, a, b) + invoke(u2, a, b)` where u1 = a+b, u2 = a*b.
#[test]
fn two_units_interleaved_agree_across_backends() {
    let add = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let mul = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  return v2\n}\n");
    // Both blobs in memory: add at 4096, mul right after it.
    let mut both = add.clone();
    both.extend_from_slice(&mul);
    let guest_src = format!(
        "memory 16\nfunc (i32, i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const {}\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = i64.const {}\n  v7 = i64.const {}\n  v8 = cap.call 11 0 (i64, i64) -> (i64) v0 (v6, v7)\n  v9 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  v10 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v8, v1, v2)\n  v11 = i32.add v9 v10\n  return v11\n}}\n",
        add.len(),
        4096 + add.len(),
        mul.len(),
    );
    let (out, _) = diff_run(&guest_src, &both, &[6, 7]);
    // (6 + 7) + (6 * 7) = 55.
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[55]),
        "{out:?}"
    );
}
