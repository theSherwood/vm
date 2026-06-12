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
/// `(outcome, final_mem)` for the caller's expectations. Natural table (no `install` room).
fn diff_run(guest_src: &str, blob_bytes: &[u8], user_args: &[i64]) -> (JitOutcome, Vec<u8>) {
    diff_run_t(guest_src, blob_bytes, user_args, 0)
}

/// Like [`diff_run`], but reserve a `2^table_log2`-slot `call_indirect` table on **both**
/// backends (identically) so the guest can `install` units — Model B2 old→new.
fn diff_run_t(
    guest_src: &str,
    blob_bytes: &[u8],
    user_args: &[i64],
    table_log2: u8,
) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(guest_src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    let mut init = vec![0u8; BLOB_OFF + blob_bytes.len()];
    init[BLOB_OFF..].copy_from_slice(blob_bytes);

    // Interpreter run.
    let mut host_i = Host::new();
    let h_i = grant_jit(&mut host_i, &m, table_log2);
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
    let h_j = grant_jit(&mut host_j, &m, table_log2);
    assert_eq!(
        h_i, h_j,
        "identical powerbox setup must mint identical handles"
    );
    let mut jargs = vec![h_j as i64];
    jargs.extend_from_slice(user_args);
    let (jout, jmem) = jit_cap_run(
        &m,
        0,
        &jargs,
        &init,
        DEFAULT_RESERVED_LOG2,
        table_log2,
        &mut host_j,
    )
    .expect("jit run");

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
        | (Err(Trap::IndirectCallType), JitOutcome::Trapped(TrapKind::IndirectCallType))
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

/// The full Model A loop, differentially: guest submits IR, both backends validate, compile, and
/// invoke it over the live window; the invoked code's store is visible in both final memories
/// (byte-identical), and the result crosses back through the cap.call.
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

/// **new→old** (JIT.md Model A): a submitted unit `call_indirect`s back into the *original
/// program's* function table. The parent's func 1 is `(a, b) -> a + b + 5000`, sitting in
/// table slot 1; the unit's entry does `call_indirect slot 1 (a, b)`. On the JIT the unit is
/// lowered against the parent `fn_table`; on the interpreter it runs as a module-1 frame whose
/// indirect call dispatches into module 0 — both reach the parent's func 1 and return the same
/// value. (This was a confirmed backend divergence before slice #1's cross-module dispatch.)
#[test]
fn new_calls_old_via_call_indirect_agrees() {
    // Parent: func 0 = entry (compiles + invokes the blob), func 1 = the indirect target.
    let parent = "memory 16\n\
func (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v6\n}\n\
func (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  v3 = i32.const 5000\n  v4 = i32.add v2 v3\n  return v4\n}\n";
    // Unit entry (i32,i32)->(i32): call_indirect slot 1 with the target's signature → new→old.
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.const 1\n  v3 = call_indirect (i32, i32) -> (i32) v2 (v0, v1)\n  return v3\n}\n");
    let guest = with_len(parent, b.len());
    let (out, _) = diff_run(&guest, &b, &[10, 20]);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[5030]), // 10 + 20 + 5000
        "{out:?}"
    );
}

/// new→old fail-closed: a unit `call_indirect`ing a slot whose signature doesn't match the
/// parent function there traps `IndirectCallType` — identically on both backends.
#[test]
fn new_to_old_signature_mismatch_traps_identically() {
    // Parent func 1 is (i32,i32)->(i32); the unit calls slot 1 with a wrong (i32)->(i32) sig.
    let parent = "memory 16\n\
func (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v5, v1, v2)\n  return v6\n}\n\
func (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n";
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.const 1\n  v3 = call_indirect (i32) -> (i32) v2 (v0)\n  return v3\n}\n");
    let guest = with_len(parent, b.len());
    let (out, _) = diff_run(&guest, &b, &[10, 20]);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::IndirectCallType)),
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

/// The compile quota (JIT.md "Code reclaim", the MVP byte-cap): with a 2-unit budget, a guest's
/// third `compile` is `-ENOMEM` — identically on both backends (the check lives in the shared
/// `Host::jit_compile` gate), and a guest cannot pressure the finite code arena unboundedly.
#[test]
fn compile_quota_enforced_identically() {
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    // Three sequential compiles of the same blob; return the third's result.
    let guest_src = "memory 16\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 4096\n  v2 = i64.const BLOBLEN\n  v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  v4 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  return v5\n}\n";
    let guest = with_len(guest_src, b.len());

    let m = parse_module(&guest).expect("parse guest");
    verify_module(&m).expect("verify guest");
    let mut init = vec![0u8; BLOB_OFF + b.len()];
    init[BLOB_OFF..].copy_from_slice(&b);

    let mut host_i = Host::new();
    let h = grant_jit(&mut host_i, &m, 0);
    host_i.set_jit_quota(2, 1 << 20);
    let mut fuel = 50_000_000u64;
    let (ires, _) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(h)],
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );
    assert_eq!(
        ires,
        Ok(vec![Value::I64(-12)]),
        "interp: third compile is -ENOMEM"
    );

    let mut host_j = Host::new();
    let h = grant_jit(&mut host_j, &m, 0);
    host_j.set_jit_quota(2, 1 << 20);
    let (jout, _) = jit_cap_run(
        &m,
        0,
        &[h as i64],
        &init,
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host_j,
    )
    .expect("jit run");
    assert!(
        matches!(jout, JitOutcome::Returned(ref s) if s == &[-12]),
        "jit: third compile is -ENOMEM, got {jout:?}"
    );
}

/// **old→new via `install`** (JIT.md Model B2): the guest compiles a unit, installs it into the
/// reserved `call_indirect` table (getting a slot index), then **old code** `call_indirect`s
/// that slot to reach the new code. Differentially: the JIT writes the unit's native entry into
/// the fn_table padding; the interpreter registers the unit as a module + fills the same table
/// slot. Both must return the same slot index and the same call result.
#[test]
fn install_then_old_calls_new_agrees() {
    // (jit, a, b) -> slot = install(compile(blob)); call_indirect[slot](a, b).
    let guest_src = "memory 16\nfunc (i32, i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32, v2: i32):\n  v3 = i64.const 4096\n  v4 = i64.const BLOBLEN\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)\n  v7 = i32.wrap_i64 v6\n  v8 = call_indirect (i32, i32) -> (i32) v7 (v1, v2)\n  return v8\n}\n";
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  v3 = i32.const 100\n  v4 = i32.add v2 v3\n  return v4\n}\n");
    let guest = with_len(guest_src, b.len());
    // Reserve a 16-slot table on both backends; the parent has 1 func, so install lands at slot 1.
    let (out, _) = diff_run_t(&guest, &b, &[6, 7], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[142]), // 6 * 7 + 100
        "{out:?}"
    );
}

/// **new→new** (JIT.md Model B2): an *invoked* unit `call_indirect`s an *installed* unit. The
/// guest installs unit A `(a,b)->a+b` at a slot, then invokes unit B whose body
/// `call_indirect[slot](a,b) + 1` reaches A. On the JIT the invoked unit dispatches the live
/// `fn_table` (which install wrote to); the interpreter gives the invoke child a snapshot of the
/// domain table + units — so both reach the installed unit identically.
#[test]
fn invoked_unit_calls_installed_unit_agrees() {
    // (jit, a, b):
    //   slot = install(compile(A));            // A = (a,b)->a+b at slot 1
    //   codeB = compile(B(slot));              // B = (a,b)-> call_indirect[slot](a,b) + 1
    //   return invoke(codeB, a, b);
    let a_blob = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    // B's entry call_indirects slot 1 (where A installs) then adds 1.
    let b_blob = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.const 1\n  v3 = call_indirect (i32, i32) -> (i32) v2 (v0, v1)\n  v4 = i32.const 1\n  v5 = i32.add v3 v4\n  return v5\n}\n");
    // Lay A at 4096, B right after it.
    let mut both = a_blob.clone();
    both.extend_from_slice(&b_blob);
    let guest_src = format!(
        "memory 16\nfunc (i32, i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32, v2: i32):\n  \
         v3 = i64.const 4096\n  v4 = i64.const {}\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)\n  \
         v7 = i64.const {}\n  v8 = i64.const {}\n  v9 = cap.call 11 0 (i64, i64) -> (i64) v0 (v7, v8)\n  \
         v10 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v9, v1, v2)\n  return v10\n}}\n",
        a_blob.len(),
        4096 + a_blob.len(),
        b_blob.len(),
    );
    let (out, _) = diff_run_t(&guest_src, &both, &[6, 7], 4);
    // B: call_indirect slot 1 = A(6,7) = 13; + 1 = 14.
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[14]),
        "{out:?}"
    );
}

/// **slot reclaim via `uninstall`** (JIT.md Model B2): after uninstalling an installed slot, a
/// `call_indirect` of it traps (`IndirectCallType`), and a later `install` reuses the freed
/// slot index — identically on both backends. (Reclaims the slot, not the code memory.)
#[test]
fn uninstall_frees_slot_then_reinstall_reuses_it_agrees() {
    // (jit, a, b):
    //   s1 = install(compile(A));   // A = a+b -> slot 1
    //   uninstall(s1);
    //   s2 = install(compile(B));   // B = a*b -> reuses slot 1
    //   return s2 * 1000 + call_indirect[s2](a, b);   // proves s2 == s1 and dispatches B
    let a_blob = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let b_blob = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  return v2\n}\n");
    let mut both = a_blob.clone();
    both.extend_from_slice(&b_blob);
    let guest_src = format!(
        "memory 16\nfunc (i32, i32, i32) -> (i32) {{\nblock0(v0: i32, v1: i32, v2: i32):\n  \
         v3 = i64.const 4096\n  v4 = i64.const {}\n  v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n  \
         v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)\n  \
         v7 = cap.call 11 4 (i64) -> (i64) v0 (v6)\n  \
         v8 = i64.const {}\n  v9 = i64.const {}\n  v10 = cap.call 11 0 (i64, i64) -> (i64) v0 (v8, v9)\n  \
         v11 = cap.call 11 3 (i64) -> (i64) v0 (v10)\n  \
         v12 = i32.wrap_i64 v11\n  v13 = call_indirect (i32, i32) -> (i32) v12 (v1, v2)\n  \
         v14 = i32.const 1000\n  v15 = i32.mul v12 v14\n  v16 = i32.add v15 v13\n  return v16\n}}\n",
        a_blob.len(),
        4096 + a_blob.len(),
        b_blob.len(),
    );
    // a=6,b=7: s2 must be slot 1 (reused), call B = 6*7 = 42 → 1*1000 + 42 = 1042.
    let (out, _) = diff_run_t(&guest_src, &both, &[6, 7], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[1042]),
        "{out:?}"
    );
}

/// `uninstall` fail-closed: clearing a real module-function slot (0) or an out-of-range slot is
/// `-EINVAL` identically; nothing is cleared.
#[test]
fn uninstall_protects_real_functions_identically() {
    // (jit) -> uninstall(0)  (slot 0 is a real module function — must be rejected).
    let guest_src = "memory 16\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 0\n  v2 = cap.call 11 4 (i64) -> (i64) v0 (v1)\n  return v2\n}\n";
    let (out, _) = diff_run_t(guest_src, &[], &[], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "{out:?}"
    );
}

/// `install` fail-closed: with **no** table reservation there is no padding slot, so `install`
/// returns `-ENOSPC` identically on both backends (and nothing is installed).
#[test]
fn install_full_table_enospc_identically() {
    // (jit) -> install(compile(blob)); return the raw result. Natural table (reserve 0) → full.
    let guest_src = "memory 16\nfunc (i32) -> (i64) {\nblock0(v0: i32):\n  v1 = i64.const 4096\n  v2 = i64.const BLOBLEN\n  v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n  v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)\n  return v4\n}\n";
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.add v0 v1\n  return v2\n}\n");
    let guest = with_len(guest_src, b.len());
    let (out, _) = diff_run_t(&guest, &b, &[], 0);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-28]),
        "{out:?}"
    );
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

/// **Threaded install** (JIT.md §6 #2): the main thread compiles a unit, **spawns a worker
/// thread**, then `install`s the unit and signals readiness through a guest atomic; the worker —
/// already running — `call_indirect`s the **post-spawn-installed** slot. This is the divergence #2
/// named: a per-vCPU/snapshotted table would hide the install from the worker. With the interp's
/// shared atomic `DomainTable` and the JIT's atomic `FnEntry` (release-ordered publication, the
/// visibility carried by the guest's own ready flag), both backends now agree: the worker reaches
/// the installed unit and returns `6 * 7 + 10 = 52`. Compile (the only `finalize`) happens *before*
/// the spawn, so install is the lone concurrent table op — no threaded-`compile` W^X concern.
///
/// `func 0` = main `(jit) -> i32`; `func 1` = worker `(sp, arg) -> i64` (so two real funcs → install
/// lands at slot 2 of the reserved table). Window: `[0]` ready flag, `[4]` slot index, `[8]` result.
///
/// **Platform parity.** Runs on every target with a JIT thread runtime (`fiber_rt`: x86-64 unix,
/// aarch64 unix/macOS, x86-64 Windows), like the other JIT thread tests. Correct on weakly-ordered
/// aarch64 because the install's visibility rides the *guest's own* acquire/release on the ready
/// flag (install stores → ready store-release → ready load-acquire → the worker's dispatch loads),
/// not the dispatch's own load order; the atomic `FnEntry` fields additionally guarantee a racy
/// reader never observes a torn (half-written) code pointer on any platform.
#[test]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn threaded_install_agrees_across_backends() {
    let b = blob("memory 16\nfunc (i32, i32) -> (i32) {\nblock0(v0: i32, v1: i32):\n  v2 = i32.mul v0 v1\n  v3 = i32.const 10\n  v4 = i32.add v2 v3\n  return v4\n}\n");
    let guest_src = concat!(
        // func 0 — main(jit): compile, spawn worker, install, publish slot + ready, join.
        "memory 16\n",
        "func (i32) -> (i32) {\n",
        "block0(v0: i32):\n",
        "  v1 = i64.const 4096\n",
        "  v2 = i64.const BLOBLEN\n",
        "  v3 = cap.call 11 0 (i64, i64) -> (i64) v0 (v1, v2)\n", // code handle
        "  v4 = i64.const 2048\n",                                // worker data-stack base (unused)
        "  v5 = i64.const 0\n",
        "  v6 = thread.spawn 1 v4 v5\n", // spawn worker BEFORE install
        "  v7 = cap.call 11 3 (i64) -> (i64) v0 (v3)\n", // install -> slot (i64)
        "  v8 = i32.wrap_i64 v7\n",
        "  v9 = i64.const 4\n",
        "  i32.store v9 v8\n", // window[4] = slot
        "  v10 = i64.const 0\n",
        "  v11 = i32.const 1\n",
        "  i32.atomic.store v10 v11\n", // window[0] = ready (release)
        "  v12 = thread.join v6\n",     // worker result (i64)
        "  v13 = i32.wrap_i64 v12\n",
        "  return v13\n",
        "}\n",
        // func 1 — worker(sp, arg): spin on ready, then call_indirect[slot](6, 7).
        "func (i64, i64) -> (i64) {\n",
        "block0(v0: i64, v1: i64):\n",
        "  br block1()\n",
        "block1():\n",
        "  v2 = i64.const 0\n",
        "  v3 = i32.atomic.load v2\n", // ready? (acquire)
        "  v4 = i32.const 0\n",
        "  v5 = i32.ne v3 v4\n",
        "  br_if v5 block2() block1()\n", // spin until ready
        "block2():\n",
        "  v6 = i64.const 4\n",
        "  v7 = i32.load v6\n", // slot (visible via the acquire)
        "  v8 = i32.const 6\n",
        "  v9 = i32.const 7\n",
        "  v10 = call_indirect (i32, i32) -> (i32) v7 (v8, v9)\n", // the post-spawn-installed unit
        "  v11 = i64.const 8\n",
        "  i32.store v11 v10\n",
        "  v12 = i64.extend_i32_u v10\n",
        "  return v12\n",
        "}\n",
    );
    let guest = with_len(guest_src, b.len());
    // Reserve a 16-slot table; 2 real funcs → install lands at slot 2.
    let (out, _) = diff_run_t(&guest, &b, &[], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[52]), // 6 * 7 + 10
        "worker must reach the post-spawn install on both backends: {out:?}"
    );
}
