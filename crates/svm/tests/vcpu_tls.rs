//! `vcpu.tls.get` / `vcpu.tls.set` — the per-vCPU thread-local register (§12).
//!
//! One i64 of svm per-vCPU state, read/written **at the execution point**, seeded at vCPU creation to
//! a dense id (root = 0, children sequential). In svm's M:N model a fiber migrates between vCPUs
//! (D57), so "which vCPU am I on now" is dynamic and only the runtime knows it — `get` returns the
//! *current* vCPU's word, correct across migration. The guest derives a vCPU id (the seed) and full
//! `__thread`-style TLS (overwrite the word with a per-CPU block pointer) on top. Both backends
//! (soundness/uniformity framing per GC.md §3.2, not exact-value interp↔JIT equality); the threaded
//! tests are gated to the stack-switch substrate like the cross-vCPU `gc.roots` test.

use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Value};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// Run entry 0 on the interpreter, returning its i64 results.
fn interp_i64s(m: &svm_ir::Module) -> Vec<i64> {
    let mut fuel = 200_000_000u64;
    run(m, 0, &[], &mut fuel)
        .unwrap_or_else(|t| panic!("interp trapped: {t:?}"))
        .iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            other => panic!("expected i64, got {other:?}"),
        })
        .collect()
}

/// `vcpu.tls.get`/`set` survive text + binary round-trips and read/write the root vCPU's word: the
/// root is seeded to 0, so the first `get` is 0; after `set 42` the second `get` is 42.
#[test]
fn vcpu_tls_round_trips_and_reads_write_on_the_root() {
    let src = "memory 16\n\
        func () -> (i64, i64) {\n\
        block0():\n\
        \x20 g0 = vcpu.tls.get\n\
        \x20 f = i64.const 42\n\
        \x20 vcpu.tls.set f\n\
        \x20 g1 = vcpu.tls.get\n\
        \x20 return g0 g1\n\
        }\n";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    assert_eq!(
        parse_module(&print_module(&m)),
        Ok(m.clone()),
        "text round-trip"
    );
    assert_eq!(
        decode_module(&encode_module(&m)),
        Ok(m.clone()),
        "binary round-trip"
    );

    assert_eq!(
        interp_i64s(&m),
        vec![0, 42],
        "interp: root seed 0, then set 42"
    );

    #[cfg(any(
        all(unix, target_arch = "x86_64"),
        all(unix, target_arch = "aarch64"),
        all(windows, target_arch = "x86_64")
    ))]
    {
        use svm_jit::{compile_and_run, JitOutcome};
        match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Returned(s) => assert_eq!(s, vec![0, 42], "jit: root seed 0, then set 42"),
            other => panic!("jit did not return: {other:?}"),
        }
    }
}

// ---- Threaded behaviour (shared registry across vCPUs) — substrate-gated like gc_roots. ----------

#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
mod threaded {
    use super::*;
    use svm_jit::{compile_and_run, JitOutcome};

    fn jit_i64s(m: &svm_ir::Module) -> Vec<i64> {
        match compile_and_run(m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Returned(s) => s,
            other => panic!("jit did not return: {other:?}"),
        }
    }

    /// **Distinct seed per vCPU.** Three spawned vCPUs each store their own `vcpu.tls.get()` (the
    /// seed svm assigned) into `slot[idx]`; the root reads the three slots back. Each vCPU's seed must
    /// be distinct and non-zero (the root, seeded 0, is not among the children). Asserted on each
    /// backend (the exact ids are an implementation detail — GC.md §3.2 — so we check structure).
    #[test]
    fn vcpu_tls_seed_is_distinct_per_vcpu() {
        // child(sp, idx): store vcpu.tls.get() at 1024 + idx*8, return 0.
        let src = "memory 16\n\
            func () -> (i64, i64, i64) {\n\
            block0():\n\
            \x20 sp = i64.const 4096\n\
            \x20 a0 = i64.const 0\n\
            \x20 h0 = thread.spawn 1 sp a0\n\
            \x20 a1 = i64.const 1\n\
            \x20 h1 = thread.spawn 1 sp a1\n\
            \x20 a2 = i64.const 2\n\
            \x20 h2 = thread.spawn 1 sp a2\n\
            \x20 j0 = thread.join h0\n\
            \x20 j1 = thread.join h1\n\
            \x20 j2 = thread.join h2\n\
            \x20 s0 = i64.const 1024\n\
            \x20 v0 = i64.load s0\n\
            \x20 s1 = i64.const 1032\n\
            \x20 v1 = i64.load s1\n\
            \x20 s2 = i64.const 1040\n\
            \x20 v2 = i64.load s2\n\
            \x20 return v0 v1 v2\n\
            }\n\
            func (i64, i64) -> (i64) {\n\
            block0(p0: i64, p1: i64):\n\
            \x20 id = vcpu.tls.get\n\
            \x20 eight = i64.const 8\n\
            \x20 off = i64.mul p1 eight\n\
            \x20 base = i64.const 1024\n\
            \x20 addr = i64.add base off\n\
            \x20 i64.store addr id\n\
            \x20 z = i64.const 0\n\
            \x20 return z\n\
            }\n";
        let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
        verify_module(&m).expect("verify");

        let check = |seeds: &[i64], backend: &str| {
            assert!(
                seeds.iter().all(|&s| s != 0),
                "{backend}: every spawned vCPU's seed must be non-zero (root is 0); got {seeds:?}"
            );
            assert!(
                seeds[0] != seeds[1] && seeds[1] != seeds[2] && seeds[0] != seeds[2],
                "{backend}: vCPU seeds must be distinct; got {seeds:?}"
            );
        };
        check(&interp_i64s(&m), "interp");
        check(&jit_i64s(&m), "jit");
    }

    /// **The headline: read at the execution point is correct across fiber migration.** vCPU A (the
    /// root) creates fiber F and resumes it once — F reads `vcpu.tls.get()` (= A's word, 0), records
    /// it, and `suspend`s. The root then spawns vCPU B, handing it F's handle; B `cont.resume`s the
    /// **same** fiber (a cross-vCPU resume = migration), so F continues on B's thread, reads
    /// `vcpu.tls.get()` again (= B's word, ≠ 0), and returns it. The two reads of the *same* fiber
    /// differ — proving the register tracks the *current* vCPU, not the fiber. Both backends share the
    /// fiber registry across vCPUs, so this holds on each (soundness per GC.md §3.2).
    #[test]
    fn vcpu_tls_tracks_current_vcpu_across_migration() {
        let src = "memory 16\n\
            func () -> (i64, i64) {\n\
            block0():\n\
            \x20 fref = ref.func 2\n\
            \x20 fsp = i64.const 4096\n\
            \x20 fh = cont.new fref fsp\n\
            \x20 a0 = i64.const 0\n\
            \x20 st0, val0 = cont.resume fh a0\n\
            \x20 bsp = i64.const 8192\n\
            \x20 bh = thread.spawn 1 bsp fh\n\
            \x20 bret = thread.join bh\n\
            \x20 s1 = i64.const 1024\n\
            \x20 r1 = i64.load s1\n\
            \x20 return r1 bret\n\
            }\n\
            func (i64, i64) -> (i64) {\n\
            block0(p0: i64, p1: i64):\n\
            \x20 ba = i64.const 0\n\
            \x20 bst, bval = cont.resume p1 ba\n\
            \x20 return bval\n\
            }\n\
            func (i64, i64) -> (i64) {\n\
            block0(q0: i64, q1: i64):\n\
            \x20 r1 = vcpu.tls.get\n\
            \x20 s1 = i64.const 1024\n\
            \x20 i64.store s1 r1\n\
            \x20 z = i64.const 0\n\
            \x20 sv = suspend z\n\
            \x20 r2 = vcpu.tls.get\n\
            \x20 return r2\n\
            }\n";
        let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
        verify_module(&m).expect("verify");

        let check = |slots: &[i64], backend: &str| {
            let (read_a, read_b) = (slots[0], slots[1]);
            assert_eq!(
                read_a, 0,
                "{backend}: the first read ran on the root vCPU (seed 0); got {read_a}"
            );
            assert!(
                read_b != 0 && read_b != read_a,
                "{backend}: after migrating to vCPU B the same fiber must read B's word (≠ 0, ≠ A's); \
                 got read_a={read_a}, read_b={read_b}"
            );
        };
        check(&interp_i64s(&m), "interp");
        check(&jit_i64s(&m), "jit");
    }
}
