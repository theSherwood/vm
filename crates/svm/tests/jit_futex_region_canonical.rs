//! PROCESS.md S1b/S1c — the **JIT half of the canonical-key futex**. The interpreter already keys a
//! `Backed` page's futex on the region's canonical `(backing, offset)` identity (`futex_region_canonical.rs`);
//! this pins the same for the JIT, whose futex previously keyed on the raw window-absolute address.
//!
//! One guest mints a §13 region, maps it at **two** window offsets (alias A at `0`, alias B at the
//! granule `G`), spawns a thread that spin-`notify`s the shared region byte through **alias A**, and
//! `atomic.wait`s on the *same region byte* through **alias B** with the value left at its expected `0`
//! (so only a `notify` can wake it). The two aliases are different window-absolute addresses backed by
//! the same region page: with canonical keys they produce the same futex key and the waiter wakes
//! (status `0`); without them the `notify` misses and the child spins to a fuel trap. Differential
//! interp==JIT — the interp already passes, so this is the JIT catching up.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent): `create_region` a 64 KiB region via the `AddressSpace` (arg `v0`), query its
/// granule `G` (op 3), map it at window offset `0` (alias A) and again at `G` (alias B) — both covering
/// region byte 0 — spawn the notifier child, then `atomic.wait` on **alias B** (window `G`), expected
/// `0`. A generous 3 s timeout keeps a *regression* (missed wakeup, keys not canonical) a clean
/// timed-out status (`2`) instead of a hang — on success the `notify` wakes it in microseconds and the
/// timeout never elapses. Returns the wait status: `0` iff a `notify` woke it.
///
/// func 1 (child): spin-`notify` **alias A** (window `0`, region byte 0) until it reports a waiter woken
/// (return `7`), bounded to `LIM` iterations so it self-terminates on a regression (return `-1`) rather
/// than spinning forever — the JIT has no fuel bound. `LIM` is far more than the microseconds the parent
/// needs to park, so a real wakeup is never missed on success.
const SRC: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  vlen = i64.const 65536\n\
  vrh = cap.call 5 5 (i64) -> (i64) v0 (vlen)\n\
  vr = i32.wrap_i64 vrh\n\
  vps = cap.call 4 3 () -> (i64) vr ()\n\
  vz = i64.const 0\n\
  vprot = i32.const 3\n\
  vm1 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vr (vz, vz, vlen, vprot)\n\
  vm2 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vr (vps, vz, vlen, vprot)\n\
  vchild = thread.spawn 1 vz vz\n\
  vexp = i32.const 0\n\
  vto = i64.const 3000000000\n\
  vst = i32.atomic.wait vps vexp vto\n\
  vjr = thread.join vchild\n\
  vst64 = i64.extend_i32_u vst\n\
  return vst64\n\
  }\n\
}\n\
func (i64, i64) -> (i64) {\n\
block 0 (vsp: i64, varg: i64) {\n\
  vz = i64.const 0\n\
  br 1(vz)\n\
}\n\
block 1 (cnt: i64) {\n\
  vlim = i64.const 20000000\n\
  vlt = i64.lt_u cnt vlim\n\
  br_if vlt 2(cnt) 3()\n\
}\n\
block 2 (cnt2: i64) {\n\
  va = i64.const 0\n\
  vone = i32.const 1\n\
  vw = atomic.notify va vone\n\
  vzero = i32.const 0\n\
  vgt = i32.lt_u vzero vw\n\
  vinc = i64.const 1\n\
  vnext = i64.add cnt2 vinc\n\
  br_if vgt 4() 1(vnext)\n\
}\n\
block 3 () {\n\
  vneg = i64.const -1\n\
  return vneg\n\
}\n\
block 4 () {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

#[test]
fn notify_through_alias_wakes_waiter_on_the_other_jit() {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    const WIN: usize = 128 << 10;

    for iter in 0..4 {
        // Interpreter reference (already canonical).
        let mut hi = Host::new();
        hi.set_region_factory(svm_run::new_shared_region);
        let ai = hi.grant_address_space(0, WIN as u64);
        let mut fuel = 50_000_000u64;
        let (ir, _) = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(ai)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut hi,
        );
        assert_eq!(
            ir,
            Ok(vec![Value::I64(0)]),
            "iter {iter}: interp — notify through alias A must wake the waiter on alias B (status 0)"
        );

        // JIT: must match — the canonical key makes the two aliases rendezvous.
        let mut hj = Host::new();
        hj.set_region_factory(svm_run::new_shared_region);
        let aj = hj.grant_address_space(0, WIN as u64);
        let (jo, _) = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[aj as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut hj as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[0]),
            "iter {iter}: jit — notify through alias A must wake the waiter on alias B; got {jo:?}"
        );
    }
}
