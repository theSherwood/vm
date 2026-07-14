//! **Per-access memory-confinement regression (D63, branchless confinement).**
//!
//! The JIT confinement lowering ([`svm_jit`]'s `mask_addr`) is the security hinge: every guest access
//! must land in `[0, reserved)` or trap `MemoryFault`. D63 made the top-level per-access lowering
//! branchless — `select_spectre_guard(oob, reserved, addr+offset)` redirects an out-of-bounds access
//! to the trailing guard page instead of an explicit `trapnz` — so this pins the observable contract
//! directly, on both backends:
//!   * an **in-bounds** load/store computes the right value, and JIT == interpreter, and
//!   * an **out-of-bounds** load *and* store both trap `MemoryFault` on the JIT (not wrap, not escape).
//!
//! Gated on `clang` (the on-ramp's own dependency); skipped if absent.

use std::process::Command;

use svm_interp::Value;
use svm_jit::{JitOutcome, TrapKind};

/// `sum16` fills a 16-elem static array and returns `A[n]` (a load at a runtime index); `poke` writes
/// `A[n]=7` and returns `A[0]` (a store at a runtime index). Small fixed arrays with a dynamic index —
/// the loop index isn't provably bounded, so every access is confined (not elided), exercising the
/// lowering under test.
const SRC: &str = r#"
static long A[16];
long sum16(long n){ for (int i=0;i<16;i++) A[i]=i*3+1; return A[n]; }
long poke(long n){ A[n]=7; return A[0]; }
"#;

fn build() -> Option<svm_llvm::Translated> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_confine_{}.c", std::process::id()));
    let bc = dir.join(format!("svm_confine_{}.bc", std::process::id()));
    std::fs::write(&c, SRC).expect("write .c");
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("note: skipping mem_confine (clang unavailable)");
        return None;
    }
    let t = svm_llvm::translate_bc_path(&bc).expect("translate confinement probe");
    let _ = std::fs::remove_file(&c);
    let _ = std::fs::remove_file(&bc);
    Some(t)
}

fn export(t: &svm_llvm::Translated, name: &str) -> u32 {
    t.exports
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("`{name}` not exported"))
        .1
}

#[test]
fn in_bounds_load_matches_interpreter() {
    let Some(t) = build() else { return };
    let sp = t.entry_sp as i64;
    let e = export(&t, "sum16");
    // A[3] = 3*3+1 = 10.
    let mut fuel = 10_000_000u64;
    let interp = match svm_interp::run(&t.module, e, &[Value::I64(sp), Value::I64(3)], &mut fuel)
        .expect("interp run")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        other => panic!("expected one i64, got {other:?}"),
    };
    let jit = match svm_jit::compile_and_run(&t.module, e, &[sp, 3]).expect("jit run") {
        JitOutcome::Returned(s) => s[0],
        other => panic!("unexpected JIT outcome {other:?}"),
    };
    assert_eq!(interp, jit, "interp {interp} vs JIT {jit}");
    assert_eq!(jit, 10, "A[3] should be 10");
}

#[test]
fn out_of_bounds_load_traps_memory_fault() {
    let Some(t) = build() else { return };
    let sp = t.entry_sp as i64;
    let e = export(&t, "sum16");
    // Index far past the 16-elem array (and past the window) — must trap, never wrap or escape.
    match svm_jit::compile_and_run(&t.module, e, &[sp, 1 << 28]).expect("jit run") {
        JitOutcome::Trapped(TrapKind::MemoryFault) => {}
        other => panic!("OOB load should MemoryFault, got {other:?}"),
    }
}

#[test]
fn out_of_bounds_store_traps_memory_fault() {
    let Some(t) = build() else { return };
    let sp = t.entry_sp as i64;
    let e = export(&t, "poke");
    // A wild *write* must fault before it can corrupt anything outside the window.
    match svm_jit::compile_and_run(&t.module, e, &[sp, 1 << 28]).expect("jit run") {
        JitOutcome::Trapped(TrapKind::MemoryFault) => {}
        other => panic!("OOB store should MemoryFault, got {other:?}"),
    }
}
