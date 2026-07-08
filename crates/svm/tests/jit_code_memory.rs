//! Pin the JIT's code-memory lifecycle: repeated compiles must not grow the process's address
//! space (ISSUES.md I3). cranelift-jit deliberately *leaks* all code memory when a `JITModule`
//! drops, and each compile reserves a 256 MiB relocation arena — before `svm-jit` freed it
//! explicitly on drop (`OwnedJit`), a compile loop leaked ~100 MiB of VA per iteration. On unix
//! overcommit that was silent VA growth; on Windows the arena is eagerly commit-charged
//! (`MEM_RESERVE | MEM_COMMIT`), so the same leak pinned the CI runner's system commit limit
//! within dozens of compiles, and *unrelated* allocations in the test binary then aborted
//! (`memory allocation of N bytes failed` → `0xc0000409`) or window commits failed
//! (`os error 1455`) — the I3 Windows flake family.
//!
//! Linux-only: `/proc/self/status` is the cheap, deterministic observable. The Windows symptom
//! (commit exhaustion) is this same leak seen through eager commit charging, so pinning VA
//! growth here covers both. (The whole file is gated — on other targets even the imports would
//! trip `-D unused-imports`.)
#![cfg(target_os = "linux")]

#[path = "support/irgen.rs"]
mod irgen;

use irgen::{fuzz_one, Gen};

fn vm_size_kib() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = s.lines().find(|l| l.starts_with("VmSize:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
fn repeated_compiles_do_not_grow_address_space() {
    // Warm-up: allocator arenas, lazy runtime setup, thread-local init — growth from these is
    // one-time and must not count against the loop.
    for seed in 0..3u64 {
        let mut g = Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1CE_F00D);
        fuzz_one(&mut g);
    }
    let before = vm_size_kib();
    // 50 differential iterations ≈ 150+ JIT compiles (each `fuzz_one` runs multiple passes).
    // With the leak, this grew ~4.9 GiB; with `OwnedJit` freeing on drop it is 0.
    for seed in 3..53u64 {
        let mut g = Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1CE_F00D);
        fuzz_one(&mut g);
    }
    let growth_mib = (vm_size_kib().saturating_sub(before)) / 1024;
    // Generous bound: even *two* retained 256 MiB arenas would trip it, while incidental
    // allocator/page-cache noise (tens of MiB at most) cannot.
    assert!(
        growth_mib < 512,
        "address space grew {growth_mib} MiB over 50 compile iterations — \
         JIT code memory is leaking again (see OwnedJit / ISSUES.md I3)"
    );
}
