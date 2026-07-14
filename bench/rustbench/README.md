# rustbench — real-program cross-engine perf

Diverse `no_std` Rust workloads (a real program each) run on **svm-jit** vs **Wasmtime-w64** vs
**native**, timed by the large/small-`n` subtraction (min over reps) — the `confine` methodology, but
on real programs (hash-table churn, a bytecode interpreter, a batch sort, a recursive-descent parser,
base64 encode; plus a BFS held back by a bug — see below) instead of confinement micro-kernels.
Driver: `bench/src/bin/rustbench.rs`.

**Why Rust.** A `no_std` + `alloc` program has *zero libc surface* (a bump `#[global_allocator]`
supplies the heap), so it compiles cleanly to every lane with no shim assembly — what made standing up
a real program like Lua impractical here. Each workload is `prelude.rs` (allocator/panic/PRNG)
prepended to `workloads/<name>.rs` (the `run(n) -> i64` logic).

**The honest comparison is `svm÷wt64`** — both LP64, same widths, same Cranelift backend. The `wt/w32`
column is the *flattered* ILP32 comparison (32-bit addressing + free 4 GiB guards) and is shown for
context only.

## Toolchain

The version match matters: rustc must emit **LLVM-18** bitcode (svm-llvm's on-ramp disassembles with
LLVM 18); rustc **1.81** is the last LLVM-18 release. wasm64 is a tier-3 target, so its lane needs
nightly `build-std`.

```
rustup toolchain install 1.81.0                       # LLVM 18 — svm-jit LP64 bitcode + native + wasm32
rustup +1.81.0 target add wasm32-unknown-unknown
rustup toolchain install nightly --component rust-src  # wasm64 via -Z build-std
```

Any missing piece just blanks that column; svm-jit + native need only `1.81.0`.

## Run (from `bench/`)

```
cargo run --release --bin rustbench
```

Sample (this machine, ×native; `svm÷wt64` lower = svm-jit faster):

```
workload    native(ns)   svm-jit    wt/w64    wt/w32   svm÷wt64
hashmap            3.6     1.9x      1.8x      1.3x     ~1.05x
vm                46       2.9x      2.3x      1.3x     ~1.22x
sort             920       2.0x      2.2x      1.3x     ~0.90x
parse            148       2.4x      2.0x      1.5x     ~1.23x
base64            84       3.1x      3.6x      1.6x     ~0.87x
```

svm-jit lands within ~±20% of Wasmtime-w64 across the five — competitive/parity, workload-dependent
(faster on `sort`/`base64`, a bit behind on the branchy interpreter and parser), consistent with the
"as fast as wasm" goal on real programs.

## Correctness (the harness earns its keep)

Every lane's `run(small)` is cross-checked against svm-jit before timing — a MISCOMPILE aborts. Adding
these workloads immediately surfaced a real **svm-jit correctness bug** on rustc-emitted bitcode: a
sixth workload, `workloads/bfs.rs` (grid BFS), returns garbage on svm-jit while native/Wasmtime agree.
It is filed as **ISSUES.md I23** (with a one-screen reproducer) and held out of the active `WORKLOADS`
list until fixed — re-enable it then.

## Adding a workload

Drop `workloads/<name>.rs` (just the `#[no_mangle] pub extern "C" fn run(n: i64) -> i64` + helpers;
call `reset_arena()` first, use `xs(&mut state)` for determinism so every lane agrees on the
checksum) and add `("<name>", small, large)` to `WORKLOADS` in the driver.
