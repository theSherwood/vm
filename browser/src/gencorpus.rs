//! Host-side **differential corpus** generator. For each guest module it (1) encodes the module to
//! its `svm-encode` binary form under `corpus/`, and (2) computes the **native** bytecode-engine
//! result for a set of args — the ground truth `corpus.mjs` checks the wasm `svm_run` against.
//!
//! The native run here uses the *exact same* `bytecode::compile_and_run` the wasm entry calls, so a
//! mismatch isolates a wasm-compilation / sandbox effect (not an engine difference). The repo
//! already gates the bytecode engine against the tree-walker oracle (`bytecode_diff.rs`); this gates
//! the *wasm build of it* against the native build.
//!
//! Status codes mirror `lib.rs`: 0 OK · 2 UNSUPPORTED (`None`) · 3 TRAP (`Err`) · 4 BAD_RESULT.

use std::io::Write;

use svm_browser::{
    capture_exec, durable_run, dynlink_exec, instantiate_exec, jit_exec, powerbox_exec,
    reflect_exec, region_exec,
};
use svm_durable::{
    init_durable_window, transform_module, write_state, STATE_NORMAL, STATE_REWINDING,
    STATE_UNWINDING,
};
use svm_interp::{bytecode, Value};

// Three op-family kernels lifted verbatim from `crates/svm/tests/bytecode_diff.rs` (known parseable
// and engine-supported), plus a divide-by-zero trap kernel.
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

const CALL: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br block1(v7, v10, v12)
block3(v13: i64):
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.add v0 v1
  return v2
}
"#;

const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"#;

const DIVTRAP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.div_s v0 v1
  return v2
}
"#;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× — total exactly 4000 on every interleaving.
// Lifted from `crates/svm/tests/bytecode_threads.rs`; exercises `thread.spawn`/`join` + atomics on
// the bytecode engine's cooperative `drive` (the browser concurrency model). Takes no args.
const THREADS: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

// Threads **tier-up** (BROWSER.md § "wasm-JIT tier", per-Worker JIT): the 4000 counter kernel, but
// each worker adds `leaf(500)` computed in a pure i64 leaf (func 2) — a hot compute loop counting to
// 500. The guest keeps running on the resumable interpreter (it drives `thread.spawn`/`join` +
// atomics), and each worker's direct `call 2` *tiers up* onto the emitted `f2` on its own Worker.
// func 2 is reachable **only** through `thread.spawn`, so `compile_module_tierup` (not the func-0
// whole-module path) is what emits it. Result 4000 on every interleaving (asserted in the JS host,
// like the plain threads kernel); tiers up once per worker (8×).
const THREADS_TIERUP: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = call 2 (v0)
  v2 = i64.const 0
  v3 = i64.atomic.rmw.add v2 v1
  v4 = i64.const 0
  return v4
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.lt_s v3 v2
  br_if v4 block2(v2, v3) block3(v3)
block2(v5: i64, v6: i64):
  v7 = i64.const 1
  v8 = i64.add v6 v7
  br block1(v5, v8)
block3(v9: i64):
  return v9
}
"#;

// Futex handoff: the root spawns a producer, then `memory.wait`s on the flag at mem[0]; the producer
// writes the payload at mem[8], sets the flag, and `memory.notify`s it. The root then reads the
// payload → 987654 on every interleaving. Exercises the wasm parallel driver's wait/notify path
// (Atomics.wait/notify across Workers), not just spawn/join.
const FUTEX: &str = r#"
memory 16
func () -> (i64) {
block0():
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v0 = i64.const 0
  v1 = i32.const 0
  v2 = i64.const 1000000000
  v5 = i32.atomic.wait v0 v1 v2
  v6 = i64.const 8
  v7 = i64.atomic.load.acquire v6
  v8 = thread.join v4
  return v7
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 8
  v2 = i64.const 987654
  i64.atomic.store.release v1 v2
  v3 = i64.const 0
  v4 = i32.const 1
  i32.atomic.store.release v3 v4
  v5 = i64.const 0
  v6 = i32.const 1
  v7 = atomic.notify v5 v6
  v8 = i64.const 0
  return v8
}
"#;

// ---- powerbox guests: exercise the real capability set (streams / clock / exit) ----------------
// Granted by entry arity (see `powerbox_exec`): 1 Stream(Out) · 2 Stream(In) · 3 Exit ·
// 4 Stream(Err) · 5 Clock. I/O is deterministic (stdout/stderr buffers, monotonic clock), so the
// native result here is an exact ground truth for the wasm `svm_run_pb`.

// `(out, in, exit)`: write a fixed 17-byte greeting to stdout via Stream.write (type 0, op 1).
const PB_HELLO: &str = r#"
memory 16
data 0 "hello, powerbox!\n"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 17
  v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)
  v6 = i32.const 0
  return v6
}
"#;

// `(out, in, exit)`: read up to 256 bytes of stdin (type 0, op 0) into the window, echo them back to
// stdout (type 0, op 1) — a full host→guest→host roundtrip through the buffers.
const PB_ECHO: &str = r#"
memory 16
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 256
  v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)
  v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)
  v7 = i32.const 0
  return v7
}
"#;

// `(out, in, exit, err, clock)`: read the monotonic clock twice (type 2, op 0) and return the delta
// — exactly 1, proving the deterministic strictly-increasing counter works under wasm.
const PB_CLOCK: &str = r#"
func (i32, i32, i32, i32, i32) -> (i64) {
block0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32):
  v5 = i32.const 0
  v6 = cap.call 2 0 (i32) -> (i64) v4(v5)
  v7 = cap.call 2 0 (i32) -> (i64) v4(v5)
  v8 = i64.sub v7 v6
  return v8
}
"#;

// `(out, in, exit)`: call Exit.exit(42) (type 1, op 0) — a non-error trap surfaced as STATUS_EXIT
// with exit code 42; the trailing return is unreachable.
const PB_EXIT: &str = r#"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i32.const 42
  cap.call 1 0 (i32) -> () v2(v3)
  v4 = i32.const 0
  return v4
}
"#;

// `(out, in, exit, err)`: write a 9-byte message to **stderr** (type 0, op 1, on the Err handle) —
// proving role routing (Out → stdout, Err → stderr).
const PB_STDERR: &str = r#"
memory 16
data 0 "warning!\n"
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.const 0
  v5 = i64.const 9
  v6 = cap.call 0 1 (i64, i64) -> (i64) v3(v4, v5)
  v7 = i32.const 0
  return v7
}
"#;

// `(out, in, exit)`: F7 name→handle resolution through the powerbox — `cap.self.resolve "stdout"`
// re-finds the stdout handle **at runtime** (never reading the param slot), then writes through it.
// Works only because the powerbox registers its caps under canonical names (PR #118); a regression in
// either build diverges on stdout. The name "stdout" lives at offset 4096, the payload at 0.
const PB_RESOLVE: &str = r#"
memory 16
data 0 "via resolve\n"
data 4096 "stdout"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 4096
  v4 = i64.const 6
  v5 = cap.self.resolve v3 v4
  v6 = i64.const 0
  v7 = i64.const 12
  v8 = cap.call 0 1 (i64, i64) -> (i64) v5(v6, v7)
  v9 = i32.const 0
  return v9
}
"#;

// Live-import guest (encoded for `live.mjs`, not part of the deterministic corpus): `(console,
// clock)` are host-fn caps (iface 13) the live cdylib bridges to real wasm imports. Writes a 16-byte
// line to stdout via `console.write(stream=0, ptr, len)`, then returns `clock.now()` — so `live.mjs`
// asserts the bytes reached the host import and the host clock value flowed back to the guest.
const LIVE_GUEST: &str = r#"
memory 16
data 0 "live from wasm!\n"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = i64.const 16
  v5 = cap.call 13 1 (i64, i64, i64) -> (i64) v0(v2, v3, v4)
  v6 = cap.call 13 0 () -> (i64) v1()
  return v6
}
"#;

// Large-I/O echo guest (encoded for `corpus.mjs`'s alloc-ABI roundtrip, not the corpus): a 4 MiB
// window, reads up to 4 MiB of stdin and echoes it to stdout — used to push **megabytes** through
// `svm_alloc`ed buffers, well past the old fixed 1 MiB scratch cap.
const BIG_ECHO: &str = r#"
memory 22
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 4194304
  v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)
  v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)
  v7 = i32.const 0
  return v7
}
"#;

// Memory-snapshot guest: the window is seeded with 16 little-endian i64 words; the guest adds `arg`
// to each in place and returns word 0's new value. The captured final image (128 bytes) is the
// interesting output — the "host hands in a buffer, guest transforms it in place" embedder shape.
const CAP_ADDK: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 128
  v5 = i64.lt_u v3 v4
  br_if v5 block2(v2, v3) block3()
block2(v6: i64, v7: i64):
  v8 = i64.load v7
  v9 = i64.add v8 v6
  i64.store v7 v9
  v10 = i64.const 8
  v11 = i64.add v7 v10
  br block1(v6, v11)
block3():
  v12 = i64.const 0
  v13 = i64.load v12
  return v13
}
"#;

// ---- §14 nested child guests (confined sub-window domains) -------------------------------------
// All lifted verbatim from `crates/svm/tests/bytecode_instantiate.rs` (known parseable + engine-
// supported, checked bit-identical to the tree-walker there). Func 0 receives an `Instantiator`
// (iface 6) over `[0, 128 KiB)`; `instantiate` is `cap.call 6 0`, `join` is `cap.call 6 1`.

// Parent instantiates a child in a 4 KiB window at 64 KiB, the child writes 123 at its own offset 7
// (→ shared backing 64 KiB + 7) and returns 42; the parent joins, reads the marker back, returns
// 42*1000 + 123 = 42123 — confined child execution over the shared data plane.
const CHILD_SHARED: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
}
"#;

// Depth-2 VM-in-VM: the child, handed an `Instantiator` over *its* window, instantiates a grandchild
// — confinement composes. The grandchild returns 77, propagated up through two joins.
const CHILD_DEPTH2: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 171
  i32.store8 v2 v3
  v4 = i64.const 2
  v5 = i64.const 2048
  v6 = i64.const 10
  v7 = i64.const 0
  v8 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v4, v5, v6, v7)
  v9 = cap.call 6 1 (i32) -> (i64) v1 (v8)
  return v9
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i64.const 77
  return v3
}
"#;

// A two-arg child receives its starter caps `(Instantiator, AddressSpace)` and uses the AddressSpace
// (iface 5, op 1 = unmap) to decommit the first 16 KiB of its **own** 64 KiB window — a confined
// page op — returning 0. Proves the §14 memory-management capability is attenuated to the child.
const CHILD_ADDRSPACE: &str = r#"memory 18
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i32.wrap_i64 v1
  v3 = i64.const 0
  v4 = i64.const 16384
  v5 = cap.call 5 1 (i64, i64) -> (i64) v2 (v3, v4)
  return v5
}
"#;

// Confinement boundary: a 4 KiB child at offset 128 KiB doesn't fit the 128 KiB holder, so
// `instantiate` returns -EINVAL (-22); the parent returns it without joining.
const CHILD_BADCARVE: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.extend_i32_s v5
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  return v1
}
"#;

// A child trap (`unreachable`) must propagate through `join` as the parent's trap (STATUS_TRAP).
const CHILD_TRAP: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 0
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
}
func (i64) -> (i64) {
block0(v0: i64):
  unreachable
}
"#;

// ---- §12 fibers (cooperative continuation switching) -------------------------------------------
// Lifted verbatim from `crates/svm/tests/bytecode_fibers.rs`. No powerbox needed (cont.* doesn't
// touch the host), so these run through the plain `svm_run0` path. `cont.new`/`cont.resume`/`suspend`.

// Resume delivers arg=7; the fiber returns arg+100 → resumer sees value 107.
const FIB_RUN: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  v4, v5 = cont.resume v2 v3
  return v5
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 100
  v1 = i64.add varg v0
  return v1
}
"#;

// Suspend round-trip: first resume (10) suspends with 11; second resume (20) → fiber returns 25.
// Result 11 + 25 = 36 — repeated park/resume of the same fiber with suspend-result delivery.
const FIB_SUSPEND: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 20
  v7, v8 = cont.resume v2 v6
  v9 = i64.add v5 v8
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
}
"#;

// Two suspends in a fiber, resumed with 3, 4, 5 — exercises multiple switches across one fiber.
const FIB_LOOP: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 3
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 4
  v7, v8 = cont.resume v2 v6
  v9 = i64.const 5
  v10, v11 = cont.resume v2 v9
  v12 = i64.add v5 v8
  v13 = i64.add v12 v11
  return v13
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = suspend varg
  v1 = suspend v0
  v2 = i64.add varg v0
  v3 = i64.add v2 v1
  return v3
}
"#;

// Resuming a never-created fiber handle is an inert FiberFault (a trap) on both engines.
const FIB_FORGED: &str = r#"
func () -> (i64) {
block0():
  v0 = i32.const 99
  v1 = i64.const 5
  v2, v3 = cont.resume v0 v1
  return v3
}
"#;

// The root activation cannot `suspend` (no resumer) — FiberFault on both engines.
const FIB_ROOTSUSPEND: &str = r#"
func () -> (i64) {
block0():
  v0 = i64.const 5
  v1 = suspend v0
  return v1
}
"#;

// ---- §14 coroutines (Instantiator.spawn_coroutine / resume + Yielder.yield) --------------------
// Lifted from `crates/svm/tests/bytecode_coroutines.rs`. Need an Instantiator (iface 6) like nested
// children, so they run through the `svm_run_nested` path. spawn=op 2, resume=op 3; yield=iface 7 op 0.

// Parent spawns a coroutine confined to [64 KiB, 128 KiB), resumes it 3×; the child yields 100, then
// 200+r1, then returns 999+r2 (r1=10, r2=20). Result 100 + 210 + 1019 + RETURNED*1_000_000.
const CORO: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
}
"#;

// Resuming a coroutine handle that was never spawned is an inert CapFault (a trap) on both engines.
const CORO_FORGED: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i32.const 9
  v2 = i64.const 0
  v3, v4 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v1, v2)
  return v4
}
"#;

// ---- §22 guest-JIT (Jit.install + cross-module call_indirect, interpreted) ----------------------
// From `crates/svm/tests/bytecode_dynlink.rs`. The unit (jit_exec's JIT_SERVICE = a*b+100) is
// host-compiled; the guest gets (jit, code, a=6, b=7). iface 11: op 3 install, op 4 uninstall.

// Install the unit (→ table slot), then call_indirect it: 6*7 + 100 = 142.
const JIT_INSTALL: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = i32.wrap_i64 v5
  v7 = call_indirect (i32, i32) -> (i32) v6 (v2, v3)
  return v7
}
"#;

// Install, then uninstall the slot, then call_indirect it → IndirectCall trap (the freed slot).
const JIT_UNINSTALL: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = cap.call 11 4 (i64) -> (i64) v0 (v5)
  v7 = i32.wrap_i64 v5
  v8 = call_indirect (i32, i32) -> (i32) v7 (v2, v3)
  return v8
}
"#;

// ---- §22 guest-JIT **across Workers** (THREADS.md 4c-domain C2) ----------------------------------
// The root gets `(jit, code)` from the shared powerbox, packs both into the single `thread.spawn` arg
// (`(code << 32) | jit`), spawns 8 worker vCPUs, joins them, and returns the shared counter at mem[8].
// Each worker drives the host-compiled unit `JIT_SERVICE(6, 7) = 6*7 + 100 = 142` on the shared
// `Domain` and atomically adds it → 8 × 142 = 1136. Schedule-independent (every worker computes 142),
// so the counter matches the cooperative oracle regardless of how the Workers interleave. `THREADS_JIT`
// macro-expands to the shared root + a worker body; the two kernels differ only in the worker.

// Worker `invoke`s the unit: `cap.call 11 1` (op 1) runs `service(6,7)` over the shared window → 142.
const THREADS_JIT_INVOKE: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  va = i32.const 6
  vb = i32.const 7
  vr = cap.call 11 1 (i64, i32, i32) -> (i32) vjit (vcode, va, vb)
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
}
"#;

// Worker `install`s the unit into the **shared** dispatch table (op 3 → a freshly raced slot) and
// `call_indirect`s its own slot — genuine concurrent installs visible across Workers via the shared
// `Domain`; `service(6,7) = 142`.
const THREADS_JIT_INSTALL: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vslot32 = i32.wrap_i64 vslot
  va = i32.const 6
  vb = i32.const 7
  vr = call_indirect (i32, i32) -> (i32) vslot32 (va, vb)
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
}
"#;

// ---- §14 instantiate **across Workers** (THREADS.md 4c-domain §14-D2) ----------------------------
// Browser-sized: carves are 64 KiB (`slog 16` — the wasm page granularity) inside a 1 MiB window
// (`memory 20`), children at `64 KiB × (i+1)`; handles at `mem[16 + i*4]`. Ground truths (40 / 72 /
// 600) are asserted in the JS host, like the threads kernel's 4000.

// Root `(instantiator) -> sum`: instantiate 8 same-module children (func 1), join, sum. 8 × 5 = 40.
const THREADS_INST: &str = r#"memory 20
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  vone = i64.const 1
  viplus = i64.add vi2 vone
  v64k = i64.const 65536
  voff = i64.mul viplus v64k
  ventry = i64.const 1
  vslog = i64.const 16
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  vinext = i64.add vi2 vone
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  return v1
}
"#;

// Same fan-out, but each child (handed its own Instantiator) instantiates a grandchild (func 2) over
// its **whole** 64 KiB window (slog 16 at off 0 — a carve may equal the authorized range), joins it,
// and returns its value — VM-in-VM-in-VM across three Worker generations. 8 × 9 = 72.
const THREADS_INST_NESTED: &str = r#"memory 20
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  vone = i64.const 1
  viplus = i64.add vi2 vone
  v64k = i64.const 65536
  voff = i64.mul viplus v64k
  ventry = i64.const 1
  vslog = i64.const 16
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  vinext = i64.add vi2 vone
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  vinst = i32.wrap_i64 v0
  ventry = i64.const 2
  voff = i64.const 0
  vslog = i64.const 16
  vquota = i64.const 0
  vgh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vgr = cap.call 6 1 (i32) -> (i64) vinst (vgh)
  return vgr
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 9
  return v1
}
"#;

// The granted "plugin" module for op 5: a 64 KiB window (== the carve, §14 transparency) with a data
// segment `"K"` (75) at offset 0; its entry reads that own data byte and returns it.
const THREADS_INST_UNIT: &str = r#"memory 16
data 0 "K"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
}
"#;

// Root `(instantiator, module) -> sum`: `instantiate_module` the granted module 8 times, join, sum —
// compile + push-to-shared-source + data materialization crossing Workers. 8 × 75 = 600.
const THREADS_INST_MOD: &str = r#"memory 20
func (i32, i32) -> (i64) {
block0(vinst0: i32, vmod0: i32):
  vmod64 = i64.extend_i32_s vmod0
  vi0 = i64.const 0
  br block1(vi0, vinst0, vmod64)
block1(vi: i64, vinst: i32, vmod: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst, vmod) block3(vinst)
block2(vi2: i64, vinst2: i32, vmod2: i64):
  vone = i64.const 1
  viplus = i64.add vi2 vone
  v64k = i64.const 65536
  voff = i64.mul viplus v64k
  ventry = i64.const 0
  vslog = i64.const 16
  vquota = i64.const 0
  vh = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst2 (vmod2, ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  vinext = i64.add vi2 vone
  br block1(vinext, vinst2, vmod2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
"#;

// ---- 4d: host I/O across Workers (the shared powerbox) ------------------------------------------
// The proven schedule-independent 4c-host kernel: root (param = stdout handle) spawns 8 workers, each
// `cap.call`-writes the SAME 5-byte line and bumps a shared counter — so result (8) AND stdout
// ("tick\n" × 8) are schedule-independent. Ground truth asserted in the JS host.
const THREADS_IO: &str = r#"memory 16
data 0 "tick\n"
func (i32) -> (i64) {
block0(v0: i32):
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br block1(v1, vh0)
block1(vi: i64, vhh: i64):
  v2 = i64.const 8
  v3 = i64.lt_u vi v2
  br_if v3 block2(vi, vhh) block3()
block2(vi2: i64, vhh2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vhh2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vhh2)
block3():
  v10 = i64.const 0
  br block4(v10)
block4(vj: i64):
  v11 = i64.const 8
  v12 = i64.lt_u vj v11
  br_if v12 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vh: i64):
  vhandle = i32.wrap_i64 vh
  vptr = i64.const 0
  vlen = i64.const 5
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
}
"#;

// ---- durability (freeze / thaw, single-fiber, IR-driven) ---------------------------------------
// From `crates/svm/tests/bytecode_durable.rs`. A program with two clock reads (each an unwind point);
// the first value is live across the second, so a freeze after the first spills it to the shadow
// stack and a thaw reloads it. base = clock_v + (clock_v + 1). The `svm-durable` transform
// instruments it; gencorpus bakes the *instrumented* module into the corpus.
const DURABLE_SRC: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
}
"#;

// ---- §22 dynamic linking (compile_linked: resolve a named import via a symbol table) -----------
// The guest gets (jit, code, clock); it installs a unit (dynlink_exec's DL_UNIT, which imports
// "clock") and call_indirects it, passing the clock handle. The unit's import was bound to the Clock
// cap by the symbol table → returns clock.now + 777 = 777 (or fail-closed when unlinked).
const DL_GUEST: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.extend_i32_u v1
  v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)
  v5 = i32.wrap_i64 v4
  v6 = call_indirect (i32) -> (i64) v5 (v2)
  return v6
}
"#;

// ---- §13 SharedRegion (host-backed memory aliased into the window) -----------------------------
// From `crates/svm/tests/shared_region.rs`. iface 4: op 0 map(win_off, region_off, len, prot),
// op 2 len, op 3 page_size. Host grants a 64 KiB region as func 0's arg.

// Map the region at offset 0 and again at offset `page_size`, store a marker at 0, load it from the
// second mapping → reads back the marker *iff* both mappings alias the same backing (the §13 promise).
const REGION_ALIAS: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 4 3 () -> (i64) v0 ()
  v2 = i64.const 0
  v3 = i32.const 3
  v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)
  v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)
  v6 = i64.const 81985529216486895
  i64.store v2 v6
  v7 = i64.load v1
  return v7
}
"#;

// Query the region's length (op 2) → the granted 64 KiB = 65536.
const REGION_LEN: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 4 2 () -> (i64) v0 ()
  return v1
}
"#;

// ---- §7 reflection (cap.self.count / cap.self.get) over a fixed 3-cap powerbox -----------------
// Lifted from `crates/svm/tests/bytecode_caps.rs`. Powerbox = Stream(Out) t0, Exit t1, host-fn t13.

// Number of caps the domain holds → 3.
const SELF_COUNT: &str = r#"
func () -> (i32) {
block0():
  v0 = cap.self.count
  return v0
}
"#;

// cap.self.get(idx) → (handle, type_id); sum them so the result depends on both.
const SELF_GET: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1, v2 = cap.self.get v0
  v3 = i32.add v1 v2
  return v3
}
"#;

// ---- tail calls (`return_call` / `return_call_indirect`, O(1) window reuse) --------------------
// Adapted from `crates/svm/tests/bytecode_tailcall.rs` to the single-i64-arg compute shape.

// Tail-recursive factorial accumulator f(n, acc) = n<1 ? acc : f(n-1, acc*n) via `return_call`; entry
// seeds acc=1. Runs in O(1) state (window reuse). Terminates for every arg (n<1 returns acc), so it
// is safe to sweep negatives; large n wraps i64 identically on both engines.
const TAIL_FACT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 1
  return_call 1(v0, v1)
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 1
  v3 = i64.lt_s v0 v2
  br_if v3 block1(v1) block2(v0, v1)
block1(v4: i64):
  return v4
block2(v5: i64, v6: i64):
  v7 = i64.mul v6 v5
  v8 = i64.const -1
  v9 = i64.add v5 v8
  return_call 1(v9, v7)
}
"#;

// `return_call_indirect` through the natural table with x=5: idx 1 = +10 (→15), idx 2 = *2 (→10);
// other indices select func 0 (recurses once then) / out-of-range → IndirectCall trap — all
// identical on both engines.
const TAIL_INDIRECT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  return_call_indirect (i64) -> (i64) v0 (v1)
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 10
  v2 = i64.add v0 v1
  return v2
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 2
  v2 = i64.mul v0 v1
  return v2
}
"#;

// ---- §17 SIMD / v128 (the bytecode engine delegates the v128 long tail to the reference) --------
// Observed via `extract_lane` so the result fits the i64 slot (the natural way a guest consumes one).

// i64x2: splat arg into both lanes, add → [2·arg, 2·arg], extract lane 0 → 2·arg (wraps identically).
const SIMD_I64X2: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
}
"#;

// i32x4: splat the low 32 bits, add lanewise, extract lane 0, sign-extend back to i64.
const SIMD_I32X4: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i32.wrap_i64 v0
  v2 = i32x4.splat v1
  v3 = i32x4.add v2 v2
  v4 = i32x4.extract_lane 0 v3
  v5 = i64.extend_i32_s v4
  return v5
}
"#;

// v128 memory round-trip: splat arg, `v128.store` it, `v128.load` it back, extract lane 1 → arg.
const SIMD_MEM: &str = r#"memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64.const 0
  v128.store v2 v1
  v3 = v128.load v2
  v4 = i64x2.extract_lane 1 v3
  return v4
}
"#;

// ---- §GC conservative root enumeration (`gc.roots`) --------------------------------------------
// Lifted from `crates/svm/tests/bytecode_gc_roots.rs`. `gc.roots vlo vhi vmask vbuf vcap` scans the
// activation for in-range words, writes them (ascending, deduped) to `vbuf`, returns the count. Run
// via the capture path (seed a 4 KiB window, snapshot it back). wasm vs native is the *same* bytecode
// engine, so it is byte-identical here (the soundness-vs-tree-walker caveat doesn't apply).

// In-range constants (one duplicated, one out of range) → roots {4096, 5000}, total 2.
const GC_BASELINE: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 4096
  vb = i64.const 5000
  vc = i64.const 5000
  vd = i64.const 9000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

// Tagged pointer: a tag in the top byte; `vmask` strips it so the bare offset 5000 is in range.
const GC_TAGGED: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 9151314442816852872
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const 72057594037927935
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

/// Lowercase-hex encode (corpus.json carries stdin/stdout/stderr as hex to stay escaping-free).
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---- scalar floating-point (f32/f64) — the one numeric family where wasm↔native can diverge ----
// Each guest reinterprets the i64 arg to an f64, computes, and reinterprets the result back to i64
// **bits**, so the differential is exact — it catches NaN-payload canonicalization and rounding,
// which integer ops can't. Float constants come from `f64.convert_i64_s` (no float-literal parsing).

// add/sub/mul/div + i64→f64 convert: ((a + 3) * 2 - 1) / 2, all in f64.
const FLOAT_ARITH: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = i64.const 3
  v3 = f64.convert_i64_s v2
  v4 = f64.add v1 v3
  v5 = i64.const 2
  v6 = f64.convert_i64_s v5
  v7 = f64.mul v4 v6
  v8 = i64.const 1
  v9 = f64.convert_i64_s v8
  v10 = f64.sub v7 v9
  v11 = f64.div v10 v6
  v12 = i64.reinterpret_f64 v11
  return v12
}
"#;

// sqrt(|a|) — sqrt rounding + abs; sqrt(NaN)/sqrt(-x) exercise NaN canonicalization.
const FLOAT_SQRT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = f64.abs v1
  v3 = f64.sqrt v2
  v4 = i64.reinterpret_f64 v3
  return v4
}
"#;

// min/max/copysign vs 1.0 — the signed-zero and NaN-propagation edges of min/max + sign transfer.
const FLOAT_MINMAX: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = i64.const 1
  v3 = f64.convert_i64_s v2
  v4 = f64.min v1 v3
  v5 = f64.max v4 v3
  v6 = f64.copysign v5 v1
  v7 = i64.reinterpret_f64 v6
  return v7
}
"#;

// f64→f32→f64 round-trip (precision loss, inf/NaN), then saturating f64→i64 (inf→MAX, NaN→0).
const FLOAT_CONVERT: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = f32.demote_f64 v1
  v3 = f64.promote_f32 v2
  v4 = i64.trunc_sat_f64_s v3
  return v4
}
"#;

// comparisons: (a < 1.0) + (a == a) — the second is 0 only for NaN (the classic NaN ≠ NaN test).
const FLOAT_CMP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.reinterpret_i64 v0
  v2 = i64.const 1
  v3 = f64.convert_i64_s v2
  v4 = f64.lt v1 v3
  v5 = f64.eq v1 v1
  v6 = i32.add v4 v5
  v7 = i64.extend_i32_u v6
  return v7
}
"#;

// f64 bit patterns spanning the corners where a backend might diverge.
const FLOAT_ARGS: &[i64] = &[
    0x0000000000000000u64 as i64, // +0.0
    0x8000000000000000u64 as i64, // -0.0
    0x3FF0000000000000u64 as i64, // 1.0
    0xBFF0000000000000u64 as i64, // -1.0
    0x4000000000000000u64 as i64, // 2.0
    0x3FB999999999999Au64 as i64, // 0.1 (rounding)
    0x400921FB54442D18u64 as i64, // π
    0x7FF0000000000000u64 as i64, // +inf
    0xFFF0000000000000u64 as i64, // -inf
    0x7FF8000000000000u64 as i64, // quiet NaN
    0x7FF0000000000001u64 as i64, // signaling NaN
    0x7FEFFFFFFFFFFFFFu64 as i64, // max finite
    0x0000000000000001u64 as i64, // smallest subnormal
];

// Fail-closed UNSUPPORTED path: a module the bytecode engine rejects (`compile_module → None`) —
// `cont.new` (fiber) **and** a coroutine `cap.call 6 2` in one module is an unsupported combination,
// so `svm_run` must return STATUS_UNSUPPORTED, matching native. (Forged handle never runs.)
const UNSUP: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i32.const 0
  v4 = i64.const 1
  v5 = i64.const 0
  v6 = i64.const 12
  v7 = i64.const 0
  v8 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v3 (v4, v5, v6, v7)
  v9 = i64.const 0
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 0
  return v0
}
"#;

/// Args fed to each kernel (all `(i64) -> (i64)`), incl. negatives and a large value.
const ARGS: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000];

fn native(m: &svm_ir::Module, args: &[Value]) -> (i32, i64) {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, 0, args, &mut fuel) {
        None => (2, 0),
        Some(Err(_)) => (3, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (0, *x),
            _ => (4, 0),
        },
    }
}

/// Encode a text module to `corpus/<name>.svmbc` and return the parsed module + file path.
fn emit(name: &str, src: &str) -> (svm_ir::Module, String) {
    let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
    let bytes = svm_encode::encode_module(&m);
    let file = format!("corpus/{name}.svmbc");
    std::fs::File::create(&file)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("write module");
    eprintln!("{name}: {} bytes", bytes.len());
    (m, file)
}

fn main() {
    // Compute corpus — (name, source, nargs): nargs==1 sweeps `ARGS`; nargs==0 runs once, no arg.
    let compute = [
        ("alu", ALU, 1u32),
        ("call", CALL, 1),
        ("mem", MEM, 1),
        ("divtrap", DIVTRAP, 1),
        ("threads", THREADS, 0),
        ("futex", FUTEX, 0),
        ("unsup", UNSUP, 0), // fail-closed: bytecode engine rejects it → STATUS_UNSUPPORTED
    ];
    // Powerbox corpus — (name, source, stdin): each runs once under the real capability set.
    let powerbox = [
        ("pb_hello", PB_HELLO, &b""[..]),
        ("pb_echo", PB_ECHO, &b"ping\n"[..]),
        ("pb_clock", PB_CLOCK, &b""[..]),
        ("pb_exit", PB_EXIT, &b""[..]),
        ("pb_stderr", PB_STDERR, &b""[..]),
        ("pb_resolve", PB_RESOLVE, &b""[..]),
    ];
    std::fs::create_dir_all("corpus").expect("mkdir corpus");

    let mut json = String::from("{\n\"compute\":[\n");
    for (i, (name, src, nargs)) in compute.iter().enumerate() {
        let (m, file) = emit(name, src);
        // args sweep for 1-arg kernels; a single no-arg case otherwise.
        let args: &[i64] = if *nargs == 1 { ARGS } else { &[0] };
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":{nargs},\"cases\":["
        ));
        for (j, &arg) in args.iter().enumerate() {
            let call_args: &[Value] = if *nargs == 1 { &[Value::I64(arg)] } else { &[] };
            let (status, value) = native(&m, call_args);
            // i64s as JSON strings so JS keeps full precision (BigInt).
            json.push_str(&format!(
                "{}{{\"arg\":\"{arg}\",\"status\":{status},\"value\":\"{value}\"}}",
                if j == 0 { "" } else { "," }
            ));
        }
        json.push_str(if i + 1 == compute.len() {
            "]}\n"
        } else {
            "]},\n"
        });
    }
    json.push_str("],\n\"powerbox\":[\n");
    for (i, (name, src, stdin)) in powerbox.iter().enumerate() {
        let (m, file) = emit(name, src);
        // Native ground truth via the *same* `powerbox_exec` the wasm `svm_run_pb` calls.
        let out = powerbox_exec(&m, stdin);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"stdin\":\"{}\",\"status\":{},\
             \"value\":\"{}\",\"exit\":{},\"stdout\":\"{}\",\"stderr\":\"{}\"}}{}",
            hex(stdin),
            out.status,
            out.value,
            out.exit_code,
            hex(&out.stdout),
            hex(&out.stderr),
            if i + 1 == powerbox.len() { "\n" } else { ",\n" },
        ));
    }
    // Capture corpus — a window seeded with 16 i64 words (word i = i*1000), the addk guest run for
    // each arg; the captured final image is the ground truth.
    json.push_str("],\n\"capture\":[\n");
    let (cap_m, cap_file) = emit("cap_addk", CAP_ADDK);
    let mut init = Vec::new();
    for i in 0..16i64 {
        init.extend_from_slice(&(i * 1000).to_le_bytes());
    }
    let cap_args: &[i64] = &[0, 42, -1];
    for (k, &arg) in cap_args.iter().enumerate() {
        let out = capture_exec(&cap_m, &init, arg);
        json.push_str(&format!(
            "  {{\"name\":\"cap_addk\",\"file\":\"{cap_file}\",\"init\":\"{}\",\"arg\":\"{arg}\",\
             \"status\":{},\"value\":\"{}\",\"snapshot\":\"{}\"}}{}",
            hex(&init),
            out.status,
            out.value,
            hex(&out.snapshot),
            if k + 1 == cap_args.len() { "\n" } else { ",\n" },
        ));
    }
    // GC-roots corpus — capture path with a 4 KiB zero window; the guest writes the roots it finds to
    // offset 0 and returns the count, so the snapshot+value is the ground truth (byte-identical here).
    let gcroots = [("gc_baseline", GC_BASELINE), ("gc_tagged", GC_TAGGED)];
    let gc_init = vec![0u8; 4096];
    json.push_str("],\n\"gcroots\":[\n");
    for (i, (name, src)) in gcroots.iter().enumerate() {
        let (m, file) = emit(name, src);
        let out = capture_exec(&m, &gc_init, 0);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"init\":\"{}\",\"arg\":\"0\",\
             \"status\":{},\"value\":\"{}\",\"snapshot\":\"{}\"}}{}",
            hex(&gc_init),
            out.status,
            out.value,
            hex(&out.snapshot),
            if i + 1 == gcroots.len() { "\n" } else { ",\n" },
        ));
    }
    // Nested-child corpus — each runs func 0 with an Instantiator over [0, 128 KiB); the (status,
    // value) is the ground truth (confined child execution, depth, attenuation, boundary, traps).
    let nested = [
        ("child_shared", CHILD_SHARED),
        ("child_depth2", CHILD_DEPTH2),
        ("child_addrspace", CHILD_ADDRSPACE),
        ("child_badcarve", CHILD_BADCARVE),
        ("child_trap", CHILD_TRAP),
        // §14 coroutines reuse the Instantiator grant, so they run on the same `svm_run_nested` path.
        ("coro_roundtrip", CORO),
        ("coro_forged", CORO_FORGED),
    ];
    json.push_str("],\n\"nested\":[\n");
    for (i, (name, src)) in nested.iter().enumerate() {
        let (m, file) = emit(name, src);
        let (status, value) = instantiate_exec(&m);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"status\":{status},\"value\":\"{value}\"}}{}",
            if i + 1 == nested.len() { "\n" } else { ",\n" },
        ));
    }
    // Fiber corpus — §12 cooperative continuations; no powerbox, so run like compute (no-arg, via
    // `svm_run0`). `native()` (deny-all `compile_and_run`) is the ground truth.
    let fibers = [
        ("fib_run", FIB_RUN),
        ("fib_suspend", FIB_SUSPEND),
        ("fib_loop", FIB_LOOP),
        ("fib_forged", FIB_FORGED),
        ("fib_rootsuspend", FIB_ROOTSUSPEND),
    ];
    json.push_str("],\n\"fiber\":[\n");
    for (i, (name, src)) in fibers.iter().enumerate() {
        let (m, file) = emit(name, src);
        let (status, value) = native(&m, &[]);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":0,\
             \"cases\":[{{\"arg\":\"0\",\"status\":{status},\"value\":\"{value}\"}}]}}{}",
            if i + 1 == fibers.len() { "\n" } else { ",\n" },
        ));
    }
    // Compute-style feature sections (1-arg sweep, svm_run): tail calls, SIMD/v128, scalar floats.
    let mut emit_sweep = |section: &str, mods: &[(&str, &str)], sweep: &[i64]| {
        json.push_str(&format!("],\n\"{section}\":[\n"));
        for (i, (name, src)) in mods.iter().enumerate() {
            let (m, file) = emit(name, src);
            json.push_str(&format!(
                "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":1,\"cases\":["
            ));
            for (j, &arg) in sweep.iter().enumerate() {
                let (status, value) = native(&m, &[Value::I64(arg)]);
                json.push_str(&format!(
                    "{}{{\"arg\":\"{arg}\",\"status\":{status},\"value\":\"{value}\"}}",
                    if j == 0 { "" } else { "," }
                ));
            }
            json.push_str(if i + 1 == mods.len() { "]}\n" } else { "]},\n" });
        }
    };
    emit_sweep(
        "tailcall",
        &[("tail_fact", TAIL_FACT), ("tail_indirect", TAIL_INDIRECT)],
        ARGS,
    );
    emit_sweep(
        "simd",
        &[
            ("simd_i64x2", SIMD_I64X2),
            ("simd_i32x4", SIMD_I32X4),
            ("simd_mem", SIMD_MEM),
        ],
        ARGS,
    );
    // Scalar floats swept over NaN/inf/subnormal/rounding bit patterns (the divergence corners).
    emit_sweep(
        "float",
        &[
            ("float_arith", FLOAT_ARITH),
            ("float_sqrt", FLOAT_SQRT),
            ("float_minmax", FLOAT_MINMAX),
            ("float_convert", FLOAT_CONVERT),
            ("float_cmp", FLOAT_CMP),
        ],
        FLOAT_ARGS,
    );
    // Reflection corpus — §7 cap.self.* over the fixed 3-cap powerbox (run via svm_run_reflect).
    // SELF_COUNT takes no arg (→ 3); SELF_GET sweeps cap indices (0,1,2 valid; 3 out of range).
    let reflect: &[(&str, &str, &[i64])] = &[
        ("self_count", SELF_COUNT, &[0]),
        ("self_get", SELF_GET, &[0, 1, 2, 3]),
    ];
    json.push_str("],\n\"reflect\":[\n");
    for (i, (name, src, args)) in reflect.iter().enumerate() {
        let (m, file) = emit(name, src);
        let nargs = m.funcs.first().map_or(0, |f| f.params.len());
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":{nargs},\"cases\":["
        ));
        for (j, &arg) in args.iter().enumerate() {
            let (status, value) = reflect_exec(&m, arg);
            json.push_str(&format!(
                "{}{{\"arg\":\"{arg}\",\"status\":{status},\"value\":\"{value}\"}}",
                if j == 0 { "" } else { "," }
            ));
        }
        json.push_str(if i + 1 == reflect.len() {
            "]}\n"
        } else {
            "]},\n"
        });
    }
    // Guest-JIT corpus — §22 install + cross-module call_indirect (interpreted); svm_run_jit.
    let jit = [
        ("jit_install", JIT_INSTALL),
        ("jit_uninstall", JIT_UNINSTALL),
    ];
    json.push_str("],\n\"jit\":[\n");
    for (i, (name, src)) in jit.iter().enumerate() {
        let (m, file) = emit(name, src);
        let (status, value) = jit_exec(&m);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"status\":{status},\"value\":\"{value}\"}}{}",
            if i + 1 == jit.len() { "\n" } else { ",\n" },
        ));
    }
    // §22 guest-JIT **across Workers** (THREADS.md 4c-domain C2) — emit the kernels for the
    // threads-spawn / browser parallel-JIT proof. Their ground truth (8 × service(6,7) = 1136) is
    // asserted in the JS host (like the threads kernel's 4000), not the corpus JSON: they need the
    // multi-Worker shared powerbox the single-vCPU corpus differential doesn't set up.
    emit("threads_jit_invoke", THREADS_JIT_INVOKE);
    emit("threads_jit_install", THREADS_JIT_INSTALL);
    // §14 instantiate **across Workers** (THREADS.md 4c-domain §14-D2) — the confined-executor-child
    // kernels + the granted module for op 5. Ground truths (40 / 72 / 600) asserted in the JS host.
    emit("threads_inst", THREADS_INST);
    emit("threads_inst_nested", THREADS_INST_NESTED);
    emit("threads_inst_mod", THREADS_INST_MOD);
    emit("threads_inst_unit", THREADS_INST_UNIT);
    // 4d host I/O across Workers — ground truth (result 8, stdout "tick\n"×8) asserted in JS.
    emit("threads_io", THREADS_IO);
    // wasm-JIT **tier-up** across Workers (BROWSER.md § "wasm-JIT tier", per-Worker JIT) — the 4000
    // kernel whose worker compute leaf tiers up onto emitted wasm. Ground truth (4000) asserted in JS.
    emit("threads_tierup", THREADS_TIERUP);
    // Dynamic-linking corpus — §22 compile_linked: resolve the unit's "clock" import via the symbol
    // table (link=1 → 777) or leave it unresolved (link=0 → fail-closed trap). One guest, both cases.
    {
        let (m, file) = emit("dl_clock", DL_GUEST);
        json.push_str("],\n\"dynlink\":[\n");
        for (k, link) in [true, false].iter().enumerate() {
            let (status, value) = dynlink_exec(&m, *link);
            json.push_str(&format!(
                "  {{\"name\":\"dl_clock\",\"file\":\"{file}\",\"link\":{},\"status\":{status},\"value\":\"{value}\"}}{}",
                if *link { 1 } else { 0 },
                if k == 1 { "\n" } else { ",\n" },
            ));
        }
    }
    // SharedRegion corpus — §13 host-backed memory; func 0 gets a 64 KiB region (svm_run_region).
    let region = [("region_alias", REGION_ALIAS), ("region_len", REGION_LEN)];
    json.push_str("],\n\"region\":[\n");
    for (i, (name, src)) in region.iter().enumerate() {
        let (m, file) = emit(name, src);
        let (status, value) = region_exec(&m);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"status\":{status},\"value\":\"{value}\"}}{}",
            if i + 1 == region.len() { "\n" } else { ",\n" },
        ));
    }
    // Durability corpus — instrument the source, then NORMAL run, UNWINDING freeze, and REWINDING
    // thaw (fed the freeze snapshot back). Each case bakes its window + clock + (status, value,
    // snapshot) ground truth; the wasm side runs the *same* instrumented module over the *same* window.
    {
        let src = svm_text::parse_module(DURABLE_SRC).expect("parse durable src");
        let inst = transform_module(&src).expect("durable transform scope");
        let bytes = svm_encode::encode_module(&inst);
        let file = "corpus/durable.svmbc".to_string();
        std::fs::File::create(&file)
            .and_then(|mut f| f.write_all(&bytes))
            .expect("write durable module");
        let clock_v = 1000i64;
        let mut normal = init_durable_window(1 << 17);
        write_state(&mut normal, STATE_NORMAL);
        let mut unwind = init_durable_window(1 << 17);
        write_state(&mut unwind, STATE_UNWINDING);
        let (sn, vn, snap_n, _) = durable_run(&inst, &normal, clock_v);
        let (su, vu, snap_frozen, clk_after) = durable_run(&inst, &unwind, clock_v);
        // Thaw: feed the freeze snapshot back as the window, flipped to REWINDING; clock continues.
        let mut rewind = snap_frozen.clone();
        write_state(&mut rewind, STATE_REWINDING);
        let (sr, vr, snap_r, _) = durable_run(&inst, &rewind, clk_after);
        let cases = [
            ("dur_normal", &normal, clock_v, sn, vn, &snap_n),
            ("dur_freeze", &unwind, clock_v, su, vu, &snap_frozen),
            ("dur_thaw", &rewind, clk_after, sr, vr, &snap_r),
        ];
        json.push_str("],\n\"durable\":[\n");
        for (k, (name, win, clk, status, value, snap)) in cases.iter().enumerate() {
            json.push_str(&format!(
                "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"init\":\"{}\",\"clock\":\"{clk}\",\
                 \"status\":{status},\"value\":\"{value}\",\"snapshot\":\"{}\"}}{}",
                hex(win),
                hex(snap),
                if k + 1 == cases.len() { "\n" } else { ",\n" },
            ));
        }
    }
    json.push_str("]\n}\n");
    std::fs::write("corpus.json", json).expect("write corpus.json");
    eprintln!("wrote corpus.json");

    // Encode the guests validated by harnesses (not the deterministic corpus): the live-import guest
    // (`live.mjs`, host-backed) and the large-I/O echo guest (`corpus.mjs`'s alloc-ABI roundtrip).
    emit("live", LIVE_GUEST);
    emit("bigecho", BIG_ECHO);
}
