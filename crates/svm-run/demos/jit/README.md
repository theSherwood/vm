# Guest-driven JIT demo

A tiny **bytecode interpreter that JITs itself, entirely inside the sandbox** — the JIT.md
Phase-4 capstone for the guest-driven `Jit` capability (Model A).

`jit_demo.c` defines a toy "calculator bytecode" (an accumulator machine over two inputs) and
runs it two ways:

1. **Interpreted** — a plain C loop, compiled into the sandbox like any guest code.
2. **JITed** — at runtime, the guest walks the same bytecode and *emits serialized SVM IR*
   (the binary `svm-encode` format, built byte-by-byte in its own window), submits the blob
   through `__vm_jit_compile`, and calls the resulting native code with `__vm_jit_invoke2`.

The host side is the `Jit` capability (iface 11): the blob passes the **same**
`decode_module` + `verify_module` gate as any module — plus the memory-match precondition,
data-segment and concurrency rejection — before a single instruction reaches Cranelift, and
the compiled unit joins *this* domain (same window, same powerbox; verification, not
isolation, is the trust boundary). A malformed blob is a clean `-22`; compile quota
exhaustion is `-12`; a trap in JITed code detect-and-kills the whole domain (§5).

It exercises **both** ways the JITed code is reached:

1. **`invoke`** (`__vm_jit_invoke2`) — the interpreter calls the JITed hot loop directly with
   raw `(i64, i64)` args. The shape that accelerates a loop your own code drives.
2. **`install` + a C function pointer** (`__vm_jit_install`, Model B2 old→new) — the same hot
   loop is emitted with the guest ABI (a leading data-SP param), installed into the
   `call_indirect` table, and called like any C function pointer at native speed. Old code
   dispatching freshly-JITed code.

Run it sandboxed:

```sh
cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c
```

Expected output:

```text
emitted 38 bytes of IR
invoke jit(5, 9) = 223 (interp 223)
installed hot(5, 9) = 223 via call_indirect slot 9
98 inputs agree (invoke + installed call_indirect): guest-emitted, host-verified, Cranelift-compiled
```

The differential test (`c_frontend.rs::c_guest_jit_demo`) runs the same program on the
reference interpreter — where `invoke` is a nested evaluation and the installed slot dispatches
through the module-aware table — and on the Cranelift JIT (native code over the live `fn_table`),
asserting identical results and output.

## Threaded JIT (`jit_threads.c`)

The single-threaded capstone's concurrent sibling (JIT.md §6 #2): `NWORKERS` guest threads each
build serialized IR for a **distinct** unit and `__vm_jit_compile` it — so several `Jit.compile`s
run at once — then invoke the native code and check it against a C reference. Each worker keeps its
blob in its own stack buffer and threads the emit cursor explicitly, so the only concurrency the VM
mediates is the cap.call into the host.

Because the guest `thread.spawn`s, the powerbox runs it through the **per-domain serialized
cap-thunk** (a `Mutex<Host>`): a worker's `Jit.compile` (`finalize_definitions`) is serialized
against its siblings' compiles while their *execution* stays fully parallel — cranelift-jit appends
new functions to fresh arena pages and never modifies running code, so a finalize never disturbs an
executing unit. The guest-facing iface 11 is unchanged; the serialization is an internal host detail.

```sh
cargo run -p svm-run -- crates/svm-run/demos/jit/jit_threads.c
```

Expected output — `0` input mismatches across every worker:

```text
0
```

The product-path smoke test is `run.rs::demo_jit_threads_runs` (through the `svm-run` binary, which
engages the locked thunk for a concurrent guest); the interp↔JIT differential for threaded compile
is pinned at the IR level by `jit_cap::threaded_compile_agrees_across_backends`.
