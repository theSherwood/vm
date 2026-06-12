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

Run it sandboxed:

```sh
cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c
```

Expected output:

```text
emitted 38 bytes of IR
interp(5, 9) = 223
jit(5, 9)    = 223
49 inputs agree: guest-emitted, host-verified, Cranelift-compiled
```

The differential test (`c_frontend.rs::c_guest_jit_demo`) runs the same program on the
reference interpreter — where `invoke` is a nested evaluation over the same window — and on
the Cranelift JIT, asserting identical results and output.
