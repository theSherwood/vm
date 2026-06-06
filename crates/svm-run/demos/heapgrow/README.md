# heapgrow demo — a guest that grows its own heap via `map`

The first demo to **consume the Memory capability** (§3e/§4): a guest `malloc` that grows the
window into the reserved tail on demand, rather than the fixed bump allocator the other demos use.
This is the §1a differentiator — "large/sparse programs that fight wasm's flat linear memory" —
shown end to end, with the sandboxed output byte-identical to a native `cc` build.

- **`vm_malloc.h`** — a tiny guest `malloc`/`free`/`calloc` whose heap lives at a fixed high base
  (256 MiB, above the ≤64 MiB backed prefix, inside the large reserved window). `malloc` bumps a
  break pointer and commits fresh pages with **`__vm_map`** — the frontend builtin that lowers to
  `cap.call` on the granted Memory handle (`map(offset, len, READ|WRITE)`). `free` is a no-op (a
  bump allocator, the §3d MVP). Sandbox build only (`__chibicc__`); native `cc` uses the real libc.
- **`heapgrow.c`** — allocates eight 128 KiB int blocks (1 MiB total, ~16× the 64 KiB initial
  window), fills/sums/frees each, and prints the running totals.

```sh
cargo run -p svm-run -- crates/svm-run/demos/heapgrow/heapgrow.c
```

The sandboxed run commits ~1 MiB of reserved-tail pages through the Memory cap (interp page map /
JIT real `mprotect`, the kernel demand-paging the physical backing) and produces the same output
as a native build (`demo_heapgrow_matches_native`). It exercises the full path the rest of the
growth work built: powerbox Memory grant → `__vm_map` builtin → `cap.call` → `MprotectWindow`
growth → masked access to the freshly-committed tail.
