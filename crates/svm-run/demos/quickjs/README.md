# QuickJS — a full JS engine as a breadth target (LLVM.md "Pending work")

Bellard's **QuickJS** (2024-01-13, MIT) driven through the LLVM→SVM-IR on-ramp:
a whole JS interpreter — NaN-boxing, a bytecode VM with computed-goto dispatch,
BigInt (`libbf`), regex (`libregexp`), Unicode tables (`libunicode`) — compiled
to bitcode, translated, verified, and run in the sandbox, byte-identical to the
same sources built natively with `cc`.

This is the densest control-flow / ABI stressor on the candidate list: if it
passes test262, very little of the language surface is left unproven. It is a
**big lift**, tracked as an in-progress target, not a landed capstone.

## Files

- `qjs_eval.c` — the minimal embedding (no `quickjs-libc`, so no ambient OS
  surface): eval a fixed JS program, stringify the result, print it. The
  program exercises recursion, closures (a `sort` comparator), the object/GC
  machinery + `JSON.stringify`, string methods, and `toFixed` float formatting.
- `build_bitcode.sh` — the fetch → compile → `llvm-link` pipeline the test
  automates (fetched-not-vendored, skips cleanly offline).

The QuickJS sources are **not vendored** (MIT, fetched + cached at test time
from `bellard.org`). Core set: `quickjs.c` (~55k lines) + `libregexp.c` +
`libunicode.c` + `cutils.c` + `libbf.c`.

## Spike status — the gap inventory (analogous to Postgres slice BM)

The linked eval build is ~1.9 MB of bitcode. Pushing it through
`cargo run --example try_translate` (in `crates/svm-llvm`) walks the fail-closed
chokepoint and quantifies the remaining work. Two classes, both already solved
for SQLite/Postgres:

1. **Address-taken libm — `js_math_funcs`.** The `Math` object stores raw C
   function pointers (`JS_CFUNC_SPECIAL_DEF("sin", 1, f_f, sin)`, …) in a
   *constant* global table, so a `&sin`/`&fabs`/`&pow`/`&atan2`/… constexpr
   appears in an initializer. The on-ramp lowers these inline for *direct*
   calls but has no funcref for an address-taken one → `Unsupported("constexpr
   reference to @fabs")`. **Fix:** `llvm-link` a real guest libm (openlibm),
   exactly as the Postgres capstone does (LLVM.md slice CO) — a guest def gives
   each name a real funcref. The full set QuickJS takes the address of:
   `fabs floor ceil trunc sqrt sin cos tan asin acos atan atan2 exp log log2
   log10 expm1 log1p sinh cosh tanh asinh acosh atanh cbrt hypot pow`.

2. **The libc waist** (undefined externs, after `-DNDEBUG` clears
   `__assert_fail`). Almost all are on-ramp-synthesized or covered by the
   Postgres shims (`libc_shim.c`/`printf_shim.c`/`os_shim.c`):
   - mem/string: `memchr memcmp bcmp strlen strcat strchr strcmp strcpy strrchr`
   - alloc: `malloc realloc free malloc_usable_size` (the §slice-X sized header)
   - stdio (printf family): `printf fprintf snprintf sprintf vsnprintf fwrite
     fputc putc puts`
   - number parse/format: `__isoc23_strtol strtod lrint`
   - **new / worth attention:** `fesetround` (FP rounding-mode control — SVM
     ops are round-to-nearest; QuickJS toggles it around some conversions),
     time (`clock_gettime gettimeofday localtime_r`), and `pthread_cond_*` /
     `pthread_mutex_*` (unused on a single-threaded eval → stubbable), `abort`.

Ordering (proposed slices): (a) openlibm link + the libc/stdio shim → first
eval runs; (b) `fesetround` semantics + `strtod`/number parity; (c) widen the
JS program, then (d) the `run-test262.c` harness over an embedded slice as the
self-validating suite (the QuickJS analog of SQLite's sqllogictest).

## Running by hand

```sh
# fetch + build the linked bitcode (→ /tmp/qjs_bc/qjs_linked.bc)
./build_bitcode.sh

# native oracle
QJS=/tmp/svm_quickjs_cache/quickjs-2024-01-13
cc -O2 -D_GNU_SOURCE -I"$QJS" qjs_eval.c \
   "$QJS"/{quickjs,libregexp,libunicode,cutils,libbf}.c -lm -o /tmp/qjs_native
/tmp/qjs_native
# → 1,2,3,5,7,8,9 | sumfib=17710 | {"a":1,"b":[true,null,"x"]} | abc | 0.3000

# translate → verify → run in the sandbox (once the gaps above close)
cd ../../../svm-llvm && cargo run --example try_translate -- \
  /tmp/qjs_bc/qjs_linked.bc /tmp/qjs_native
```
