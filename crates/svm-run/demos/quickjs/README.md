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
- `libc_shim.c` — the small libc surface the on-ramp neither synthesizes nor
  covers via a reused shim: `fesetround`/`fegetround`, `strtol` (+ the C23
  `__isoc23_strtol` alias), `lrint`, `abort`, `malloc_usable_size`.
- `build_bitcode.sh` — the fetch → compile → `llvm-link` pipeline the test
  automates (fetched-not-vendored, skips cleanly offline).

The rest of the libc/stdio waist is **reused, not rewritten**: the Postgres
guest printf engine (`../postgres/printf_shim.c`, the runtime-`va_list`
`vsnprintf` family) and the correctly-rounded guest `strtod`
(`../strtod/strtod.c`). The native oracle keeps real libc — the shims match it
byte-for-byte — so they go into the guest link only.

The QuickJS sources are **not vendored** (MIT, fetched + cached at test time
from `bellard.org`). Core set: `quickjs.c` (~55k lines) + `libregexp.c` +
`libunicode.c` + `cutils.c` + `libbf.c`.

## Progress + the live gap list

The linked eval build is ~1.9 MB of bitcode. Pushing it through
`cargo run --example try_translate` (in `crates/svm-llvm`) walks the fail-closed
chokepoint one gap at a time. Status:

**DONE — address-taken libm (`js_math_funcs`).** The `Math` object stores raw C
function pointers (`JS_CFUNC_SPECIAL_DEF("sin", 1, f_f, sin)`, …) in a
*constant* global table, so a `&sin`/`&fabs`/`&pow`/… constexpr appears in an
initializer with no funcref → `Unsupported("constexpr reference to @fabs")`.
Closed by `llvm-link`ing guest **openlibm** (the slice BQ/CO mechanism), the
set + the five extras QuickJS adds (`asinh`/`acosh`/`atanh`/`log1p`/`hypot`).

**DONE — struct-constant operands (a translator fix).** QuickJS returns
`JS_EXCEPTION` as a 16-byte `JSValue` struct constant — `ret {i64,i64} {0,6}`.
The on-ramp tracked only *local* aggregates field-wise, so the constant fell to
the scalar path and fail-closed. Fixed in `svm-llvm` (`agg_fields`); test
`struct_constant_return`.

**DONE — the libc/stdio waist.** After `-DNDEBUG` (drops `__assert_fail`), the
undefined externs are the reused printf engine (`vsnprintf`/`snprintf`/… — the
runtime-`va_list` family), the guest `strtod`, and the small `libc_shim.c`
(`fesetround`/`strtol`/`__isoc23_strtol`/`lrint`/`abort`/`malloc_usable_size`).
The mem/string + alloc + non-varargs stdio names (`memcpy`/`fwrite`/`puts`/
`malloc`/…) are on-ramp-synthesized (slices N/O/X), not gaps.

**DONE — dynamic `alloca`.** `JS_CallInternal` allocates its operand stack with
a **runtime-sized `alloca`** (`alloca i8, i64 %n`); the on-ramp now lowers it via
a per-frame `DYN_TOP` running top (bumped by `align16(count·elem)`), and a call in
such a function hands the callee that top so its frame sits above the
variable-length region. Test `dynamic_alloca_runtime_count` (interp == JIT).

**NEXT (blocking) — `llvm.frameaddress`.** With alloca cleared, `JS_CallInternal`
reaches `js_check_stack_overflow`, which reads the stack pointer via
`__builtin_frame_address(0)`. QuickJS assumes a *downward* native stack
(`stack_limit = stack_top - stack_size`), but the SVM data-stack grows *up*, so a
naive `sp` return fires the check on every call — needs a stack-direction-aware
lowering (a downward-mapped `C − sp` proxy, or wiring `JS_UpdateStackTop` to the
window bounds). Its own slice.

**NEXT (semantic) — directed-rounding dtoa.** QuickJS's shortest Number→string
(`js_ecvt1`) toggles `FE_DOWNWARD`/`FE_UPWARD` to find the shortest round-trip
decimal, but the SVM float ops are round-to-nearest only (no rounding-mode op).
`toFixed`/`toPrecision` use `FE_TONEAREST` (which `fesetround` honors), so the
current driver is unaffected; general `String(0.1)` needs a rounding-mode
primitive or a directed-rounding-free guest dtoa.

Then: (a) widen the JS program (regex, BigInt, `try`/`catch` — note JS
exceptions ride QuickJS's own bytecode, not host unwinding); (b) the
`run-test262.c` harness over an embedded slice — the self-validating suite,
QuickJS's analog of SQLite's sqllogictest.

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
