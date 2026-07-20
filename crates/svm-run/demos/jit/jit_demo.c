// Guest-driven JIT demo (DESIGN.md §22, Model A): a tiny **bytecode interpreter that JITs
// itself** — entirely inside the sandbox.
//
// A toy "calculator bytecode" (a two-input expression machine over `(a, b)`) is first run on a
// plain C interpreter loop. Then the SAME bytecode is compiled at runtime: this program walks
// it and emits serialized SVM IR — the binary `svm-encode` format, built byte-by-byte in guest
// memory — and submits it through the `Jit` capability (`__vm_jit_compile`). The host verifies
// the blob (the same decode+verify gate every module passes) and Cranelift-compiles it into
// THIS domain: same window, same powerbox, no nested sandbox. `__vm_jit_invoke2` then calls the
// native code directly, and the demo checks it agrees with the interpreter on a grid of inputs.
//
// This is the classic JIT-inside-the-sandbox problem WebAssembly handles poorly (immutable
// modules force guests to ship their own interpreter forever, or round-trip to the host for a
// fresh module). Here the guest gets a native fast path without leaving the sandbox's security
// model: a malformed/forged blob is rejected fail-closed (-22), a trap in JITed code
// detect-and-kills the whole domain, and compilation is quota-bounded (-12).
//
// Run it sandboxed:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_demo.c

#include <svm.h>

int write(int fd, char *buf, long n);

static void puts1(char *s) {
  long n = 0;
  while (s[n])
    n++;
  write(1, s, n);
}

static void put_i64(long v) {
  char tmp[24];
  int i = 0;
  if (v < 0) {
    write(1, "-", 1);
    v = -v;
  }
  if (v == 0) {
    write(1, "0", 1);
    return;
  }
  while (v) {
    tmp[i++] = '0' + (v % 10);
    v /= 10;
  }
  while (i)
    write(1, &tmp[--i], 1);
}

// --- the toy calculator bytecode ------------------------------------------------------------
// An accumulator machine over inputs (a, b): the accumulator starts at `a`; each op folds in
// `b` or an immediate. OP_END returns the accumulator.
enum { OP_ADDB, OP_MULB, OP_ADDK, OP_MULK, OP_END };
typedef struct {
  int op;
  long k;
} Ins;

static long interp(Ins *prog, long a, long b) {
  long acc = a;
  for (int i = 0;; i++) {
    if (prog[i].op == OP_ADDB)
      acc += b;
    else if (prog[i].op == OP_MULB)
      acc *= b;
    else if (prog[i].op == OP_ADDK)
      acc += prog[i].k;
    else if (prog[i].op == OP_MULK)
      acc *= prog[i].k;
    else
      return acc;
  }
}

// --- the runtime emitter: bytecode -> serialized SVM IR -------------------------------------
// The binary layout mirrors `crates/svm-encode` (LEB128 + one-byte opcodes): magic "SVM\0" +
// version (3), a memory descriptor, a data-segment count, a §7 import count (0 here — this unit
// is self-contained), a function count, then per function its
// params/results/blocks, each block its params, instruction count, instructions, terminator.
// Values are block-local indices (params first, then each instruction's result).

static int n_out;

static void eb(char *buf, int v) { buf[n_out++] = (char)v; }

static void uleb(char *buf, unsigned long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7;
    if (v) {
      eb(buf, b7 | 0x80);
    } else {
      eb(buf, b7);
      return;
    }
  }
}

static void sleb(char *buf, long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7; // arithmetic shift: sign-extends
    int done = (v == 0 && !(b7 & 0x40)) || (v == -1 && (b7 & 0x40));
    eb(buf, done ? b7 : (b7 | 0x80));
    if (done)
      return;
  }
}

// Emit a one-function module — the unit's entry, `(i64, i64) -> (i64)` (the fixed shape
// `__vm_jit_invoke2` calls) — computing the bytecode program as straight-line IR.
//
// `abi_sp` selects the entry's calling convention:
//   0 = raw `(i64 a, i64 b) -> (i64)`         — for `__vm_jit_invoke2` (no C frame).
//   1 = guest ABI `(i64 sp, i64 a, i64 b) -> (i64)` — for `install` + a C function-pointer
//       call: this frontend threads the data-stack pointer as every function's hidden first
//       param, so a unit reached through a C `(*fp)(a, b)` MUST declare it (the leaf body
//       ignores it). Picking the wrong shape is a clean `IndirectCallType` trap, not an escape.
static long emit_unit(Ins *prog, char *buf, int abi_sp) {
  int nparams = abi_sp + 2; // [sp,] a, b
  long a_idx = abi_sp;      // the accumulator seed (v0 raw, v1 with the SP param)
  long b_idx = abi_sp + 1;
  n_out = 0;
  // Header: magic + version.
  eb(buf, 'S');
  eb(buf, 'V');
  eb(buf, 'M');
  eb(buf, 0);
  eb(buf, 5); // format v5 (v4 sections + the impl-export section)
  // Memory descriptor: present, size_log2 16. The validator's memory-match precondition
  // requires the blob to declare the SAME window as this module — chibicc keeps a small
  // program like this one at the 64 KiB default (a mismatch is a clean -22, not an escape).
  eb(buf, 1);
  eb(buf, 16);
  eb(buf, 0); // no data segments (the validator rejects them anyway)
  eb(buf, 0); // no imports — this unit is self-contained (v2 import section)
  eb(buf, 0); // no exports — invoked by handle, not by name (v3 export section)
  eb(buf, 0); // no impl exports (v5 impl-export section)
  eb(buf, 1); // one function
  // params (nparams × i64), results (i64) — type tag 1 = i64.
  eb(buf, nparams);
  for (int k = 0; k < nparams; k++)
    eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  // One block whose params mirror the function's.
  eb(buf, 1);
  eb(buf, nparams);
  for (int k = 0; k < nparams; k++)
    eb(buf, 1);
  // Instruction count: ADDB/MULB are one binop; ADDK/MULK are a const + a binop.
  long ni = 0;
  for (int i = 0; prog[i].op != OP_END; i++)
    ni += (prog[i].op == OP_ADDK || prog[i].op == OP_MULK) ? 2 : 1;
  uleb(buf, ni);
  // The straight-line body. Opcodes mirror svm-encode: 0x11 = i64.const (+ sleb immediate);
  // 0x40 + BinOp index = i64 binop (add = 0x40, mul = 0x42) + uleb operand indices.
  long acc = a_idx;
  long next = nparams; // results follow the block params
  for (int i = 0; prog[i].op != OP_END; i++) {
    long rhs;
    int mul = (prog[i].op == OP_MULB || prog[i].op == OP_MULK);
    if (prog[i].op == OP_ADDK || prog[i].op == OP_MULK) {
      eb(buf, 0x11);
      sleb(buf, prog[i].k);
      rhs = next++;
    } else {
      rhs = b_idx;
    }
    eb(buf, mul ? 0x42 : 0x40);
    uleb(buf, acc);
    uleb(buf, rhs);
    acc = next++;
  }
  // return <acc>
  eb(buf, 0x83);
  uleb(buf, 1);
  uleb(buf, acc);
  return n_out;
}

int main() {
  // The "hot function": ((a * 3 + b) * b) + 7.
  Ins prog[5];
  prog[0].op = OP_MULK;
  prog[0].k = 3;
  prog[1].op = OP_ADDB;
  prog[2].op = OP_MULB;
  prog[3].op = OP_ADDK;
  prog[3].k = 7;
  prog[4].op = OP_END;

  static char buf[256];
  long bad = 0;

  // --- Path 1: raw `invoke` of the JITed hot loop (the interpreter-accelerates-itself shape).
  long n = emit_unit(prog, buf, 0); // raw (i64, i64) -> (i64)
  puts1("emitted ");
  put_i64(n);
  puts1(" bytes of IR\n");
  long code = __vm_jit_compile(buf, n);
  if (code < 0) {
    puts1("jit compile failed: ");
    put_i64(code);
    puts1("\n");
    return 1;
  }
  for (long a = -3; a <= 3; a++)
    for (long b = -3; b <= 3; b++)
      if (interp(prog, a, b) != __vm_jit_invoke2(code, a, b))
        bad++;
  puts1("invoke jit(5, 9) = ");
  put_i64(__vm_jit_invoke2(code, 5, 9));
  puts1(" (interp ");
  put_i64(interp(prog, 5, 9));
  puts1(")\n");
  __vm_jit_release(code);

  // --- Path 2: old→new via install. Emit the SAME hot loop with the guest ABI (a leading
  // data-SP param), install it into the call_indirect table, and call it like an ordinary C
  // function pointer — old code dispatching freshly-JITed code at native speed.
  long n2 = emit_unit(prog, buf, 1); // guest ABI (i64 sp, i64, i64) -> (i64)
  long code2 = __vm_jit_compile(buf, n2);
  long slot = code2 < 0 ? code2 : __vm_jit_install(code2);
  if (slot < 0) {
    puts1("install failed: ");
    put_i64(slot);
    puts1("\n");
    return 1;
  }
  long (*hot)(long, long) = (long (*)(long, long))slot;
  for (long a = -3; a <= 3; a++)
    for (long b = -3; b <= 3; b++)
      if (interp(prog, a, b) != hot(a, b))
        bad++;
  puts1("installed hot(5, 9) = ");
  put_i64(hot(5, 9));
  puts1(" via call_indirect slot ");
  put_i64(slot);
  puts1("\n");
  __vm_jit_release(code2);

  if (bad) {
    puts1("MISMATCHES: ");
    put_i64(bad);
    puts1("\n");
    return 1;
  }
  puts1("98 inputs agree (invoke + installed call_indirect): "
        "guest-emitted, host-verified, Cranelift-compiled\n");
  return 0;
}
