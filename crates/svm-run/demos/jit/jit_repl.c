// Auto-compacting guest-driven JIT REPL (DESIGN.md §22): the **prompt body** of a long-lived REPL
// that JITs a fresh unit every prompt — and never exhausts the code arena, because the embedder
// recompacts between prompts.
//
// `cranelift-jit`'s code arena has no per-function free: every `__vm_jit_compile` consumes arena
// bytes and nothing is returned, so a REPL that compiles each prompt would eventually hit `-ENOMEM`.
// The reclaim (DESIGN.md §22) is **whole-module recompaction** at a quiescent point — and the only sound
// quiescent point is *between* prompts, when the guest has returned to the host. So this is not a
// standalone `cargo run` program like `jit_demo.c`: it is driven by the embedder's `svm_run::JitSession`,
// which re-enters this entry once per prompt over a **persistent window** and auto-compacts when the
// live code crosses a byte watermark. The guest never observes the reclaim — its accumulator and
// every result are byte-identical with compaction off vs on.
//
// Each prompt: build serialized SVM IR for the unit `(a, b) -> a*b + 10`, `__vm_jit_compile` it (a
// *fresh* compilation — new arena bytes — even though the blob is identical), `__vm_jit_invoke2` it
// with `(x, x)` where `x = prompt + 2`, then **release** the handle (so it becomes dead code the next
// compaction reclaims) and fold the result into a running accumulator. The accumulator and the prompt
// counter live in **zero-initialized BSS** globals — no `data` segment, so they are *not* re-applied
// when the session reseeds the window each prompt; they simply persist as carried window state. The
// per-prompt result is returned, so the embedder sees the REPL transcript.
//
// Driven by `c_frontend.rs::c_guest_jit_repl_compacts`, which runs many prompts with the watermark
// off and on and asserts identical results/window while the on-run's code-arena occupancy stays
// bounded by the live set. Run standalone, it executes exactly one prompt:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_repl.c

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

// --- a serialized-IR emitter for one unit `(i64 a, i64 b) -> (i64)` = `a*b + 10` ---------------
// The binary layout mirrors `crates/svm-encode` (LEB128 + one-byte opcodes), exactly as
// `jit_demo.c` builds it. Opcodes: 0x11 = i64.const (+ sleb), 0x40 = i64.add, 0x42 = i64.mul
// (+ uleb operand indices), 0x83 = return.
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

static long emit_unit(char *buf) {
  n_out = 0;
  // Header: magic + version.
  eb(buf, 'S');
  eb(buf, 'V');
  eb(buf, 'M');
  eb(buf, 0);
  eb(buf, 5); // format v5 (v4 sections + the impl-export section)
  // Memory descriptor: present, size_log2 16 — must match this module's window (the validator's
  // memory-match precondition), which chibicc keeps at the 64 KiB default for a small program.
  eb(buf, 1);
  eb(buf, 16);
  eb(buf, 0); // no data segments
  eb(buf, 0); // no imports — self-contained unit (v2 import section)
  eb(buf, 0); // no exports — invoked by handle, not by name (v3 export section)
  eb(buf, 0); // no impl exports (v5 impl-export section)
  eb(buf, 1); // one function
  // params (a, b) : i64, i64 ; results : i64 (type tag 1 = i64).
  eb(buf, 2);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  // One block whose params mirror the function's.
  eb(buf, 1);
  eb(buf, 2);
  eb(buf, 1);
  eb(buf, 1);
  // 3 instructions: mul, const 10, add.
  uleb(buf, 3);
  // v2 = i64.mul v0 v1   (a * b)
  eb(buf, 0x42);
  uleb(buf, 0);
  uleb(buf, 1);
  // v3 = i64.const 10
  eb(buf, 0x11);
  sleb(buf, 10);
  // v4 = i64.add v2 v3   (+ 10)
  eb(buf, 0x40);
  uleb(buf, 2);
  uleb(buf, 3);
  // return v4
  eb(buf, 0x83);
  uleb(buf, 1);
  uleb(buf, 4);
  return n_out;
}

// Persistent REPL state — zero-initialized BSS (no `data` segment), so the session's per-prompt
// window reseed leaves them untouched and they carry across prompts as ordinary window bytes.
static long g_acc;    // running accumulator (the REPL's "session state")
static long g_prompt; // how many prompts have run

int main(void) {
  char buf[256];
  long x = g_prompt + 2; // this prompt's input (matches the IR-level session test: x = i + 2)
  g_prompt++;

  long n = emit_unit(buf);
  long code = __vm_jit_compile(buf, n);
  if (code < 0) {
    puts1("jit compile failed: ");
    put_i64(code);
    puts1("\n");
    return (int)code;
  }
  long r = __vm_jit_invoke2(code, x, x); // x*x + 10
  __vm_jit_release(code);                // dead code now — the next compaction reclaims it

  g_acc += r;
  puts1("prompt ");
  put_i64(g_prompt);
  puts1(": +");
  put_i64(r);
  puts1(" -> acc=");
  put_i64(g_acc);
  puts1("\n");
  return (int)g_acc;
}
