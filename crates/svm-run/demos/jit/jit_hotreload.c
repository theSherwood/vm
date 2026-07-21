// **Hot reload / live patching** over the in-guest dynamic linker (DESIGN.md §22): redefining a
// symbol gives it a *new* slot, but units already linked to the *old* one keep their binding — old
// callers stay pinned to the old version, new callers bind to the new. This is the slot model's
// live-patch behaviour, made concrete.
//
//   vm_dlopen("f", a+100)   — f v1
//   vm_dlopen("g", caller)  — g calls f BY NAME → binds f v1's slot
//   vm_dlopen("f", a+200)   — HOT RELOAD: f v2 at a new slot (v1's slot stays installed for g)
//   vm_dlopen("h", caller)  — h calls f BY NAME → binds f v2's slot
//   g(5) = 105   (pinned to f v1: a+100)
//   h(5) = 205   (sees f v2: a+200)
//
// `g` and `h` are the *same* unit bytes loaded at different times; the only difference is which `f`
// was current when each was linked. Driven by `c_frontend.rs::c_guest_jit_hotreload_demo`
// (interp == JIT). Run standalone:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_hotreload.c

#include <svm.h>
#include <vm_dl.h>

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

// --- the byte emitter (serialized SVM IR v2), as in the other jit demos -----------------------
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
    v >>= 7;
    int done = (v == 0 && !(b7 & 0x40)) || (v == -1 && (b7 & 0x40));
    eb(buf, done ? b7 : (b7 | 0x80));
    if (done)
      return;
  }
}
static void emit_header(char *buf) {
  n_out = 0;
  eb(buf, 'S');
  eb(buf, 'V');
  eb(buf, 'M');
  eb(buf, 0);
  eb(buf, 6); // format v6 (v5 sections + the interface section)
  eb(buf, 1);
  eb(buf, 16);
  eb(buf, 0);
}
static void emit_i64_pair_sig(char *buf) {
  eb(buf, 2);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
}
// `f(a, b) = a + k`, self-contained. Values: v0,v1 = params; v2 = k; v3 = a + k.
static long emit_adder(char *buf, long k) {
  emit_header(buf);
  eb(buf, 0); // 0 imports
  eb(buf, 0); // 0 exports (v3 export section)
  eb(buf, 0); // 0 interfaces (v6 interface section)
  eb(buf, 0); // 0 impl exports (v5 impl-export section)
  eb(buf, 1); // 1 function
  emit_i64_pair_sig(buf);
  eb(buf, 1); // 1 block
  eb(buf, 2); // block params (i64, i64)
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 2);    // 2 instructions
  eb(buf, 0x11); // v2 = i64.const k
  sleb(buf, k);
  eb(buf, 0x40); // v3 = i64.add v0 v2
  uleb(buf, 0);
  uleb(buf, 2);
  eb(buf, 0x83); // return v3
  uleb(buf, 1);
  uleb(buf, 3);
  return n_out;
}
// `caller(a, b) = f(a, b)` — imports `f` by name (idx 0) and forwards. The *same* bytes are loaded
// as `g` and `h`; each binds to whichever `f` is current at load. Values: v2 = handle const; v3 = f.
static long emit_caller(char *buf) {
  emit_header(buf);
  eb(buf, 1); // 1 import: "f"
  eb(buf, 1);
  eb(buf, 'f');
  emit_i64_pair_sig(buf);
  eb(buf, 0); // mode: required (v4)
  eb(buf, 0); // 0 exports (v3 export section)
  eb(buf, 0); // 0 interfaces (v6 interface section)
  eb(buf, 0); // 0 impl exports (v5 impl-export section)
  eb(buf, 1); // 1 function
  emit_i64_pair_sig(buf);
  eb(buf, 1); // 1 block
  eb(buf, 2); // block params (i64, i64)
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 2);    // 2 instructions
  eb(buf, 0x10); // v2 = i32.const 0  (import handle placeholder)
  sleb(buf, 0);
  eb(buf, 0x7C); // v3 = call.import "f" (import 0) (v0, v1)
  uleb(buf, 0);
  emit_i64_pair_sig(buf);
  uleb(buf, 2); // handle = v2
  eb(buf, 2);   // 2 args
  uleb(buf, 0);
  uleb(buf, 1);
  eb(buf, 0x83); // return v3
  uleb(buf, 1);
  uleb(buf, 3);
  return n_out;
}

int main() {
  static char buf[256];

  if (vm_dlopen("f", buf, emit_adder(buf, 100)) < 0) { // f v1 = a + 100
    puts1("f v1 load failed\n");
    return 1;
  }
  if (vm_dlopen("g", buf, emit_caller(buf)) < 0) { // g links to f v1 by name
    puts1("g load failed\n");
    return 1;
  }
  if (vm_dlopen("f", buf, emit_adder(buf, 200)) < 0) { // HOT RELOAD: f v2 = a + 200
    puts1("f v2 reload failed\n");
    return 1;
  }
  if (vm_dlopen("h", buf, emit_caller(buf)) < 0) { // h links to f v2 by name
    puts1("h load failed\n");
    return 1;
  }

  puts1("g(5) = ");
  put_i64(vm_dlcall2("g", 5, 0)); // f v1: 5 + 100 = 105
  puts1("  [pinned to f v1: a+100]\n");
  puts1("h(5) = ");
  put_i64(vm_dlcall2("h", 5, 0)); // f v2: 5 + 200 = 205
  puts1("  [sees f v2: a+200]\n");
  puts1("hot reload: old callers keep the old binding\n");
  return 0;
}
