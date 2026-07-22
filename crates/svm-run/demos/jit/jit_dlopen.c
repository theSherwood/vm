// A guest-side **linking loader** in C (DESIGN.md §22): build up a set of functions at runtime where
// each one may call earlier ones **by name**, using the `vm_dlopen`/`vm_dlsym`/`vm_dlclose` library
// (`<vm_dl.h>`) over the `Jit` capability. This is the guest-C twin of the `dynlink_repl.rs` harness:
// a growing symbol table where definitions persist and compose by name — `dlopen` for SVM, in-sandbox.
//
// The guest emits three units and `vm_dlopen`s each under a name:
//   * `add(a, b) = a + b`                       — a leaf (no imports)
//   * `mul(a, b) = a * b`                        — a leaf
//   * `poly(a, b) = add(mul(a, a), b) = a*a + b` — imports `mul` AND `add` **by name**
// `poly`'s imports are resolved against the loader's registry, re-verified, and installed; calling it
// dispatches through the shared table to the installed `add`/`mul`. `poly(5,2)=27`, `poly(3,4)=13`.
// Then `vm_dlclose("poly")` unloads it and `vm_dlsym` confirms it's gone.
//
// Driven by `c_frontend.rs::c_guest_jit_dlopen_demo` (interp == JIT differential). Run standalone:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_dlopen.c

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

// --- the byte emitter: serialized SVM IR (binary `svm-encode` v2), as in jit_link.c -----------
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
// Magic + v2 + memory(16) + 0 data segments (common to every unit).
static void emit_header(char *buf) {
  n_out = 0;
  eb(buf, 'S');
  eb(buf, 'V');
  eb(buf, 'M');
  eb(buf, 0);
  eb(buf, 8); // format v8 (single-string import names; call.sym link form)
  eb(buf, 1);
  eb(buf, 16);
  eb(buf, 0);
}
// A `(i64, i64) -> (i64)` signature.
static void emit_i64_pair_sig(char *buf) {
  eb(buf, 2);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
}
// `(a, b) -> a <op> b`, self-contained. opcode 0x40 = i64.add, 0x42 = i64.mul.
static long emit_binop(char *buf, int opcode) {
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
  eb(buf, 1);      // 1 instruction
  eb(buf, opcode); // v2 = <op> v0 v1
  uleb(buf, 0);
  uleb(buf, 1);
  eb(buf, 0x83); // return v2
  uleb(buf, 1);
  uleb(buf, 2);
  return n_out;
}
// `poly(a, b) = add(mul(a, a), b)` — imports `mul` (idx 0) and `add` (idx 1) by name.
// Values: v0,v1 = params; v2 = handle const; v3 = mul(a,a); v4 = handle const; v5 = add(v3,b).
static long emit_poly(char *buf) {
  emit_header(buf);
  eb(buf, 2); // 2 imports (v7: ns + name + shape ref)
  eb(buf, 3); // "mul"
  eb(buf, 'm');
  eb(buf, 'u');
  eb(buf, 'l');
  eb(buf, 0);   // shape tag: func
  uleb(buf, 0); //   -> type entry 0
  eb(buf, 0);   // mode: required (v4)
  eb(buf, 3); // "add"
  eb(buf, 'a');
  eb(buf, 'd');
  eb(buf, 'd');
  eb(buf, 0);   // shape tag: func
  uleb(buf, 0); //   -> type entry 0
  eb(buf, 0);   // mode: required (v4)
  eb(buf, 0); // 0 exports (v3 export section)
  eb(buf, 1); // 1 type entry (v7 type section)
  eb(buf, 0); //   tag: Func
  emit_i64_pair_sig(buf);
  eb(buf, 0); // 0 impl exports (v5 impl-export section)
  eb(buf, 1); // 1 function
  emit_i64_pair_sig(buf);
  eb(buf, 1); // 1 block
  eb(buf, 2); // block params (i64, i64)
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 4);    // 4 instructions
  eb(buf, 0x10); // v2 = i32.const 0  (import handle placeholder)
  sleb(buf, 0);
  eb(buf, 0x0E); // v3 = call.sym "mul" (import 0) (a, a) — v8 link-form placeholder
  uleb(buf, 0);
  emit_i64_pair_sig(buf);
  uleb(buf, 2); // handle = v2
  eb(buf, 2);   // 2 args
  uleb(buf, 0);
  uleb(buf, 0);
  eb(buf, 0x10); // v4 = i32.const 0  (import handle placeholder)
  sleb(buf, 0);
  eb(buf, 0x0E); // v5 = call.sym "add" (import 1) (v3, b) — v8 link-form placeholder
  uleb(buf, 1);
  emit_i64_pair_sig(buf);
  uleb(buf, 4); // handle = v4
  eb(buf, 2);   // 2 args
  uleb(buf, 3);
  uleb(buf, 1);
  eb(buf, 0x83); // return v5
  uleb(buf, 1);
  uleb(buf, 5);
  return n_out;
}

int main() {
  static char buf[256];

  // Load the two leaves, then `poly` — which links to both **by name** at load.
  if (vm_dlopen("add", buf, emit_binop(buf, 0x40)) < 0 ||
      vm_dlopen("mul", buf, emit_binop(buf, 0x42)) < 0) {
    puts1("leaf load failed\n");
    return 1;
  }
  long ps = vm_dlopen("poly", buf, emit_poly(buf));
  if (ps < 0) {
    puts1("poly load failed: ");
    put_i64(ps);
    puts1("\n");
    return 1;
  }

  puts1("poly(5, 2) = ");
  put_i64(vm_dlcall2("poly", 5, 2)); // add(mul(5,5),2) = 27
  puts1("\n");
  puts1("poly(3, 4) = ");
  put_i64(vm_dlcall2("poly", 3, 4)); // add(mul(3,3),4) = 13
  puts1("\n");

  // `vm_dlclose` unloads a symbol; `vm_dlsym` then reports it gone.
  vm_dlclose("poly");
  puts1(vm_dlsym("poly") < 0 ? "poly unloaded; dlsym -> -1\n" : "poly STILL loaded\n");
  puts1("linked by name via vm_dlopen/vm_dlsym/vm_dlclose\n");
  return 0;
}
