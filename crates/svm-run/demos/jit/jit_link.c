// Guest-side **dynamic linking** in C (DESIGN.md §22): a guest program that loads a separately
// "compiled" code unit which references another unit **by name**, and links them at runtime — the
// `dlopen`/`dlsym` story, done entirely inside the sandbox over the `Jit` capability.
//
// The guest emits two serialized SVM IR units byte-by-byte (the binary `svm-encode` v2 format):
//   * `service(a, b) = a*a + b`            — self-contained; compiled then **installed** into the
//                                            shared call_indirect table, yielding a slot.
//   * `client(a, b) = F(a, b) + 100`       — carries an **unresolved import** `F`; it knows the
//                                            service only by the name "F".
// The guest then builds a tiny **symbol table** binding "F" -> the service's slot and hands the
// client + table to `__vm_jit_compile_linked` (iface 11 op 5). The host resolves the import by
// name, **re-verifies**, and compiles — so the client reaches the service through the table by name,
// exactly like a loaded `.so` calling a symbol the loader resolved. `client(5, 2) = (25+2)+100 = 127`.
//
// Driven by `c_frontend.rs::c_guest_jit_link_demo` (interp == JIT differential). Run standalone:
//
//   cargo run -p svm-run -- crates/svm-run/demos/jit/jit_link.c

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

// --- the byte emitter: serialized SVM IR (binary `svm-encode` v2) ---------------------------
// One-byte opcodes + LEB128, mirroring `crates/svm-encode`. Value operands are block-local indices
// (block params first, then each instruction's result). `n_out` is the running write cursor.
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

// The header common to every unit: magic + v2 + memory(16) + 0 data segments. Leaves the import
// count, function count, and bodies to the caller (they differ between service and client).
static void emit_header(char *buf) {
  n_out = 0;
  eb(buf, 'S');
  eb(buf, 'V');
  eb(buf, 'M');
  eb(buf, 0);
  eb(buf, 2);  // format v2 (the import section below)
  eb(buf, 1);  // memory present
  eb(buf, 16); // size_log2 = 16 (must match this module's 64 KiB window)
  eb(buf, 0);  // no data segments
}

// A `(i64, i64) -> (i64)` signature: params (count 2, both i64=1), results (count 1, i64).
static void emit_i64_pair_sig(char *buf) {
  eb(buf, 2);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 1);
}

// `service(a, b) = a*a + b` — a self-contained unit (no imports). Opcodes: 0x42 = i64.mul,
// 0x40 = i64.add, 0x83 = return. Values: v0,v1 = params; v2 = mul; v3 = add.
static long emit_service(char *buf) {
  emit_header(buf);
  eb(buf, 0); // 0 imports
  eb(buf, 1); // 1 function
  emit_i64_pair_sig(buf);
  eb(buf, 1); // 1 block
  eb(buf, 2); // block params: (i64, i64)
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 2);     // 2 instructions
  eb(buf, 0x42);  // v2 = i64.mul v0 v0
  uleb(buf, 0);
  uleb(buf, 0);
  eb(buf, 0x40);  // v3 = i64.add v2 v1
  uleb(buf, 2);
  uleb(buf, 1);
  eb(buf, 0x83);  // return v3
  uleb(buf, 1);
  uleb(buf, 3);
  return n_out;
}

// `client(a, b) = F(a, b) + 100`, where `F` is an **unresolved import** the loader binds by name.
// Opcodes: 0x10 = i32.const (the import's handle placeholder — patched to the resolved slot and
// reused as the call_indirect index), 0x7C = call.import, 0x11 = i64.const, 0x40 = i64.add.
// Values: v0,v1 = params; v2 = handle const; v3 = F(a,b); v4 = 100; v5 = v3+v4.
static long emit_client(char *buf) {
  emit_header(buf);
  // Import section: one import, "F" : (i64, i64) -> (i64).
  eb(buf, 1); // 1 import
  eb(buf, 1); // name length
  eb(buf, 'F');
  emit_i64_pair_sig(buf); // the import's op signature
  eb(buf, 1);             // 1 function
  emit_i64_pair_sig(buf);
  eb(buf, 1); // 1 block
  eb(buf, 2); // block params: (i64, i64)
  eb(buf, 1);
  eb(buf, 1);
  eb(buf, 4);    // 4 instructions
  eb(buf, 0x10); // v2 = i32.const 0  (import handle placeholder)
  sleb(buf, 0);
  eb(buf, 0x7C);          // v3 = call.import "F" ...
  uleb(buf, 0);           //   import index 0
  emit_i64_pair_sig(buf); //   self-describing sig (i64, i64) -> (i64)
  uleb(buf, 2);           //   handle operand = v2
  eb(buf, 2);             //   2 args
  uleb(buf, 0);           //     v0
  uleb(buf, 1);           //     v1
  eb(buf, 0x11);          // v4 = i64.const 100
  sleb(buf, 100);
  eb(buf, 0x40); // v5 = i64.add v3 v4
  uleb(buf, 3);
  uleb(buf, 4);
  eb(buf, 0x83); // return v5
  uleb(buf, 1);
  uleb(buf, 5);
  return n_out;
}

int main() {
  static char svc_buf[128];
  static char cli_buf[128];
  static char symtab[16];

  // 1) Compile + install the service → its call_indirect table slot.
  long svc_len = emit_service(svc_buf);
  long svc = __vm_jit_compile(svc_buf, svc_len);
  if (svc < 0) {
    puts1("service compile failed: ");
    put_i64(svc);
    puts1("\n");
    return 1;
  }
  long slot = __vm_jit_install(svc);
  if (slot < 0) {
    puts1("service install failed: ");
    put_i64(slot);
    puts1("\n");
    return 1;
  }

  // 2) Build the symbol table binding "F" -> that slot (the loader's registry), using the slot the
  //    install actually returned — `[count=1][namelen=1]['F'][kind=0/Slot][slot uleb]`.
  n_out = 0;
  uleb(symtab, 1);            // 1 entry
  uleb(symtab, 1);            // name length
  eb(symtab, 'F');           // name
  eb(symtab, 0);             // kind 0 = Slot
  uleb(symtab, (unsigned long)slot);
  long st_len = n_out;

  // 3) Compile the client, resolving its import `F` by name against the table — host-assisted.
  long cli_len = emit_client(cli_buf);
  long cli = __vm_jit_compile_linked(cli_buf, cli_len, symtab, st_len);
  if (cli < 0) {
    puts1("linked compile failed: ");
    put_i64(cli);
    puts1("\n");
    return 1;
  }

  // 4) Invoke the client — it reaches the installed service through the table, by name.
  long r = __vm_jit_invoke2(cli, 5, 2);
  puts1("client(5, 2) = ");
  put_i64(r);
  puts1(r == 127 ? "  [linked by name: service(5,2)+100]\n" : "  [WRONG]\n");
  return r == 127 ? 0 : 1;
}
