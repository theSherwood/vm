#ifndef VM_DL_H
#define VM_DL_H
// In-guest dynamic-linking loader — `vm_dlopen`/`vm_dlsym`/`vm_dlclose` over the `Jit` capability
// (DYNLINK.md C3b). A "shared object" is serialized SVM IR; a symbol is an installed `call_indirect`
// slot (an **unforgeable funcref**, §3c-checked at the call). This header is the ergonomic layer over
// the raw `__vm_jit_compile_linked` / `__vm_jit_install` primitives: it keeps a `name → slot` registry
// and marshals it into the symbol-table buffer the host resolves against, so a loaded unit can
// reference any already-loaded symbol **by name** — and a mis-link is caught by the host's
// re-verification, never trusted.
//
// Why this is better than POSIX `dlopen`: the object is re-verified on load (a malicious one can't
// escape — worst case it corrupts its own window), loading is capability-gated (you need the `Jit`
// handle, and the IR arrives through the powerbox — no ambient "load any file"), and `dlsym` yields a
// checked funcref slot, not a raw pointer.
#include <svm.h>

#ifndef VM_DL_MAX
#define VM_DL_MAX 64 // max simultaneously-loaded symbols
#endif
#ifndef VM_DL_NAMEMAX
#define VM_DL_NAMEMAX 32 // max symbol-name length (incl. NUL)
#endif

typedef struct {
  char name[VM_DL_NAMEMAX];
  int used;
  long slot; // the installed call_indirect slot (the funcref other units link to)
  long code; // the Jit code handle (for invoking the symbol directly, via __vm_jit_invoke2)
} VmDlSym;

static VmDlSym vm_dl_reg[VM_DL_MAX];
static int vm_dl_count;

static int vm_dl_streq(const char *a, const char *b) {
  while (*a && *a == *b) {
    a++;
    b++;
  }
  return *a == *b;
}

static int vm_dl_strlen(const char *s) {
  int n = 0;
  while (s[n])
    n++;
  return n;
}

// Append an unsigned LEB128 to `buf` at `*pos`.
static void vm_dl_uleb(char *buf, long *pos, unsigned long v) {
  for (;;) {
    int b7 = v & 0x7f;
    v >>= 7;
    buf[(*pos)++] = (char)(v ? (b7 | 0x80) : b7);
    if (!v)
      return;
  }
}

// Marshal the whole registry into a `compile_linked` symbol table (the C2 wire form): `count`, then
// per entry `name` (uleb len + bytes), a `kind` byte (`0` = a table slot), and the slot (uleb).
// Passing the *whole* registry is fine — the host only binds the imports a unit actually declares.
static long vm_dl_build_symtab(char *buf) {
  long pos = 0;
  vm_dl_uleb(buf, &pos, (unsigned long)vm_dl_count);
  for (int i = 0; i < VM_DL_MAX; i++) {
    if (!vm_dl_reg[i].used)
      continue;
    int nlen = vm_dl_strlen(vm_dl_reg[i].name);
    vm_dl_uleb(buf, &pos, (unsigned long)nlen);
    for (int k = 0; k < nlen; k++)
      buf[pos++] = vm_dl_reg[i].name[k];
    buf[pos++] = 0; // kind 0 = Slot
    vm_dl_uleb(buf, &pos, (unsigned long)vm_dl_reg[i].slot);
  }
  return pos;
}

// `vm_dlsym(name)` → the symbol's installed slot (a funcref another unit can `call_indirect`), or
// `-1` if it is not loaded.
static long vm_dlsym(const char *name) {
  for (int i = 0; i < VM_DL_MAX; i++)
    if (vm_dl_reg[i].used && vm_dl_streq(vm_dl_reg[i].name, name))
      return vm_dl_reg[i].slot;
  return -1;
}

// `vm_dlopen(name, ir, ir_len)`: load a unit (serialized SVM IR) that may import already-loaded
// symbols **by name**. Resolve its imports against the registry, compile (the host re-verifies),
// install it into the shared table, and register it under `name`. Returns the slot (>= 0), or a
// negative errno (-22 link/verify failed, -28 table full, -12 registry full). Idempotent names are
// the caller's concern: a repeat `name` registers a *second* entry (the newest wins on lookup — the
// hot-reload shape).
static long vm_dlopen(const char *name, const void *ir, long ir_len) {
  static char symtab[VM_DL_MAX * (VM_DL_NAMEMAX + 8) + 8];
  long st_len = vm_dl_build_symtab(symtab);
  long code = __vm_jit_compile_linked((void *)ir, ir_len, symtab, st_len);
  if (code < 0)
    return code;
  long slot = __vm_jit_install(code);
  if (slot < 0) {
    __vm_jit_release(code);
    return slot;
  }
  // Register — or **hot-reload**: if the name is already loaded, overwrite its slot+code in place.
  // The *previous* slot stays installed, so any unit already linked to it keeps working (it baked
  // that slot at link time); only a *later* `vm_dlopen`'s symbol table sees the new slot. That is
  // the live-patch shape: old callers pinned to the old version, new callers bound to the new.
  int free_idx = -1;
  for (int i = 0; i < VM_DL_MAX; i++) {
    if (vm_dl_reg[i].used && vm_dl_streq(vm_dl_reg[i].name, name)) {
      vm_dl_reg[i].slot = slot;
      vm_dl_reg[i].code = code;
      return slot;
    }
    if (free_idx < 0 && !vm_dl_reg[i].used)
      free_idx = i;
  }
  if (free_idx < 0)
    return -12; // registry full
  int k = 0;
  while (name[k] && k < VM_DL_NAMEMAX - 1) {
    vm_dl_reg[free_idx].name[k] = name[k];
    k++;
  }
  vm_dl_reg[free_idx].name[k] = 0;
  vm_dl_reg[free_idx].slot = slot;
  vm_dl_reg[free_idx].code = code;
  vm_dl_reg[free_idx].used = 1;
  vm_dl_count++;
  return slot;
}

// `vm_dlcall2(name, a, b)`: look the name up and invoke it (the REPL "eval"). The symbol's unit must
// have the raw `(i64, i64) -> (i64)` entry shape `__vm_jit_invoke2` requires. Returns the result, or
// `-1` if the name is not loaded.
static long vm_dlcall2(const char *name, long a, long b) {
  for (int i = 0; i < VM_DL_MAX; i++)
    if (vm_dl_reg[i].used && vm_dl_streq(vm_dl_reg[i].name, name))
      return __vm_jit_invoke2(vm_dl_reg[i].code, a, b);
  return -1;
}

// `vm_dlclose(name)`: uninstall the symbol's slot (a stale `call_indirect` of it then traps) and
// drop it from the registry. Returns 0, or `-1` if the name is not loaded. (The code memory itself
// is not reclaimed — the JIT arena has no per-function free; this frees the *slot* + the name.)
static int vm_dlclose(const char *name) {
  for (int i = 0; i < VM_DL_MAX; i++) {
    if (vm_dl_reg[i].used && vm_dl_streq(vm_dl_reg[i].name, name)) {
      __vm_jit_uninstall(vm_dl_reg[i].slot);
      __vm_jit_release(vm_dl_reg[i].code);
      vm_dl_reg[i].used = 0;
      vm_dl_count--;
      return 0;
    }
  }
  return -1;
}

#endif // VM_DL_H
