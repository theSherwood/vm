// codegen_ir.c — SVM text IR backend for chibicc (DESIGN.md §3d).
//
// Walks chibicc's typed AST and emits the SVM text IR our verifier/interpreter/JIT
// consume, instead of x86-64 assembly. This is the C frontend's lowering: source UB is
// resolved here into the IR's total semantics (§3b), and the *verifier* — not this code
// — is what enforces escape-freedom (§2a), so the frontend is outside the escape-TCB.
//
// Status (grown incrementally): `int`/`long`/`void` functions with integer expressions
// (constants, +/-/*'/'%, bitwise, shifts, comparisons, unary -/!/~, integer casts),
// **scalar locals** in the §3d data-stack window (load/store, assignment, `&`/`*`,
// pointers to locals), **structured control flow** (`if`/`else`, `while`, `for` → a
// block CFG), and **functions** — parameters and **direct calls incl. recursion**, made
// correct by threading the **data-stack pointer**: it is parameter `v0` of every
// function and block, a local lives at `sp + offset`, and a call gives the callee a
// fresh frame at `sp + frame_size` so recursion never clobbers a parent frame.
// Short-circuit `&&`/`||` and ternary `?:` lower to a diamond whose merge block carries
// the result as a second block parameter (alongside the data-SP). **Arrays and
// structs/unions** work too: indexing is chibicc's `*(base + i*size)` (an array decays
// to its i64 address in value context), and `s.field` / `p->field` add the member offset
// (`ND_MEMBER`); initializers lower to per-element/-member scalar stores. **By-value
// aggregates** (sret, §3d D39) work: a struct/union passes by hidden pointer (the callee
// copies it into its own frame), a struct/union return uses a hidden `sret` pointer the
// callee writes through (the function is `-> ()`), and whole-aggregate assignment is a
// byte copy. **Globals + string literals** live at fixed
// window offsets in a data region below the data stack, emitted as module-level `data`
// segments (§3a) the runtime copies in (string literals as `data ro`, D40). **Pointer
// initializers become relocations** (§3a): `char *p = "..."`, `&global`, `&arr[k]`,
// function pointers, and arrays/structs of them are resolved at compile time to the
// target's window offset (or funcref index, §3c) + addend, written little-endian into the
// data image. A synthetic **`_start`** (function 0) takes the powerbox capability handles,
// stashes them in a reserved region, then calls `main` with the initial
// data-SP. **Stdio over the powerbox** (§3e):
// `write`/`read`/`exit` are recognized builtins lowered to `cap.call` on the stashed
// Stream/Exit handles — so real C reaches stdout/stdin and terminates with an exit code.
// **Floats** (`float`=f32, `double`/`long double`=f64) work too: arithmetic, compares,
// `-`/`!`, literals, and all int<->float / float<->float conversions (float->int is
// saturating, §3b). **`break`/`continue`** (a loop-context stack) and **`switch`** (a
// dispatch chain threading the value through `(sp, val)` compare blocks, with
// fall-through and `case` ranges) work. **Varargs** use a flat-buffer ABI (§3d): the
// caller marshals promoted args into a buffer between the frames and passes a hidden
// pointer; the callee sees it as `__va_area__` (see include/stdarg.h) — enough for a
// guest-C `printf`. Expression-level control flow (`&&`/`||`/`?:`) opens blocks, which
// would strand values computed earlier in the same C expression; such values are spilled
// to a per-frame scratch region (`eval2`/`spill`/`reload`) and reloaded in the merge
// block. `malloc`/`free` need no frontend support — they are ordinary guest C (a bump
// allocator over a window heap, §3d). **SSA promotion** (§3d "the pass that matters for
// speed") now runs the reverse of chibicc's allocate-all-locals-to-memory: a scalar local
// that is never address-taken becomes a real SSA value threaded as a block parameter (like
// the data-SP), so the JIT keeps it in a register instead of issuing a masked load/store
// per access — eliminating most of the loop-body memory traffic. chibicc's `A op= B`
// desugaring (`tmp = &A, *tmp = *tmp op B`) is un-desugared for plain-variable targets so
// loop counters/accumulators promote; address-taken locals, narrow types (char/short/
// _Bool, whose store truncation we keep in memory), aggregates, and `_Atomic` stay in
// memory. **Indirect calls** (C function pointers) lower to `call_indirect` through the
// function table (§3c): a function designator decays to its `ref.func` index (widened to
// the 8-byte pointer rep), and `fp(args)` wraps it back to the i32 table index and
// dispatches with the callee's static signature (incl. the leading data-SP). **General
// `goto`/labels** work: each C label maps to an IR block (allocated on first reference, so
// forward gotos resolve), and a `goto` branches to it threading the data-SP + promoted
// locals — the same SSA-block mechanism as `break`/`continue` and loops.
// anything unsupported is a hard error (so we never emit IR we cannot stand behind). This
// is enough C surface for a capable VM: globals, structs, pointers, loops, recursion,
// floats, varargs/`printf`, and heap allocation all run on interp and JIT.

#include "chibicc.h"

static FILE *o;
static int nv;    // next SSA value index in the *current block* (resets per block; the only
                  // values crossing a block are its parameters — the data-SP + promoted locals)
static int nb;        // next block label number in the current function
static int sret_param; // v-index of the hidden struct-return pointer (§3d sret), or -1
static bool term;     // is the current block already terminated?
static int cur_frame; // the current function's data-stack frame size (bytes)

// Stack of enclosing loops/switches, so `break`/`continue` (which chibicc lowers to a
// goto against the loop's brk/cont label) can branch to the right block.
struct LoopCtx {
  char *brk_label;
  int brk_blk;
  char *cont_label;
  int cont_blk;
};
static struct LoopCtx loopstk[64];
static int loopsp;

// Each `case`/`default` label gets its own IR block; this append-only map (keyed by the
// chibicc node) lets the body's ND_CASE find the block the switch dispatch branches to.
static Node *case_node[4096];
static int case_blk[4096];
static int ncase;
static int case_block_of(Node *n) {
  for (int i = 0; i < ncase; i++)
    if (case_node[i] == n)
      return case_blk[i];
  return -1; // unreachable: every case is registered before the body is emitted
}

// Each C label (and the `goto`s that target it) maps to one IR block, keyed by chibicc's
// resolved `unique_label`. A block number is allocated on first reference — whether that
// is the label or a (forward) `goto` — so forward gotos work (svm-text resolves block
// targets by name, not position). Reset per function.
static char *label_name[1024];
static int label_blk[1024];
static int nlabel;
static int label_block_of(char *uniq) {
  for (int i = 0; i < nlabel; i++)
    if (!strcmp(label_name[i], uniq))
      return label_blk[i];
  label_name[nlabel] = uniq;
  int b = nb++;
  label_blk[nlabel++] = b;
  return b;
}

// The data-stack pointer (frame base) is threaded as the first parameter of every IR
// function and every IR block — `v0` in each block (§3d). A local at frame offset N
// lives at `sp + N`; a call gives the callee a fresh frame at `sp + cur_frame`, so
// recursion/reentrancy never clobber a parent frame. (A real backend register-pins the
// data-SP; we make it an explicit value, relying on the masking lowering for §4 safety.)
#define SP "v0"

// The module's function definitions (main first). `call` targets a function by index.
// A synthetic `_start` is emitted as function 0 when `main` exists, so real functions
// start at `start_off` (1); `_start` sets up the data-SP and calls `main`.
static Obj *funcs[1024];
static int nfuncs;
static int start_off; // 1 if a `_start` occupies function index 0, else 0

// Globals + string literals live at fixed window offsets in the data region [16,
// data_end); the data stack starts at data_end (main's initial data-SP, baked into
// `_start`). The low 16 bytes are a runtime-reserved region: it holds the powerbox
// capability handles (so no global/local address is 0 = C NULL either).
static int data_end = 16;

// Capability handles the runtime hands `_start` (§3c/§3e) are stashed in the reserved
// region; the stdio builtins load them from here. Layout: stdout@0, stdin@4, exit@8.
#define STDOUT_SLOT 0
#define STDIN_SLOT 4
#define EXIT_SLOT 8

static int func_index(Obj *fn) {
  for (int i = 0; i < nfuncs; i++)
    if (funcs[i] == fn)
      return start_off + i;
  error_tok(fn->tok, "codegen_ir: call to a function with no definition (no linker yet)");
}

// --- SSA promotion (DESIGN §3d "the pass that matters for speed") ----------------------
//
// chibicc allocates every local to memory; we run the reverse pass and promote scalar
// locals that are never address-taken to real SSA values, so the JIT register-allocates
// them instead of issuing a masked load/store per access. State no longer lives only in
// memory, so a promoted local must cross block boundaries: like the data-SP (v0), each
// promoted local is threaded as a block parameter of *every* block, and its current SSA
// value is tracked per block. A promoted local with slot `s` is block parameter v(s+1)
// (right after the data-SP); merge/dispatch blocks carry their extra value after those.
//
// This "thread every promoted local through every block" shape is the same one already
// proven correct for the data-SP: it is SSA-valid by construction (each block parameter is
// the φ), so it needs no dominance/liveness analysis — Cranelift drops the dead ones.
#define MAXPROMO 256
static int npromo;                 // promoted locals in the current function
static char *promo_ty[MAXPROMO];   // IR type of each promoted slot
static int curval[MAXPROMO];       // current SSA value of each slot in the current block

// A local is promoted iff its frame offset was set to the sentinel -(slot+1) (see
// prepare_func); a real memory local keeps a non-negative offset.
static bool is_promoted(Obj *v) { return v->is_local && v->offset < 0; }
static int slot_of(Obj *v) { return -v->offset - 1; }

// Only full-width scalars are promoted: a narrow type (char/short/_Bool) would need its
// store-truncation re-emitted on every assignment, so it stays in memory where the narrow
// store/load already does it. Aggregates live by address; `_Atomic` needs real memory.
static bool promotable_ty(Type *ty) {
  if (ty->is_atomic)
    return false;
  switch (ty->kind) {
  case TY_INT:
  case TY_LONG:
  case TY_ENUM:
  case TY_PTR:
  case TY_FLOAT:
  case TY_DOUBLE:
  case TY_LDOUBLE:
    return true;
  default:
    return false;
  }
}

// The current block's promoted-local block-argument list (", vA, vB, ...") for a branch,
// and the matching parameter declaration (", vS: ty, ...") for a block header. Both return
// a pointer into a static buffer, so use the result before the next call.
static char *cvals(void) {
  static char buf[8192];
  int p = 0;
  buf[0] = '\0';
  for (int s = 0; s < npromo; s++)
    p += snprintf(buf + p, sizeof buf - p, ", v%d", curval[s]);
  return buf;
}
static char *cparams(void) {
  static char buf[8192];
  int p = 0;
  buf[0] = '\0';
  for (int s = 0; s < npromo; s++)
    p += snprintf(buf + p, sizeof buf - p, ", v%d: %s", s + 1, promo_ty[s]);
  return buf;
}

// Open a new IR block with label `id`: emit its header (the data-SP `v0` plus every
// promoted local as a parameter) and reset per-block state. Block labels resolve by name,
// so forward references are fine. On entry each promoted slot's value is its parameter.
static void open_block(int id) {
  fprintf(o, "block%d(" SP ": i64%s):\n", id, cparams());
  nv = npromo + 1; // v0 is the data-SP; v1..vN are the promoted locals
  for (int s = 0; s < npromo; s++)
    curval[s] = s + 1;
  term = false;
}

// The data stack (address-taken locals, §3d) lives in the window. We reserve the low
// bytes so a local's address is never 0 (C `NULL`), and lay locals out from there.
#define STACK_BASE 16

// Map an integer C type to its IR scalar type. (LP64: int=i32, long/pointer=i64, §3d.)
static char *irty(Type *ty) {
  switch (ty->kind) {
  case TY_BOOL:
  case TY_CHAR:
  case TY_SHORT:
  case TY_INT:
  case TY_ENUM:
    return "i32";
  case TY_LONG:
  case TY_PTR:
  case TY_ARRAY: // an array decays to its address (a pointer) in value context
  case TY_FUNC:  // a function decays to its funcref index, widened to the 8-byte ptr rep
    return "i64";
  case TY_FLOAT:
    return "f32";
  case TY_DOUBLE:
  case TY_LDOUBLE: // long double = f64 (no 80-bit; pinned, §3d)
    return "f64";
  default:
    error_tok(ty->name, "codegen_ir: unsupported type");
  }
}

// True if `ty` is a 64-bit value in our model (i64 or f64). Used within a known int- or
// float-only context, so the i64/f64 ambiguity never matters.
static bool is64(Type *ty) { return irty(ty)[1] == '6'; }

// True if `ty` is a floating-point type.
static bool is_flt(Type *ty) {
  return ty->kind == TY_FLOAT || ty->kind == TY_DOUBLE || ty->kind == TY_LDOUBLE;
}

static int gen_expr(Node *node); // emits the IR, returns the result's SSA index
static int gen_addr(Node *node); // emits the IR, returns the SSA index of an lvalue's address

// Per-function scratch region (last SCRATCH_BYTES of the frame) for spilling SSA values
// across expression-level control flow (&&/||/?:), which opens new blocks. Such a branch
// strands any value computed earlier in the same C expression (it lives in the old
// block), so we store it to scratch and reload it in the merge block.
#define SCRATCH_BYTES 512
static int cur_scratch; // frame offset of this function's scratch region
static int spill_top;   // next free scratch slot (8-byte slots, LIFO)

// True if evaluating `n` opens any block (so values live across it must be spilled).
static bool has_branch(Node *n) {
  if (!n)
    return false;
  if (n->kind == ND_LOGAND || n->kind == ND_LOGOR || n->kind == ND_COND)
    return true;
  if (has_branch(n->lhs) || has_branch(n->rhs) || has_branch(n->cond) ||
      has_branch(n->then) || has_branch(n->els))
    return true;
  for (Node *a = n->args; a; a = a->next)
    if (has_branch(a))
      return true;
  return false;
}

// Spill SSA value `v` (IR type `irt`) to the next scratch slot; return the slot index.
static int spill(int v, char *irt) {
  int idx = spill_top++;
  int off = nv++;
  fprintf(o, "  v%d = i64.const %d\n", off, cur_scratch + idx * 8);
  int a = nv++;
  fprintf(o, "  v%d = i64.add " SP " v%d\n", a, off);
  fprintf(o, "  %s.store v%d v%d\n", irt, a, v);
  return idx;
}

// Reload a spilled value from scratch slot `idx`.
static int reload(int idx, char *irt) {
  int off = nv++;
  fprintf(o, "  v%d = i64.const %d\n", off, cur_scratch + idx * 8);
  int a = nv++;
  fprintf(o, "  v%d = i64.add " SP " v%d\n", a, off);
  int r = nv++;
  fprintf(o, "  v%d = %s.load v%d\n", r, irt, a);
  return r;
}

// Evaluate `a` then `b`, returning both result indices valid in the final block: if `b`
// opens a block, spill `a` across it (via scratch memory, which all blocks share).
static void eval2(Node *a, Node *b, int *va, int *vb) {
  *va = gen_expr(a);
  if (has_branch(b)) {
    int save = spill_top;
    int idx = spill(*va, irty(a->ty));
    *vb = gen_expr(b);
    *va = reload(idx, irty(a->ty));
    spill_top = save;
  } else {
    *vb = gen_expr(b);
  }
}

// True if a value of type `ty` is held *by address* (arrays/aggregates, and functions):
// reading the lvalue yields its address, not a loaded scalar. A function designator
// decays to its funcref index (§3c) — its "value" is its address — exactly like an array.
static bool by_address(Type *ty) {
  return ty->kind == TY_ARRAY || ty->kind == TY_STRUCT || ty->kind == TY_UNION ||
         ty->kind == TY_FUNC;
}

// A by-value aggregate (struct/union): passed/returned via a hidden pointer (§3d D39).
static bool is_agg(Type *ty) { return ty->kind == TY_STRUCT || ty->kind == TY_UNION; }

// The IR type of a C parameter/argument *as passed*: a by-value aggregate goes by hidden
// pointer (i64); everything else is its ordinary value type (an array already decays).
static char *pass_irty(Type *ty) { return is_agg(ty) ? "i64" : irty(ty); }

// Load the C value of type `ty` from address `addr` (an SSA i64); return its SSA index.
static int gen_load(Type *ty, int addr) {
  if (by_address(ty))
    return addr; // arrays/aggregates decay to their address
  int r = nv++;
  switch (ty->kind) {
  case TY_BOOL:
    fprintf(o, "  v%d = i32.load8_u v%d\n", r, addr);
    break;
  case TY_CHAR:
    fprintf(o, "  v%d = i32.load8_%s v%d\n", r, ty->is_unsigned ? "u" : "s", addr);
    break;
  case TY_SHORT:
    fprintf(o, "  v%d = i32.load16_%s v%d\n", r, ty->is_unsigned ? "u" : "s", addr);
    break;
  case TY_INT:
  case TY_ENUM:
    fprintf(o, "  v%d = i32.load v%d\n", r, addr);
    break;
  case TY_LONG:
  case TY_PTR:
    fprintf(o, "  v%d = i64.load v%d\n", r, addr);
    break;
  case TY_FLOAT:
    fprintf(o, "  v%d = f32.load v%d\n", r, addr);
    break;
  case TY_DOUBLE:
  case TY_LDOUBLE:
    fprintf(o, "  v%d = f64.load v%d\n", r, addr);
    break;
  default:
    error_tok(ty->name, "codegen_ir: unsupported load type");
  }
  return r;
}

// Store SSA value `val` of type `ty` to address `addr` (an SSA i64). Narrow stores keep
// the low bytes (matching C truncation on assignment).
static void gen_store(Type *ty, int addr, int val) {
  switch (ty->kind) {
  case TY_BOOL:
  case TY_CHAR:
    fprintf(o, "  i32.store8 v%d v%d\n", addr, val);
    break;
  case TY_SHORT:
    fprintf(o, "  i32.store16 v%d v%d\n", addr, val);
    break;
  case TY_INT:
  case TY_ENUM:
    fprintf(o, "  i32.store v%d v%d\n", addr, val);
    break;
  case TY_LONG:
  case TY_PTR:
    fprintf(o, "  i64.store v%d v%d\n", addr, val);
    break;
  case TY_FLOAT:
    fprintf(o, "  f32.store v%d v%d\n", addr, val);
    break;
  case TY_DOUBLE:
  case TY_LDOUBLE:
    fprintf(o, "  f64.store v%d v%d\n", addr, val);
    break;
  default:
    error_tok(ty->name, "codegen_ir: unsupported store type");
  }
}

// Copy `size` bytes from address `src` to address `dst` (both SSA i64), greedily in
// 8/4/2/1-byte chunks. Used for by-value aggregate copies (§3d D39): struct args/returns
// and whole-struct assignment. `dst`/`src` must not overlap (distinct objects), except a
// self-copy `dst == src`, which is a harmless identity.
static void gen_memcpy(int dst, int src, int size) {
  for (int i = 0; i < size;) {
    int chunk = (size - i >= 8) ? 8 : (size - i >= 4) ? 4 : (size - i >= 2) ? 2 : 1;
    char *ld = chunk == 8   ? "i64.load"
               : chunk == 4 ? "i32.load"
               : chunk == 2 ? "i32.load16_u"
                            : "i32.load8_u";
    char *st = chunk == 8   ? "i64.store"
               : chunk == 4 ? "i32.store"
               : chunk == 2 ? "i32.store16"
                            : "i32.store8";
    int off = nv++;
    fprintf(o, "  v%d = i64.const %d\n", off, i);
    int sa = nv++;
    fprintf(o, "  v%d = i64.add v%d v%d\n", sa, src, off);
    int val = nv++;
    fprintf(o, "  v%d = %s v%d\n", val, ld, sa);
    int da = nv++;
    fprintf(o, "  v%d = i64.add v%d v%d\n", da, dst, off);
    fprintf(o, "  %s v%d v%d\n", st, da, val);
    i += chunk;
  }
}

// The address of an lvalue, as an SSA i64.
static int gen_addr(Node *node) {
  switch (node->kind) {
  case ND_VAR: {
    if (node->var->is_function) {
      // A function designator decays to its funcref index (§3c): `ref.func` yields the
      // i32 function-table index; widen it to the 8-byte C pointer representation
      // (function pointers are stored as integers in memory, §3d).
      int rf = nv++;
      fprintf(o, "  v%d = ref.func %d\n", rf, func_index(node->var));
      int r = nv++;
      fprintf(o, "  v%d = i64.extend_i32_u v%d\n", r, rf);
      return r;
    }
    if (node->var->is_local) {
      if (is_promoted(node->var))
        error_tok(node->tok, "codegen_ir: internal — address of a promoted local");
      // The local lives at sp + frame-offset (§3d data stack). Emit the const (lower
      // index) before the add that uses it, so value numbering stays monotonic.
      int off = nv++;
      fprintf(o, "  v%d = i64.const %d\n", off, node->var->offset);
      int r = nv++;
      fprintf(o, "  v%d = i64.add " SP " v%d\n", r, off);
      return r;
    }
    // A global lives at a fixed window offset in the data region below the stack.
    int r = nv++;
    fprintf(o, "  v%d = i64.const %d\n", r, node->var->offset);
    return r;
  }
  case ND_DEREF:
    return gen_expr(node->lhs); // the pointer value *is* the address
  case ND_FUNCALL:
    // A struct/union-returning call: its lvalue is the sret buffer gen_expr writes to.
    return gen_expr(node);
  case ND_COMMA:
    gen_expr(node->lhs);
    return gen_addr(node->rhs);
  case ND_MEMBER: {
    // &(s.field) = &s + field offset.
    int base = gen_addr(node->lhs);
    int off = nv++;
    fprintf(o, "  v%d = i64.const %d\n", off, node->member->offset);
    int r = nv++;
    fprintf(o, "  v%d = i64.add v%d v%d\n", r, base, off);
    return r;
  }
  default:
    error_tok(node->tok, "codegen_ir: expression is not an lvalue");
  }
}

// Emit `vR = <prefix>.<op> vA vB` over the operands' (common) width; return R.
static int binop(Node *node, char *op) {
  int a, b;
  eval2(node->lhs, node->rhs, &a, &b);
  int r = nv++;
  fprintf(o, "  v%d = %s.%s v%d v%d\n", r, irty(node->lhs->ty), op, a, b);
  return r;
}

// A comparison: the op width is the operands' type; the result is always i32 0/1. Integer
// `lt`/`le` take a signedness suffix; float compares (and `eq`/`ne`) do not.
static int cmpop(Node *node, char *base, bool sign) {
  int a, b;
  eval2(node->lhs, node->rhs, &a, &b);
  int r = nv++;
  Type *ot = node->lhs->ty;
  char *p = irty(ot);
  if (sign && !is_flt(ot))
    fprintf(o, "  v%d = %s.%s_%s v%d v%d\n", r, p, base, ot->is_unsigned ? "u" : "s", a, b);
  else
    fprintf(o, "  v%d = %s.%s v%d v%d\n", r, p, base, a, b);
  return r;
}

// Convert SSA value `a` of type `from` to type `to` (the C "usual conversions"): int<->
// int (extend/wrap), float<->float (promote/demote), and int<->float. float->int is
// **saturating** (`trunc_sat`) so an out-of-range conversion is total, not a trap (§3b).
static int gen_convert(int a, Type *from, Type *to) {
  // An aggregate (struct/union) value is an *address*; casting one to a scalar reinterprets
  // its bytes — chibicc emits this when it initializes a union via its first member
  // (`v.i = (int)expr`) — so load the scalar through the address.
  if (is_agg(from) && !by_address(to))
    return gen_load(to, a);
  // Otherwise anything held by address (array/function decay, or an aggregate→aggregate
  // copy handled elsewhere by memcpy) converts with no value change.
  if (by_address(from) || by_address(to))
    return a;
  bool ff = is_flt(from), tf = is_flt(to);
  if (is64(from) == is64(to) && ff == tf)
    return a; // same IR type — no-op
  int r = nv++;
  if (!ff && !tf) {
    if (is64(to))
      fprintf(o, "  v%d = i64.extend_i32_%s v%d\n", r, from->is_unsigned ? "u" : "s", a);
    else
      fprintf(o, "  v%d = i32.wrap_i64 v%d\n", r, a);
  } else if (ff && tf) {
    fprintf(o, "  v%d = %s v%d\n", r, is64(to) ? "f64.promote_f32" : "f32.demote_f64", a);
  } else if (!ff && tf) {
    fprintf(o, "  v%d = %s.convert_%s_%s v%d\n", r, irty(to), irty(from),
            from->is_unsigned ? "u" : "s", a);
  } else {
    fprintf(o, "  v%d = %s.trunc_sat_%s_%s v%d\n", r, irty(to), irty(from),
            to->is_unsigned ? "u" : "s", a);
  }
  return r;
}

static int gen_cast(Node *node) {
  // A cast to void just discards the value (after evaluating it for side effects).
  if (node->ty->kind == TY_VOID)
    return gen_expr(node->lhs);
  return gen_convert(gen_expr(node->lhs), node->lhs->ty, node->ty);
}

// Evaluate `node` and convert the result to `target` (used for the `?:` arms).
static int gen_expr_as(Node *node, Type *target) {
  return gen_convert(gen_expr(node), node->ty, target);
}

// Evaluate `node` to an i32 truth value (0/1): `(v != 0)` over the operand's width.
static int gen_truth(Node *node) {
  int v = gen_expr(node);
  char *p = irty(node->ty);
  int z = nv++;
  fprintf(o, "  v%d = %s.const 0\n", z, p);
  int r = nv++;
  fprintf(o, "  v%d = %s.ne v%d v%d\n", r, p, v, z);
  return r;
}

// Open a merge/dispatch block taking `(sp, <promoted locals>, vR: ty)`: the carried value
// vR follows the promoted locals at index npromo+1, and nv resumes after it.
#define MERGE_VAL (npromo + 1)
static void open_merge(int id, char *ty) {
  fprintf(o, "block%d(" SP ": i64%s, v%d: %s):\n", id, cparams(), MERGE_VAL, ty);
  nv = npromo + 2;
  for (int s = 0; s < npromo; s++)
    curval[s] = s + 1;
  term = false;
}

// `a && b` and `a || b` → i32 0/1, short-circuit; the result is carried into the merge.
static int gen_logand(Node *node) {
  int ta = gen_truth(node->lhs);
  int rhs = nb++, fls = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", ta, rhs, cvals(), fls,
          cvals());
  open_block(rhs);
  int tb = gen_truth(node->rhs);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), tb);
  open_block(fls);
  int z = nv++;
  fprintf(o, "  v%d = i32.const 0\n", z);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), z);
  open_merge(merge, "i32");
  return MERGE_VAL;
}

static int gen_logor(Node *node) {
  int ta = gen_truth(node->lhs);
  int tru = nb++, rhs = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", ta, tru, cvals(), rhs,
          cvals());
  open_block(tru);
  int one = nv++;
  fprintf(o, "  v%d = i32.const 1\n", one);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), one);
  open_block(rhs);
  int tb = gen_truth(node->rhs);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), tb);
  open_merge(merge, "i32");
  return MERGE_VAL;
}

// `cond ? then : els` → branches converted to the result type, carried into the merge.
static int gen_cond(Node *node) {
  int c = gen_truth(node->cond);
  int th = nb++, el = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", c, th, cvals(), el,
          cvals());

  if (node->ty->kind == TY_VOID) {
    // A void `?:` — both arms are evaluated for effect only, no carried value.
    open_block(th);
    gen_expr(node->then);
    if (!term)
      fprintf(o, "  br block%d(" SP "%s)\n", merge, cvals());
    open_block(el);
    gen_expr(node->els);
    if (!term)
      fprintf(o, "  br block%d(" SP "%s)\n", merge, cvals());
    open_block(merge);
    return 0;
  }

  open_block(th);
  int vt = gen_expr_as(node->then, node->ty);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), vt);
  open_block(el);
  int ve = gen_expr_as(node->els, node->ty);
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", merge, cvals(), ve);
  open_merge(merge, irty(node->ty));
  return MERGE_VAL;
}

// Widen an i32 value to i64 (for capability args that cross as i64 slots).
static int widen_i64(int v, Type *ty) {
  if (is64(ty))
    return v;
  int r = nv++;
  fprintf(o, "  v%d = i64.extend_i32_%s v%d\n", r, ty->is_unsigned ? "u" : "s", v);
  return r;
}

// Load a stashed capability handle from the reserved region.
static int load_handle(int slot) {
  int a = nv++;
  fprintf(o, "  v%d = i64.const %d\n", a, slot);
  int h = nv++;
  fprintf(o, "  v%d = i32.load v%d\n", h, a);
  return h;
}

// Stdio builtins map directly onto the powerbox (§3e): the lowest libc layer. `write`/
// `read` → Stream.write/read on the stdout/stdin handle (fd is ignored for now — always
// the std stream); `exit` → Exit then `unreachable`. A function with these names need
// only be *declared* in the C source; we intercept the call instead of emitting `call`.
static int gen_builtin_stream(Node *node, int slot, int op) {
  Node *a = node->args;
  if (!a || !a->next || !a->next->next)
    error_tok(node->tok, "codegen_ir: write/read(fd, buf, len) expects 3 arguments");
  gen_expr(a); // fd — evaluated for effect, then ignored (always the std stream)
  int buf, lenv;
  eval2(a->next, a->next->next, &buf, &lenv);
  int len = widen_i64(lenv, a->next->next->ty);
  int h = load_handle(slot);
  int r = nv++;
  fprintf(o, "  v%d = cap.call 0 %d (i64, i64) -> (i64) v%d (v%d, v%d)\n", r, op, h, buf, len);
  if (node->ty->kind == TY_VOID)
    return 0;
  if (is64(node->ty))
    return r;
  int w = nv++;
  fprintf(o, "  v%d = i32.wrap_i64 v%d\n", w, r);
  return w;
}

static int gen_builtin_exit(Node *node) {
  if (!node->args)
    error_tok(node->tok, "codegen_ir: exit(code) expects 1 argument");
  int code = gen_expr(node->args);
  int h = load_handle(EXIT_SLOT);
  fprintf(o, "  cap.call 1 0 (i32) -> () v%d (v%d)\n", h, code);
  fprintf(o, "  unreachable\n"); // Exit is noreturn (§3e)
  term = true;
  return 0;
}

static int gen_expr(Node *node) {
  switch (node->kind) {
  case ND_NUM: {
    int r = nv++;
    if (is_flt(node->ty))
      fprintf(o, "  v%d = %s.const %.17g\n", r, irty(node->ty), (double)node->fval);
    else
      fprintf(o, "  v%d = %s.const %ld\n", r, irty(node->ty), (long)node->val);
    return r;
  }
  case ND_ADD:
    return binop(node, "add");
  case ND_SUB:
    return binop(node, "sub");
  case ND_MUL:
    return binop(node, "mul");
  case ND_DIV:
    return binop(node, is_flt(node->ty) ? "div" : node->ty->is_unsigned ? "div_u" : "div_s");
  case ND_MOD:
    return binop(node, node->ty->is_unsigned ? "rem_u" : "rem_s");
  case ND_BITAND:
    return binop(node, "and");
  case ND_BITOR:
    return binop(node, "or");
  case ND_BITXOR:
    return binop(node, "xor");
  case ND_SHL:
    return binop(node, "shl");
  case ND_SHR:
    return binop(node, node->lhs->ty->is_unsigned ? "shr_u" : "shr_s");
  case ND_EQ:
    return cmpop(node, "eq", false);
  case ND_NE:
    return cmpop(node, "ne", false);
  case ND_LT:
    return cmpop(node, "lt", true);
  case ND_LE:
    return cmpop(node, "le", true);
  case ND_NEG: {
    int a = gen_expr(node->lhs);
    char *p = irty(node->ty);
    int r = nv++;
    if (is_flt(node->ty)) {
      fprintf(o, "  v%d = %s.neg v%d\n", r, p, a);
    } else {
      // -x  ==  0 - x
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n", z, p);
      fprintf(o, "  v%d = %s.sub v%d v%d\n", r, p, z, a);
    }
    return r;
  }
  case ND_NOT: {
    // !x  ==  (x == 0), result i32
    int a = gen_expr(node->lhs);
    Type *ot = node->lhs->ty;
    int r = nv++;
    if (is_flt(ot)) {
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n", z, irty(ot));
      fprintf(o, "  v%d = %s.eq v%d v%d\n", r, irty(ot), a, z);
    } else {
      fprintf(o, "  v%d = %s.eqz v%d\n", r, irty(ot), a);
    }
    return r;
  }
  case ND_BITNOT: {
    // ~x  ==  x ^ -1
    int a = gen_expr(node->lhs);
    char *p = irty(node->ty);
    int m = nv++;
    fprintf(o, "  v%d = %s.const -1\n", m, p);
    int r = nv++;
    fprintf(o, "  v%d = %s.xor v%d v%d\n", r, p, a, m);
    return r;
  }
  case ND_CAST:
    return gen_cast(node);
  case ND_LOGAND:
    return gen_logand(node);
  case ND_LOGOR:
    return gen_logor(node);
  case ND_COND:
    return gen_cond(node);
  case ND_COMMA:
    gen_expr(node->lhs);
    return gen_expr(node->rhs);
  case ND_VAR:
    if (node->var->is_local && is_promoted(node->var))
      return curval[slot_of(node->var)]; // a promoted local is its current SSA value
    return gen_load(node->ty, gen_addr(node));
  case ND_DEREF:
    return gen_load(node->ty, gen_addr(node));
  case ND_MEMBER:
    return gen_load(node->ty, gen_addr(node));
  case ND_ADDR:
    return gen_addr(node->lhs);
  case ND_FUNCALL: {
    // A call is direct when the callee is a named function; otherwise it is an indirect
    // call through a function-pointer *value* (a funcref index, §3c).
    bool direct = node->lhs->kind == ND_VAR && node->lhs->var->is_function;
    // Intercept the stdio builtins (powerbox §3e) before treating it as a guest call.
    if (direct) {
      char *fname = node->lhs->var->name;
      if (fname) {
        if (!strcmp(fname, "write"))
          return gen_builtin_stream(node, STDOUT_SLOT, 1);
        if (!strcmp(fname, "read"))
          return gen_builtin_stream(node, STDIN_SLOT, 0);
        if (!strcmp(fname, "exit") || !strcmp(fname, "_exit"))
          return gen_builtin_exit(node);
      }
    }
    // Evaluate the arguments (already cast to parameter types / default-promoted by the
    // parser). Keep their types too, for marshalling variadic args. If any argument — or,
    // for an indirect call, the callee expression — opens a block, spill each live value
    // across the rest so they all land in the final block.
    int argv[64];
    Type *argt[64];
    int n = 0;
    bool argbranch = false;
    if (!direct)
      argbranch |= has_branch(node->lhs);
    for (Node *a = node->args; a; a = a->next)
      argbranch |= has_branch(a);
    int spillsave = spill_top;
    // The indirect callee (a funcref) is evaluated first and kept live across the args.
    int fnval = 0, fnslot = 0;
    if (!direct) {
      fnval = gen_expr(node->lhs);
      if (argbranch)
        fnslot = spill(fnval, "i64");
    }
    int spillslot[64];
    for (Node *a = node->args; a; a = a->next) {
      if (n == 64)
        error_tok(node->tok, "codegen_ir: too many call arguments");
      argt[n] = a->ty;
      argv[n] = gen_expr(a); // a by-value aggregate yields its address (passed by pointer)
      if (argbranch)
        spillslot[n] = spill(argv[n], pass_irty(a->ty));
      n++;
    }
    if (argbranch) {
      if (!direct)
        fnval = reload(fnslot, "i64");
      for (int i = 0; i < n; i++)
        argv[i] = reload(spillslot[i], pass_irty(argt[i]));
      spill_top = spillsave;
    }

    bool variadic = node->func_ty && node->func_ty->is_variadic;
    int nfixed = n;
    int vbuf = 0; // the marshalled-varargs buffer pointer (passed as the trailing arg)
    int extra = 0;
    if (variadic) {
      nfixed = 0;
      for (Type *pt = node->func_ty->params; pt; pt = pt->next)
        nfixed++;
      int nva = n - nfixed;
      // Marshal the variadic args into a buffer just above our frame (and below the
      // callee's): one promoted 8-byte slot each (§3d).
      int fc = nv++;
      fprintf(o, "  v%d = i64.const %d\n", fc, cur_frame);
      vbuf = nv++;
      fprintf(o, "  v%d = i64.add " SP " v%d\n", vbuf, fc);
      for (int j = 0; j < nva; j++) {
        int v = argv[nfixed + j];
        Type *t = argt[nfixed + j];
        int addr = vbuf;
        if (j > 0) {
          int o2 = nv++;
          fprintf(o, "  v%d = i64.const %d\n", o2, j * 8);
          addr = nv++;
          fprintf(o, "  v%d = i64.add v%d v%d\n", addr, vbuf, o2);
        }
        if (is_flt(t)) {
          if (!is64(t)) { // promote float -> double (defensive; parser usually did)
            int p = nv++;
            fprintf(o, "  v%d = f64.promote_f32 v%d\n", p, v);
            v = p;
          }
          fprintf(o, "  f64.store v%d v%d\n", addr, v);
        } else if (is64(t)) {
          fprintf(o, "  i64.store v%d v%d\n", addr, v);
        } else {
          fprintf(o, "  i32.store v%d v%d\n", addr, v);
        }
      }
      extra = align_to(nva * 8, 16); // the callee frame sits above the buffer
    }

    // The callee gets a fresh frame above ours (and above any varargs buffer).
    int fs = nv++;
    fprintf(o, "  v%d = i64.const %d\n", fs, cur_frame + extra);
    int csp = nv++;
    fprintf(o, "  v%d = i64.add " SP " v%d\n", csp, fs);

    bool is_void = node->ty->kind == TY_VOID;
    // A struct/union return uses the §3d sret ABI: the caller passes the address of its
    // return buffer as a hidden first argument (right after the data-SP), the callee writes
    // the result through it, and the IR call yields no value.
    bool agg_ret = is_agg(node->ty);
    // For an indirect call, wrap the 8-byte funcref down to the i32 table index, and (for a
    // struct return) materialize the sret buffer address — both *before* allocating the
    // result index, so block-local value numbering stays monotonic (operands precede it).
    int idx32 = -1;
    if (!direct) {
      idx32 = nv++;
      fprintf(o, "  v%d = i32.wrap_i64 v%d\n", idx32, fnval);
    }
    int sret_addr = 0;
    if (agg_ret) {
      int so = nv++;
      fprintf(o, "  v%d = i64.const %d\n", so, node->ret_buffer->offset);
      sret_addr = nv++;
      fprintf(o, "  v%d = i64.add " SP " v%d\n", sret_addr, so); // buffer in the caller frame
    }
    bool ir_void = is_void || agg_ret; // a struct-returning call is void at the IR level
    int r = ir_void ? 0 : nv++;
    if (direct) {
      int idx = func_index(node->lhs->var);
      if (ir_void)
        fprintf(o, "  call %d (v%d", idx, csp);
      else
        fprintf(o, "  v%d = call %d (v%d", r, idx, csp);
    } else {
      // Indirect dispatch through the function table (§3c): call with the callee's static
      // signature, which must match the target's exactly — leading data-SP i64, then the
      // hidden sret pointer (struct return), the params, and the trailing varargs pointer —
      // or the runtime type-id check traps (a forged or mismatched index is inert).
      if (ir_void)
        fprintf(o, "  call_indirect (i64");
      else
        fprintf(o, "  v%d = call_indirect (i64", r);
      if (agg_ret)
        fprintf(o, ", i64"); // the hidden sret pointer
      for (Type *pt = node->func_ty->params; pt; pt = pt->next)
        fprintf(o, ", %s", pass_irty(pt));
      if (variadic)
        fprintf(o, ", i64"); // the hidden varargs-buffer pointer
      fprintf(o, ") -> (%s) v%d(v%d", ir_void ? "" : irty(node->ty), idx32, csp);
    }
    if (agg_ret)
      fprintf(o, ", v%d", sret_addr); // the hidden sret arg, right after the data-SP
    for (int i = 0; i < nfixed; i++)
      fprintf(o, ", v%d", argv[i]);
    if (variadic)
      fprintf(o, ", v%d", vbuf); // the hidden varargs-buffer pointer
    fprintf(o, ")\n");
    // The call's value: a struct return is the sret buffer address; a void call's result is
    // discarded; otherwise the IR result.
    return agg_ret ? sret_addr : r;
  }
  case ND_ASSIGN: {
    // Assigning a promoted local just rebinds its current SSA value — no store. The rhs
    // was cast to the lhs type by the parser, so its IR type already matches the slot.
    if (node->lhs->kind == ND_VAR && node->lhs->var->is_local &&
        is_promoted(node->lhs->var)) {
      int val = gen_expr(node->rhs);
      curval[slot_of(node->lhs->var)] = val;
      return val;
    }
    // Whole-struct/union assignment is a byte copy (§3d D39): the rhs yields its address.
    if (is_agg(node->ty)) {
      int dst = gen_addr(node->lhs);
      int src;
      if (has_branch(node->rhs)) {
        int save = spill_top;
        int idx = spill(dst, "i64");
        src = gen_expr(node->rhs);
        dst = reload(idx, "i64");
        spill_top = save;
      } else {
        src = gen_expr(node->rhs);
      }
      gen_memcpy(dst, src, node->ty->size);
      return dst; // the assignment's value is the aggregate, used by address
    }
    int addr = gen_addr(node->lhs);
    int val;
    if (has_branch(node->rhs)) {
      int save = spill_top;
      int idx = spill(addr, "i64");
      val = gen_expr(node->rhs);
      addr = reload(idx, "i64");
      spill_top = save;
    } else {
      val = gen_expr(node->rhs);
    }
    gen_store(node->lhs->ty, addr, val);
    return val; // an assignment is an expression yielding the stored value
  }
  case ND_NULL_EXPR: {
    // "Do nothing" (e.g. a non-VLA size computation). Materialize a harmless value so
    // the (always-discarded) result index is still valid.
    int r = nv++;
    fprintf(o, "  v%d = i32.const 0\n", r);
    return r;
  }
  case ND_MEMZERO: {
    // A promoted local is zero-initialized by binding it to a typed zero (no store).
    if (node->var->is_local && is_promoted(node->var)) {
      int s = slot_of(node->var);
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n", z, promo_ty[s]);
      curval[s] = z;
      return z;
    }
    // Zero-initialize the local's whole frame region (§3d data stack). ND_MEMZERO is only
    // emitted for stack locals, so the address is sp-relative: sp + (offset + i).
    int sz = node->var->ty->size;
    int base = node->var->offset;
    for (int i = 0; i < sz;) {
      int off = nv++;
      fprintf(o, "  v%d = i64.const %d\n", off, base + i);
      int a = nv++;
      fprintf(o, "  v%d = i64.add " SP " v%d\n", a, off);
      int z = nv++;
      if (sz - i >= 8) {
        fprintf(o, "  v%d = i64.const 0\n  i64.store v%d v%d\n", z, a, z);
        i += 8;
      } else if (sz - i >= 4) {
        fprintf(o, "  v%d = i32.const 0\n  i32.store v%d v%d\n", z, a, z);
        i += 4;
      } else if (sz - i >= 2) {
        fprintf(o, "  v%d = i32.const 0\n  i32.store16 v%d v%d\n", z, a, z);
        i += 2;
      } else {
        fprintf(o, "  v%d = i32.const 0\n  i32.store8 v%d v%d\n", z, a, z);
        i += 1;
      }
    }
    int r = nv++;
    fprintf(o, "  v%d = i32.const 0\n", r); // discarded result
    return r;
  }
  default:
    error_tok(node->tok, "codegen_ir: unsupported expression (kind=%d)", node->kind);
  }
}

static void gen_stmt(Node *node);

// `if (cond) then [else els]` → a diamond of param-free blocks (state is in memory).
static void gen_if(Node *node) {
  int c = gen_truth(node->cond); // normalize to an i32 0/1 br_if condition
  int t = nb++, e = nb++, end = nb++;
  fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", c, t, cvals(), e, cvals());
  term = true;

  open_block(t);
  gen_stmt(node->then);
  if (!term)
    fprintf(o, "  br block%d(" SP "%s)\n", end, cvals());

  open_block(e);
  if (node->els)
    gen_stmt(node->els);
  if (!term)
    fprintf(o, "  br block%d(" SP "%s)\n", end, cvals());

  open_block(end);
}

// `for (init; cond; inc) body` (and `while`, with init/inc absent): cond/body/cont/end
// blocks with a back-edge. `continue` targets the `cont` block (which runs `inc` then
// re-tests `cond`); `break` targets `end`.
static void gen_for(Node *node) {
  if (node->init)
    gen_stmt(node->init);
  int cond = nb++, body = nb++, cont = nb++, end = nb++;
  fprintf(o, "  br block%d(" SP "%s)\n", cond, cvals());
  term = true;

  open_block(cond);
  if (node->cond) {
    int c = gen_truth(node->cond); // normalize to an i32 0/1 br_if condition
    fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", c, body, cvals(), end,
            cvals());
  } else {
    fprintf(o, "  br block%d(" SP "%s)\n", body, cvals()); // `for(;;)` — unconditional
  }
  term = true;

  open_block(body);
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, node->cont_label, cont};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP "%s)\n", cont, cvals());

  open_block(cont);
  if (node->inc)
    gen_expr(node->inc);
  fprintf(o, "  br block%d(" SP "%s)\n", cond, cvals());

  open_block(end);
}

// `do body while (cond)`: body runs once, then `cont` re-tests. `break` → end.
static void gen_do(Node *node) {
  int body = nb++, cont = nb++, end = nb++;
  fprintf(o, "  br block%d(" SP "%s)\n", body, cvals());
  term = true;

  open_block(body);
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, node->cont_label, cont};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP "%s)\n", cont, cvals());

  open_block(cont);
  int c = gen_truth(node->cond); // normalize to an i32 0/1 br_if condition
  fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s)\n", c, body, cvals(), end,
          cvals());

  open_block(end);
}

// `switch (cond) { case ...: ... }` — a dispatch chain that threads the switch value
// through `(sp, val)` compare blocks (values can't otherwise cross blocks), branching to
// each case's block; the body's ND_CASE labels open those blocks and fall through.
static void gen_switch(Node *node) {
  int v = gen_expr(node->cond);
  char *p = irty(node->cond->ty);

  for (Node *c = node->case_next; c; c = c->case_next) {
    case_node[ncase] = c;
    case_blk[ncase++] = nb++;
  }
  int end = nb++;
  int defblk = node->default_case ? nb++ : end;
  if (node->default_case) {
    case_node[ncase] = node->default_case;
    case_blk[ncase++] = defblk;
  }

  // Dispatch: each compare block carries (sp, <promoted locals>, val) and forwards the
  // value (at index MERGE_VAL, after the promoted locals) to the next compare block.
  int check = nb++;
  fprintf(o, "  br block%d(" SP "%s, v%d)\n", check, cvals(), v);
  term = true;
  for (Node *c = node->case_next; c; c = c->case_next) {
    open_merge(check, p);
    int val = MERGE_VAL;
    int next = nb++;
    int hit = nv++;
    if (c->begin == c->end) {
      int k = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", k, p, c->begin);
      fprintf(o, "  v%d = %s.eq v%d v%d\n", hit, p, val, k);
    } else {
      // [GNU] case range begin..end: (val - begin) <=u (end - begin)
      int kb = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", kb, p, c->begin);
      int d = nv++;
      fprintf(o, "  v%d = %s.sub v%d v%d\n", d, p, val, kb);
      int kr = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", kr, p, c->end - c->begin);
      fprintf(o, "  v%d = %s.le_u v%d v%d\n", hit, p, d, kr);
    }
    fprintf(o, "  br_if v%d block%d(" SP "%s) block%d(" SP "%s, v%d)\n", hit,
            case_block_of(c), cvals(), next, cvals(), val);
    check = next;
  }
  // No case matched → default (or break past the switch).
  open_merge(check, p);
  fprintf(o, "  br block%d(" SP "%s)\n", defblk, cvals());
  term = true;

  // The body: ND_CASE labels open their blocks; `break` (cont_label NULL so `continue`
  // passes through to an enclosing loop) targets `end`.
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, NULL, -1};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP "%s)\n", end, cvals());
  open_block(end);
}

static void gen_stmt(Node *node) {
  switch (node->kind) {
  case ND_BLOCK:
    for (Node *n = node->body; n; n = n->next) {
      // Drop dead code after a terminator — but a `case`/`default` or a `goto` label
      // reopens a reachable block, so it is always emitted.
      if (term && n->kind != ND_CASE && n->kind != ND_LABEL)
        continue;
      gen_stmt(n);
    }
    return;
  case ND_SWITCH:
    gen_switch(node);
    return;
  case ND_CASE: {
    // A case/default label: fall-through from the previous case branches in here.
    int blk = case_block_of(node);
    if (!term)
      fprintf(o, "  br block%d(" SP "%s)\n", blk, cvals());
    open_block(blk);
    gen_stmt(node->lhs);
    return;
  }
  case ND_EXPR_STMT:
    gen_expr(node->lhs); // value discarded
    return;
  case ND_IF:
    gen_if(node);
    return;
  case ND_FOR:
    gen_for(node);
    return;
  case ND_DO:
    gen_do(node);
    return;
  case ND_LABEL: {
    // A C label: its block is a `goto`/fall-through target. Fall into it from the
    // preceding statement (if reachable), open it, then emit the labelled statement.
    int blk = label_block_of(node->unique_label);
    if (!term)
      fprintf(o, "  br block%d(" SP "%s)\n", blk, cvals());
    open_block(blk);
    gen_stmt(node->lhs);
    return;
  }
  case ND_GOTO: {
    // break/continue: branch to the matching enclosing loop's break/continue block.
    for (int i = loopsp - 1; i >= 0; i--) {
      if (node->unique_label && loopstk[i].brk_label &&
          !strcmp(node->unique_label, loopstk[i].brk_label)) {
        fprintf(o, "  br block%d(" SP "%s)\n", loopstk[i].brk_blk, cvals());
        term = true;
        return;
      }
      if (node->unique_label && loopstk[i].cont_label &&
          !strcmp(node->unique_label, loopstk[i].cont_label)) {
        fprintf(o, "  br block%d(" SP "%s)\n", loopstk[i].cont_blk, cvals());
        term = true;
        return;
      }
    }
    // A general `goto`: branch to its target label's block (allocated on first reference,
    // so forward gotos resolve). The data-SP + promoted locals thread through as args.
    fprintf(o, "  br block%d(" SP "%s)\n", label_block_of(node->unique_label), cvals());
    term = true;
    return;
  }
  case ND_RETURN:
    if (node->lhs && is_agg(node->lhs->ty)) {
      // struct/union return: copy the result into the caller's sret buffer (§3d), then
      // return no value (the IR function is `-> ()`).
      int src = gen_expr(node->lhs); // an aggregate yields its address (by_address)
      gen_memcpy(sret_param, src, node->lhs->ty->size);
      fprintf(o, "  return\n");
    } else if (node->lhs) {
      int v = gen_expr(node->lhs);
      fprintf(o, "  return v%d\n", v);
    } else {
      fprintf(o, "  return\n");
    }
    term = true;
    return;
  default:
    error_tok(node->tok, "codegen_ir: unsupported statement (kind=%d)", node->kind);
  }
}

// The set of locals whose address is taken (so they cannot be promoted), collected per
// function by `scan` below.
static Obj *ataken[4096];
static int n_ataken;
static void mark_ataken(Obj *v) {
  for (int i = 0; i < n_ataken; i++)
    if (ataken[i] == v)
      return;
  if (n_ataken < 4096)
    ataken[n_ataken++] = v;
}
static bool is_ataken(Obj *v) {
  for (int i = 0; i < n_ataken; i++)
    if (ataken[i] == v)
      return true;
  return false;
}

// Walk the AST and mark every local whose address is taken with `&`. Anything reachable
// only through an address (e.g. `&a[i]` reads `i` but takes the array's address) is found
// by the recursion. `&local` is the only way a *scalar* local's address escapes here.
static void scan(Node *n) {
  if (!n)
    return;
  if (n->kind == ND_ADDR && n->lhs && n->lhs->kind == ND_VAR && n->lhs->var->is_local)
    mark_ataken(n->lhs->var);
  scan(n->lhs);
  scan(n->rhs);
  scan(n->cond);
  scan(n->then);
  scan(n->els);
  scan(n->init);
  scan(n->inc);
  for (Node *b = n->body; b; b = b->next)
    scan(b);
  for (Node *a = n->args; a; a = a->next)
    scan(a);
}

// True for chibicc's synthetic unnamed locals (e.g. the `tmp = &A` temporary it injects
// for compound assignment); they have an empty name, which a real C variable never has.
static bool is_synthetic(Obj *v) { return v->is_local && v->name && v->name[0] == '\0'; }

// Within `n`, repoint any `*tmp` (DEREF of the synthetic pointer `tmp`) to the lvalue `A`.
static void repoint_deref(Node *n, Obj *tmp, Node *a) {
  if (!n)
    return;
  if (n->lhs && n->lhs->kind == ND_DEREF && n->lhs->lhs->kind == ND_VAR &&
      n->lhs->lhs->var == tmp)
    n->lhs = a;
  else
    repoint_deref(n->lhs, tmp, a);
  if (n->rhs && n->rhs->kind == ND_DEREF && n->rhs->lhs->kind == ND_VAR &&
      n->rhs->lhs->var == tmp)
    n->rhs = a;
  else
    repoint_deref(n->rhs, tmp, a);
}

// chibicc lowers `A op= B` (and `A++`/`A--`) to `tmp = &A, *tmp = *tmp op B`, taking the
// address of A so it is evaluated once. That `&A` would block promotion of A. When A is a
// plain variable its address has no side effects, so we undo the desugaring back to the
// direct `A = A op B` (no address taken) — letting loop counters and accumulators promote.
// Other lvalues (`a[i] += …`, `s.f += …`, `*p += …`) keep chibicc's form unchanged.
static Node *undo_compound(Node *n) {
  if (n->kind != ND_COMMA)
    return n;
  Node *e1 = n->lhs, *e2 = n->rhs;
  if (e1->kind != ND_ASSIGN || e1->lhs->kind != ND_VAR || !is_synthetic(e1->lhs->var))
    return n;
  // chibicc assigns `tmp = (T*)&A`, so peel the pointer cast off the `&A`.
  Node *addr = e1->rhs;
  while (addr->kind == ND_CAST)
    addr = addr->lhs;
  if (addr->kind != ND_ADDR)
    return n;
  Node *a = addr->lhs;   // the lvalue whose address was taken
  if (a->kind != ND_VAR) // only plain variables have a side-effect-free address
    return n;
  Obj *tmp = e1->lhs->var;
  if (e2->kind != ND_ASSIGN || e2->lhs->kind != ND_DEREF ||
      e2->lhs->lhs->kind != ND_VAR || e2->lhs->lhs->var != tmp)
    return n;
  // Rewrite `*tmp = *tmp op B` into `A = A op B`, reusing the existing nodes.
  e2->lhs = a;                    // assignment target: A
  repoint_deref(e2->rhs, tmp, a); // the `*tmp` operand(s) inside the op: A
  e2->next = n->next;             // preserve list position
  return e2;
}

// Run `undo_compound` over the whole tree (children first, so nested compounds collapse
// before their parents), rewriting each child slot in place.
static void rewrite(Node **pp) {
  Node *n = *pp;
  if (!n)
    return;
  rewrite(&n->lhs);
  rewrite(&n->rhs);
  rewrite(&n->cond);
  rewrite(&n->then);
  rewrite(&n->els);
  rewrite(&n->init);
  rewrite(&n->inc);
  for (Node **b = &n->body; *b; b = &(*b)->next)
    rewrite(b);
  for (Node **a = &n->args; *a; a = &(*a)->next)
    rewrite(a);
  *pp = undo_compound(n);
}

// Classify and lay out a function's locals (DESIGN §3d). First un-desugar compound
// assignment and find address-taken locals; then give each promotable scalar local an SSA
// slot (recorded as a negative `offset` sentinel) and each remaining local a frame-relative
// memory offset, with the spill scratch region reserved at the top of the frame.
static void prepare_func(Obj *fn) {
  if (!fn->is_definition)
    return;
  rewrite(&fn->body);
  n_ataken = 0;
  scan(fn->body);

  int slot = 0, off = 0;
  for (Obj *v = fn->locals; v; v = v->next) {
    bool promote = promotable_ty(v->ty) && !is_ataken(v) && !is_synthetic(v) &&
                   v != fn->va_area && v != fn->alloca_bottom && slot < MAXPROMO;
    if (promote) {
      v->offset = -(slot + 1); // sentinel: a promoted local has no frame slot
      slot++;
    } else {
      off = align_to(off, v->align);
      v->offset = off;
      off += v->ty->size;
    }
  }
  // Reserve the spill scratch region at the top of the frame (see SCRATCH_BYTES).
  off = align_to(off, 16) + SCRATCH_BYTES;
  fn->stack_size = align_to(off, 16);
}

static void gen_func(Obj *fn) {
  if (!fn->is_definition)
    return;

  nb = 0;
  nlabel = 0;
  cur_frame = fn->stack_size;
  cur_scratch = fn->stack_size - SCRATCH_BYTES; // scratch sits at the top of the frame
  spill_top = 0;
  Type *ret = fn->ty->return_ty;
  bool variadic = fn->ty->is_variadic;

  // Rebuild the promoted-slot tables from the offset sentinels set by prepare_func.
  npromo = 0;
  for (Obj *v = fn->locals; v; v = v->next)
    if (is_promoted(v)) {
      int s = slot_of(v);
      promo_ty[s] = irty(v->ty);
      if (s + 1 > npromo)
        npromo = s + 1;
    }

  // Signature: `func (i64 sp [, i64 sret], <param tys> [, i64 va_ptr]) -> (<ret ty>)`. v0
  // is the data-SP; a struct/union-returning function takes a hidden sret pointer right
  // after it and returns `()` (§3d D39); a variadic function takes a trailing pointer to
  // the marshalled args (§3d). A by-value aggregate parameter is passed by pointer (i64).
  fprintf(o, "func (i64");
  if (is_agg(ret))
    fprintf(o, ", i64"); // the hidden sret pointer
  for (Obj *p = fn->params; p; p = p->next)
    fprintf(o, ", %s", pass_irty(p->ty));
  if (variadic)
    fprintf(o, ", i64");
  if (ret->kind == TY_VOID || is_agg(ret))
    fprintf(o, ") -> () {\n");
  else
    fprintf(o, ") -> (%s) {\n", irty(ret));

  // Entry block: params are `sp` (v0), [the sret pointer], the C params, then the va ptr.
  fprintf(o, "block%d(" SP ": i64", nb++);
  int np = 1;
  sret_param = -1;
  if (is_agg(ret)) {
    sret_param = np;
    fprintf(o, ", v%d: i64", np++);
  }
  for (Obj *p = fn->params; p; p = p->next)
    fprintf(o, ", v%d: %s", np++, pass_irty(p->ty));
  int va_param = np;
  if (variadic)
    fprintf(o, ", v%d: i64", np++);
  fprintf(o, "):\n");
  nv = np;
  term = false;
  // Each C parameter: a promoted param's current value *is* its incoming SSA value (no
  // store); a memory param is spilled to its frame slot (an aggregate param is a pointer
  // to the caller's value, so the callee copies it into its own frame — by-value, §3d).
  bool param_slot[MAXPROMO] = {false};
  int pi = (sret_param < 0) ? 1 : 2; // incoming param values follow sp (and sret, if any)
  for (Obj *p = fn->params; p; p = p->next) {
    if (is_promoted(p)) {
      int s = slot_of(p);
      curval[s] = pi;
      param_slot[s] = true;
    } else {
      int off = nv++;
      fprintf(o, "  v%d = i64.const %d\n", off, p->offset);
      int addr = nv++;
      fprintf(o, "  v%d = i64.add " SP " v%d\n", addr, off);
      if (is_agg(p->ty))
        gen_memcpy(addr, pi, p->ty->size); // copy the caller's aggregate into our frame
      else
        gen_store(p->ty, addr, pi);
    }
    pi++;
  }
  // Stash the va pointer into __va_area__'s slot so va_start can load it.
  if (variadic) {
    int off = nv++;
    fprintf(o, "  v%d = i64.const %d\n", off, fn->va_area->offset);
    int addr = nv++;
    fprintf(o, "  v%d = i64.add " SP " v%d\n", addr, off);
    fprintf(o, "  i64.store v%d v%d\n", addr, va_param);
  }
  // A promoted non-parameter local starts defined (zero) so it is a valid SSA value on
  // every path before its first assignment (and this subsumes its ND_MEMZERO).
  for (Obj *v = fn->locals; v; v = v->next)
    if (is_promoted(v) && !param_slot[slot_of(v)]) {
      int s = slot_of(v);
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n", z, promo_ty[s]);
      curval[s] = z;
    }

  gen_stmt(fn->body);
  // Falling off the end: C `main` returns 0; for other paths it is UB, and returning a
  // zero is a safe, defined value. Every block needs a terminator (§3b).
  if (!term) {
    if (ret->kind == TY_VOID || is_agg(ret)) {
      fprintf(o, "  return\n"); // void, or a struct-returning func that wrote via sret
    } else {
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n  return v%d\n", z, irty(ret), z);
    }
  }
  fprintf(o, "}\n\n");
}

// Window page size, matching the runtime (`svm-interp`/`svm-jit` use 4 KiB). Read-only data is
// laid out on its own page(s) so a `data ro` segment can be protected without touching writable
// data (protection is page-granular).
#define DATA_PAGE 4096

// A read-only data global (§3a / D40): a string literal — an anonymous (`.L..`) char array with
// initializer bytes (this includes `__func__`/`__FUNCTION__`). chibicc tracks no `const`, and
// these are the non-modifiable data; writing to one is UB, so mapping it read-only turns that
// into a clean detect-and-kill fault.
static bool is_rodata(Obj *g) {
  return !g->is_function && g->init_data && g->name && g->name[0] == '.' && g->name[1] == 'L' &&
         g->ty->kind == TY_ARRAY && g->ty->base && g->ty->base->kind == TY_CHAR;
}

// Lay globals out at fixed window offsets; set `data_end` (the data-stack base). Writable data
// (and the [0,16) handle slots) goes first from 16, then a page boundary, then read-only string
// literals on their own page(s), then another page boundary before the data stack — so the
// `data ro` segments are page-isolated for protection (§3a / D40). Returns true if any global.
static bool layout_globals(Obj *prog) {
  int off = 16;
  bool any = false;
  // Pass 1: writable globals (and BSS) packed from 16.
  for (Obj *g = prog; g; g = g->next) {
    if (g->is_function || is_rodata(g))
      continue;
    off = align_to(off, g->align);
    g->offset = off;
    off += g->ty->size;
    any = true;
  }
  // Pass 2: read-only string literals, on a fresh page so they share no page with writable data.
  bool any_ro = false;
  for (Obj *g = prog; g; g = g->next) {
    if (!is_rodata(g))
      continue;
    if (!any_ro) {
      off = align_to(off, DATA_PAGE);
      any_ro = true;
    }
    off = align_to(off, g->align);
    g->offset = off;
    off += g->ty->size;
    any = true;
  }
  // End the RO region on a page boundary too, so the data stack never shares its page.
  if (any_ro)
    off = align_to(off, DATA_PAGE);
  data_end = align_to(off, 16);
  return any;
}

// Resolve a relocation's target symbol to the value stored in the data image: a data
// global's window offset, or a function's funcref index (§3c — a function pointer in
// memory is its function-table index). Every global offset (layout_globals) and function
// index (funcs[]) is assigned before data is emitted, so the value is a compile-time
// constant — there is no runtime relocation step.
static long symbol_value(Obj *prog, char *name) {
  for (Obj *s = prog; s; s = s->next)
    if (s->name && !strcmp(s->name, name))
      return s->is_function ? func_index(s) : s->offset;
  return 0; // unreachable for a well-formed whole-program module (defensive NULL)
}

// Emit a module-level `data` segment (§3a) for each initialized global: the runtime copies
// the bytes into the window at instantiation, replacing the old per-byte `_start` init stores.
// Pointer initializers (`char *p = "..."`, `&global`, `&arr[k]`, function pointers, and
// arrays/structs of them) become **relocations** (§3a): each writes the 8-byte little-endian
// window address of its target symbol + addend into the image, computed here since all
// offsets/indices are known.
static void emit_data_segments(Obj *prog) {
  for (Obj *g = prog; g; g = g->next) {
    if (g->is_function || !g->init_data)
      continue;
    int size = g->ty->size;
    unsigned char *buf = calloc(size ? size : 1, 1);
    memcpy(buf, g->init_data, size);
    for (Relocation *r = g->rel; r; r = r->next) {
      unsigned long val = (unsigned long)(symbol_value(prog, *r->label) + r->addend);
      for (int i = 0; i < 8 && r->offset + i < size; i++)
        buf[r->offset + i] = (unsigned char)(val >> (8 * i)); // little-endian (§3b)
    }
    fprintf(o, "data %s%d \"", is_rodata(g) ? "ro " : "", g->offset);
    for (int i = 0; i < size; i++) {
      unsigned char c = buf[i];
      if (c == '\\')
        fprintf(o, "\\\\");
      else if (c == '"')
        fprintf(o, "\\\"");
      else if (c >= 0x20 && c <= 0x7e)
        fputc(c, o);
      else
        fprintf(o, "\\x%02x", c);
    }
    fprintf(o, "\"\n");
    free(buf);
  }
}

// Synthetic entry (function 0): stash the powerbox capability handles, then call `main` with
// the initial data-SP (= data_end). Global data is now placed by module-level `data` segments
// (§3a, see `emit_data_segments`), not written here. The runtime invokes this with the granted
// handles `(stdout, stdin, exit)` as i32 arguments.
static void emit_start(Obj *main_fn) {
  npromo = 0; // _start is hand-written and threads no promoted locals
  Type *mret = main_fn->ty->return_ty;
  bool is_void = mret->kind == TY_VOID;
  fprintf(o, "func (i32, i32, i32) -> (%s) {\n", is_void ? "" : irty(mret));
  fprintf(o, "block0(v0: i32, v1: i32, v2: i32):\n"); // stdout, stdin, exit
  nv = 3;
  // Stash each handle in its reserved slot so the stdio builtins can load it.
  int slots[3] = {STDOUT_SLOT, STDIN_SLOT, EXIT_SLOT};
  for (int i = 0; i < 3; i++) {
    int a = nv++;
    fprintf(o, "  v%d = i64.const %d\n", a, slots[i]);
    fprintf(o, "  i32.store v%d v%d\n", a, i);
  }
  int sp = nv++;
  fprintf(o, "  v%d = i64.const %d\n", sp, data_end);
  // `int main()` (empty parens) is variadic in chibicc, so it expects the hidden va
  // pointer; main never reads it, so any in-window pointer (the sp) does.
  char va[24] = "";
  if (main_fn->ty->is_variadic)
    snprintf(va, sizeof va, ", v%d", sp);
  if (is_void) {
    fprintf(o, "  call %d (v%d%s)\n  return\n", func_index(main_fn), sp, va);
  } else {
    int r = nv++;
    fprintf(o, "  v%d = call %d (v%d%s)\n  return v%d\n", r, func_index(main_fn), sp, va, r);
  }
  fprintf(o, "}\n\n");
}

void codegen_ir(Obj *prog, FILE *out) {
  o = out;

  // Order the function definitions with `main` first. A `_start` wrapper (function 0)
  // then sets up the data-SP and calls `main`, so real functions begin at index 1.
  nfuncs = 0;
  for (Obj *fn = prog; fn; fn = fn->next)
    if (fn->is_function && fn->is_definition && fn->name && !strcmp(fn->name, "main"))
      funcs[nfuncs++] = fn;
  for (Obj *fn = prog; fn; fn = fn->next)
    if (fn->is_function && fn->is_definition && !(fn->name && !strcmp(fn->name, "main")))
      funcs[nfuncs++] = fn;

  bool has_main = nfuncs > 0 && funcs[0]->name && !strcmp(funcs[0]->name, "main");
  start_off = has_main ? 1 : 0;

  // `_start` stashes the capability handles in the window, so a module with an entry
  // always needs one.
  bool need_mem = layout_globals(prog) || has_main;
  for (int i = 0; i < nfuncs; i++) {
    prepare_func(funcs[i]);
    if (funcs[i]->locals)
      need_mem = true;
  }

  // A 2^16-byte window is ample for the globals + data stack of the programs we lower
  // today (the size becomes program-driven once a real data segment / heap land).
  if (need_mem)
    fprintf(o, "memory 16\n\n");

  // Global initializers become module-level `data` segments (§3a), placed by the runtime.
  emit_data_segments(prog);

  if (has_main)
    emit_start(funcs[0]);
  for (int i = 0; i < nfuncs; i++)
    gen_func(funcs[i]);
}
