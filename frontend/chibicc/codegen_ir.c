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
// (`ND_MEMBER`); initializers lower to per-element/-member scalar stores. By-value
// aggregate *arguments/returns* (sret, §3d D39) and whole-struct assignment are not done
// yet — pointers to aggregates are fine. **Globals + string literals** live at fixed
// window offsets in a data region below the data stack; a synthetic **`_start`**
// (function 0) takes the powerbox capability handles, stashes them in a reserved region,
// writes globals' initializer bytes into the window, then calls `main` with the initial
// data-SP. (A real read-only data segment per §3a later replaces the byte stores; globals
// holding pointers/relocations are not handled yet.) **Stdio over the powerbox** (§3e):
// `write`/`read`/`exit` are recognized builtins lowered to `cap.call` on the stashed
// Stream/Exit handles — so real C reaches stdout/stdin and terminates with an exit code.
// **Floats** (`float`=f32, `double`/`long double`=f64) work too: arithmetic, compares,
// `-`/`!`, literals, and all int<->float / float<->float conversions (float->int is
// saturating, §3b). **`break`/`continue`** (a loop-context stack) and **`switch`** (a
// dispatch chain threading the value through `(sp, val)` compare blocks, with
// fall-through and `case` ranges) work. Indirect calls, by-value aggregate args/returns,
// general `goto`, varargs/`printf`, and `malloc` remain; anything unsupported is a hard
// error (so we never emit IR we cannot stand behind). The everything-in-memory model
// (no SSA promotion yet) is the main perf gap.

#include "chibicc.h"

static FILE *o;
static int nv;    // next SSA value index in the *current block* (blocks are param-free,
                  // so values never cross a block boundary — they reset per block)
static int nb;        // next block label number in the current function
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

// Open a new IR block with label `id`: emit its header (taking the data-SP as `v0`) and
// reset per-block state. Block labels resolve by name, so forward references are fine.
static void open_block(int id) {
  fprintf(o, "block%d(" SP ": i64):\n", id);
  nv = 1; // v0 is the data-SP parameter
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

// True if a value of type `ty` is held *by address* (arrays/aggregates): reading the
// lvalue yields its address, not a loaded scalar.
static bool by_address(Type *ty) {
  return ty->kind == TY_ARRAY || ty->kind == TY_STRUCT || ty->kind == TY_UNION;
}

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

// The address of an lvalue, as an SSA i64.
static int gen_addr(Node *node) {
  switch (node->kind) {
  case ND_VAR: {
    int r = nv++;
    if (node->var->is_local) {
      // The local lives at sp + frame-offset (§3d data stack).
      int off = nv++;
      fprintf(o, "  v%d = i64.const %d\n", off, node->var->offset);
      fprintf(o, "  v%d = i64.add " SP " v%d\n", r, off);
    } else {
      // A global lives at a fixed window offset in the data region below the stack.
      fprintf(o, "  v%d = i64.const %d\n", r, node->var->offset);
    }
    return r;
  }
  case ND_DEREF:
    return gen_expr(node->lhs); // the pointer value *is* the address
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
  int a = gen_expr(node->lhs);
  int b = gen_expr(node->rhs);
  int r = nv++;
  fprintf(o, "  v%d = %s.%s v%d v%d\n", r, irty(node->lhs->ty), op, a, b);
  return r;
}

// A comparison: the op width is the operands' type; the result is always i32 0/1. Integer
// `lt`/`le` take a signedness suffix; float compares (and `eq`/`ne`) do not.
static int cmpop(Node *node, char *base, bool sign) {
  int a = gen_expr(node->lhs);
  int b = gen_expr(node->rhs);
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

// Open a merge block taking `(sp, v1: ty)`: the carried result is v1, nv resumes at 2.
static void open_merge(int id, char *ty) {
  fprintf(o, "block%d(" SP ": i64, v1: %s):\n", id, ty);
  nv = 2;
  term = false;
}

// `a && b` and `a || b` → i32 0/1, short-circuit; the result is carried into the merge.
static int gen_logand(Node *node) {
  int ta = gen_truth(node->lhs);
  int rhs = nb++, fls = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", ta, rhs, fls);
  open_block(rhs);
  int tb = gen_truth(node->rhs);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, tb);
  open_block(fls);
  int z = nv++;
  fprintf(o, "  v%d = i32.const 0\n", z);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, z);
  open_merge(merge, "i32");
  return 1;
}

static int gen_logor(Node *node) {
  int ta = gen_truth(node->lhs);
  int tru = nb++, rhs = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", ta, tru, rhs);
  open_block(tru);
  int one = nv++;
  fprintf(o, "  v%d = i32.const 1\n", one);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, one);
  open_block(rhs);
  int tb = gen_truth(node->rhs);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, tb);
  open_merge(merge, "i32");
  return 1;
}

// `cond ? then : els` → branches converted to the result type, carried into the merge.
static int gen_cond(Node *node) {
  int c = gen_truth(node->cond);
  int th = nb++, el = nb++, merge = nb++;
  fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", c, th, el);

  if (node->ty->kind == TY_VOID) {
    // A void `?:` — both arms are evaluated for effect only, no carried value.
    open_block(th);
    gen_expr(node->then);
    if (!term)
      fprintf(o, "  br block%d(" SP ")\n", merge);
    open_block(el);
    gen_expr(node->els);
    if (!term)
      fprintf(o, "  br block%d(" SP ")\n", merge);
    open_block(merge);
    return 0;
  }

  open_block(th);
  int vt = gen_expr_as(node->then, node->ty);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, vt);
  open_block(el);
  int ve = gen_expr_as(node->els, node->ty);
  fprintf(o, "  br block%d(" SP ", v%d)\n", merge, ve);
  open_merge(merge, irty(node->ty));
  return 1;
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
  int buf = gen_expr(a->next);
  int len = widen_i64(gen_expr(a->next->next), a->next->next->ty);
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
    return gen_load(node->ty, gen_addr(node));
  case ND_DEREF:
    return gen_load(node->ty, gen_addr(node));
  case ND_MEMBER:
    return gen_load(node->ty, gen_addr(node));
  case ND_ADDR:
    return gen_addr(node->lhs);
  case ND_FUNCALL: {
    if (node->lhs->kind != ND_VAR || !node->lhs->var->is_function)
      error_tok(node->tok, "codegen_ir: only direct calls are supported yet");
    // Intercept the stdio builtins (powerbox §3e) before treating it as a guest call.
    char *fname = node->lhs->var->name;
    if (fname) {
      if (!strcmp(fname, "write"))
        return gen_builtin_stream(node, STDOUT_SLOT, 1);
      if (!strcmp(fname, "read"))
        return gen_builtin_stream(node, STDIN_SLOT, 0);
      if (!strcmp(fname, "exit") || !strcmp(fname, "_exit"))
        return gen_builtin_exit(node);
    }
    // Evaluate the arguments (already cast to the parameter types by the parser).
    int argv[64];
    int n = 0;
    for (Node *a = node->args; a; a = a->next) {
      if (n == 64)
        error_tok(node->tok, "codegen_ir: too many call arguments");
      argv[n++] = gen_expr(a);
    }
    // The callee gets a fresh frame above ours: callee_sp = sp + our frame size.
    int fs = nv++;
    fprintf(o, "  v%d = i64.const %d\n", fs, cur_frame);
    int csp = nv++;
    fprintf(o, "  v%d = i64.add " SP " v%d\n", csp, fs);

    int idx = func_index(node->lhs->var);
    bool is_void = node->ty->kind == TY_VOID;
    int r = is_void ? 0 : nv++;
    if (is_void)
      fprintf(o, "  call %d (v%d", idx, csp);
    else
      fprintf(o, "  v%d = call %d (v%d", r, idx, csp);
    for (int i = 0; i < n; i++)
      fprintf(o, ", v%d", argv[i]);
    fprintf(o, ")\n");
    return r; // for a void call the value is discarded
  }
  case ND_ASSIGN: {
    int addr = gen_addr(node->lhs);
    int val = gen_expr(node->rhs);
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
  int c = gen_expr(node->cond);
  int t = nb++, e = nb++, end = nb++;
  fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", c, t, e);
  term = true;

  open_block(t);
  gen_stmt(node->then);
  if (!term)
    fprintf(o, "  br block%d(" SP ")\n", end);

  open_block(e);
  if (node->els)
    gen_stmt(node->els);
  if (!term)
    fprintf(o, "  br block%d(" SP ")\n", end);

  open_block(end);
}

// `for (init; cond; inc) body` (and `while`, with init/inc absent): cond/body/cont/end
// blocks with a back-edge. `continue` targets the `cont` block (which runs `inc` then
// re-tests `cond`); `break` targets `end`.
static void gen_for(Node *node) {
  if (node->init)
    gen_stmt(node->init);
  int cond = nb++, body = nb++, cont = nb++, end = nb++;
  fprintf(o, "  br block%d(" SP ")\n", cond);
  term = true;

  open_block(cond);
  if (node->cond) {
    int c = gen_expr(node->cond);
    fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", c, body, end);
  } else {
    fprintf(o, "  br block%d(" SP ")\n", body); // `for(;;)` — unconditional
  }
  term = true;

  open_block(body);
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, node->cont_label, cont};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP ")\n", cont);

  open_block(cont);
  if (node->inc)
    gen_expr(node->inc);
  fprintf(o, "  br block%d(" SP ")\n", cond);

  open_block(end);
}

// `do body while (cond)`: body runs once, then `cont` re-tests. `break` → end.
static void gen_do(Node *node) {
  int body = nb++, cont = nb++, end = nb++;
  fprintf(o, "  br block%d(" SP ")\n", body);
  term = true;

  open_block(body);
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, node->cont_label, cont};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP ")\n", cont);

  open_block(cont);
  int c = gen_expr(node->cond);
  fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ")\n", c, body, end);

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

  // Dispatch: each compare block tests one case and forwards (sp, val) to the next.
  int check = nb++;
  fprintf(o, "  br block%d(" SP ", v%d)\n", check, v);
  term = true;
  for (Node *c = node->case_next; c; c = c->case_next) {
    fprintf(o, "block%d(" SP ": i64, v1: %s):\n", check, p);
    nv = 2;
    int next = nb++;
    int hit = nv++;
    if (c->begin == c->end) {
      int k = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", k, p, c->begin);
      fprintf(o, "  v%d = %s.eq v1 v%d\n", hit, p, k);
    } else {
      // [GNU] case range begin..end: (val - begin) <=u (end - begin)
      int kb = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", kb, p, c->begin);
      int d = nv++;
      fprintf(o, "  v%d = %s.sub v1 v%d\n", d, p, kb);
      int kr = nv++;
      fprintf(o, "  v%d = %s.const %ld\n", kr, p, c->end - c->begin);
      fprintf(o, "  v%d = %s.le_u v%d v%d\n", hit, p, d, kr);
    }
    fprintf(o, "  br_if v%d block%d(" SP ") block%d(" SP ", v1)\n", hit,
            case_block_of(c), next);
    check = next;
  }
  // No case matched → default (or break past the switch).
  fprintf(o, "block%d(" SP ": i64, v1: %s):\n  br block%d(" SP ")\n", check, p, defblk);
  term = true;

  // The body: ND_CASE labels open their blocks; `break` (cont_label NULL so `continue`
  // passes through to an enclosing loop) targets `end`.
  loopstk[loopsp++] = (struct LoopCtx){node->brk_label, end, NULL, -1};
  gen_stmt(node->then);
  loopsp--;
  if (!term)
    fprintf(o, "  br block%d(" SP ")\n", end);
  open_block(end);
}

static void gen_stmt(Node *node) {
  switch (node->kind) {
  case ND_BLOCK:
    for (Node *n = node->body; n; n = n->next) {
      // Drop dead code after a terminator — but a `case`/`default` label reopens a
      // reachable block, so it is always emitted.
      if (term && n->kind != ND_CASE)
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
      fprintf(o, "  br block%d(" SP ")\n", blk);
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
  case ND_GOTO: {
    // break/continue: branch to the matching enclosing loop's break/continue block.
    for (int i = loopsp - 1; i >= 0; i--) {
      if (node->unique_label && loopstk[i].brk_label &&
          !strcmp(node->unique_label, loopstk[i].brk_label)) {
        fprintf(o, "  br block%d(" SP ")\n", loopstk[i].brk_blk);
        term = true;
        return;
      }
      if (node->unique_label && loopstk[i].cont_label &&
          !strcmp(node->unique_label, loopstk[i].cont_label)) {
        fprintf(o, "  br block%d(" SP ")\n", loopstk[i].cont_blk);
        term = true;
        return;
      }
    }
    error_tok(node->tok, "codegen_ir: general goto/labels not supported yet");
  }
  case ND_RETURN:
    if (node->lhs) {
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

// Lay this function's locals out as *frame-relative* offsets (from 0); each local lives
// at `sp + offset` at run time (§3d). Sets each local's `offset` and `stack_size`.
static void assign_offsets(Obj *fn) {
  int off = 0;
  for (Obj *v = fn->locals; v; v = v->next) {
    off = align_to(off, v->align);
    v->offset = off;
    off += v->ty->size;
  }
  fn->stack_size = align_to(off, 16);
}

static void gen_func(Obj *fn) {
  if (!fn->is_definition)
    return;

  nb = 0;
  cur_frame = fn->stack_size;
  Type *ret = fn->ty->return_ty;

  // Signature: `func (i64 sp, <param tys>) -> (<ret ty>)` — the data-SP is param 0.
  fprintf(o, "func (i64");
  for (Obj *p = fn->params; p; p = p->next)
    fprintf(o, ", %s", irty(p->ty));
  if (ret->kind == TY_VOID)
    fprintf(o, ") -> () {\n");
  else
    fprintf(o, ") -> (%s) {\n", irty(ret));

  // Entry block: params are `sp` (v0) then the C params (v1..vN). Spill each C param to
  // its data-stack slot (sp + offset) so the body reads/writes it like any other local.
  fprintf(o, "block%d(" SP ": i64", nb++);
  int np = 1;
  for (Obj *p = fn->params; p; p = p->next)
    fprintf(o, ", v%d: %s", np++, irty(p->ty));
  fprintf(o, "):\n");
  nv = np;
  term = false;
  int pi = 1;
  for (Obj *p = fn->params; p; p = p->next) {
    int off = nv++;
    fprintf(o, "  v%d = i64.const %d\n", off, p->offset);
    int addr = nv++;
    fprintf(o, "  v%d = i64.add " SP " v%d\n", addr, off);
    gen_store(p->ty, addr, pi++);
  }

  gen_stmt(fn->body);
  // Falling off the end: C `main` returns 0; for other paths it is UB, and returning a
  // zero is a safe, defined value. Every block needs a terminator (§3b).
  if (!term) {
    if (ret->kind == TY_VOID) {
      fprintf(o, "  return\n");
    } else {
      int z = nv++;
      fprintf(o, "  v%d = %s.const 0\n  return v%d\n", z, irty(ret), z);
    }
  }
  fprintf(o, "}\n\n");
}

// Lay globals + string literals out at fixed window offsets in the data region from 16;
// set `data_end` (the data-stack base). Returns true if any global exists.
static bool layout_globals(Obj *prog) {
  int off = 16;
  bool any = false;
  for (Obj *g = prog; g; g = g->next) {
    if (g->is_function)
      continue;
    off = align_to(off, g->align);
    g->offset = off;
    off += g->ty->size;
    any = true;
  }
  data_end = align_to(off, 16);
  return any;
}

// Emit stores that write a global's initializer bytes into its window slot. Per-byte for
// simplicity (these run once, in `_start`); a future real data segment (§3a) replaces it.
static void emit_init_data(int base, char *data, int sz) {
  for (int i = 0; i < sz; i++) {
    int a = nv++;
    fprintf(o, "  v%d = i64.const %d\n", a, base + i);
    int z = nv++;
    fprintf(o, "  v%d = i32.const %d\n", z, (unsigned char)data[i]);
    fprintf(o, "  i32.store8 v%d v%d\n", a, z);
  }
}

// Synthetic entry (function 0): stash the powerbox capability handles, initialize global
// data, then call `main` with the initial data-SP (= data_end). The runtime invokes this
// with the granted handles `(stdout, stdin, exit)` as i32 arguments.
static void emit_start(Obj *prog, Obj *main_fn) {
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
  for (Obj *g = prog; g; g = g->next) {
    if (g->is_function || !g->init_data)
      continue;
    if (g->rel)
      error_tok(g->tok, "codegen_ir: global initialized with a pointer (relocation) "
                        "not supported yet");
    emit_init_data(g->offset, g->init_data, g->ty->size);
  }
  int sp = nv++;
  fprintf(o, "  v%d = i64.const %d\n", sp, data_end);
  if (is_void) {
    fprintf(o, "  call %d (v%d)\n  return\n", func_index(main_fn), sp);
  } else {
    int r = nv++;
    fprintf(o, "  v%d = call %d (v%d)\n  return v%d\n", r, func_index(main_fn), sp, r);
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
    assign_offsets(funcs[i]);
    if (funcs[i]->locals)
      need_mem = true;
  }

  // A 2^16-byte window is ample for the globals + data stack of the programs we lower
  // today (the size becomes program-driven once a real data segment / heap land).
  if (need_mem)
    fprintf(o, "memory 16\n\n");

  if (has_main)
    emit_start(prog, funcs[0]);
  for (int i = 0; i < nfuncs; i++)
    gen_func(funcs[i]);
}
