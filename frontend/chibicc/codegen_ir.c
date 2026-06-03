// codegen_ir.c — SVM text IR backend for chibicc (DESIGN.md §3d).
//
// Walks chibicc's typed AST and emits the SVM text IR our verifier/interpreter/JIT
// consume, instead of x86-64 assembly. This is the C frontend's lowering: source UB is
// resolved here into the IR's total semantics (§3b), and the *verifier* — not this code
// — is what enforces escape-freedom (§2a), so the frontend is outside the escape-TCB.
//
// Status (grown incrementally): a single paramless `int`/`long`/`void` function with
// integer expressions (constants, +/-/*'/'%, bitwise, shifts, comparisons, unary
// -/!/~, integer casts), **scalar locals** in the §3d data-stack window (load/store,
// assignment, `&`/`*`, pointers to locals), and **structured control flow** —
// `if`/`else`, `while`, `for` — lowered to a param-free block CFG (state lives in
// memory, so values never cross a block). Calls/params, `break`/`continue`/`switch`,
// short-circuit `&&`/`||`/`?:`, arrays/structs, and floats come next; anything
// unsupported is a hard error (so we never emit IR we cannot stand behind).

#include "chibicc.h"

static FILE *o;
static int nv;    // next SSA value index in the *current block* (blocks are param-free,
                  // so values never cross a block boundary — they reset per block)
static int nb;    // next block label number in the current function
static bool term; // is the current block already terminated?

// Open a new IR block with label `id`: emit its header and reset per-block state. (Block
// labels are resolved by name, so forward references to not-yet-opened blocks are fine.)
static void open_block(int id) {
  fprintf(o, "block%d():\n", id);
  nv = 0;
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
    return "i64";
  default:
    error_tok(ty->name, "codegen_ir: unsupported type");
  }
}

// True if `ty` is a 64-bit value in our model.
static bool is64(Type *ty) { return irty(ty)[1] == '6'; }

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
  default:
    error_tok(ty->name, "codegen_ir: unsupported store type");
  }
}

// The address of an lvalue, as an SSA i64.
static int gen_addr(Node *node) {
  switch (node->kind) {
  case ND_VAR:
    if (!node->var->is_local)
      error_tok(node->tok, "codegen_ir: global variables not supported yet");
    int r = nv++;
    fprintf(o, "  v%d = i64.const %d\n", r, node->var->offset);
    return r;
  case ND_DEREF:
    return gen_expr(node->lhs); // the pointer value *is* the address
  case ND_COMMA:
    gen_expr(node->lhs);
    return gen_addr(node->rhs);
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

// A comparison: the op width is the operands' type; the result is always i32 0/1.
static int cmpop(Node *node, char *base, bool sign) {
  int a = gen_expr(node->lhs);
  int b = gen_expr(node->rhs);
  int r = nv++;
  char *p = irty(node->lhs->ty);
  if (sign)
    fprintf(o, "  v%d = %s.%s_%s v%d v%d\n", r, p, base,
            node->lhs->ty->is_unsigned ? "u" : "s", a, b);
  else
    fprintf(o, "  v%d = %s.%s v%d v%d\n", r, p, base, a, b);
  return r;
}

// An integer cast: i32<->i64 extend/wrap; same-width casts are no-ops here (narrowing
// to char/short within i32 is handled when locals/loads land).
static int gen_cast(Node *node) {
  int a = gen_expr(node->lhs);
  bool from64 = is64(node->lhs->ty);
  bool to64 = is64(node->ty);
  if (from64 == to64)
    return a;
  int r = nv++;
  if (to64)
    fprintf(o, "  v%d = i64.extend_i32_%s v%d\n", r,
            node->lhs->ty->is_unsigned ? "u" : "s", a);
  else
    fprintf(o, "  v%d = i32.wrap_i64 v%d\n", r, a);
  return r;
}

static int gen_expr(Node *node) {
  switch (node->kind) {
  case ND_NUM: {
    int r = nv++;
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
    return binop(node, node->ty->is_unsigned ? "div_u" : "div_s");
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
    // -x  ==  0 - x
    int a = gen_expr(node->lhs);
    char *p = irty(node->ty);
    int z = nv++;
    fprintf(o, "  v%d = %s.const 0\n", z, p);
    int r = nv++;
    fprintf(o, "  v%d = %s.sub v%d v%d\n", r, p, z, a);
    return r;
  }
  case ND_NOT: {
    // !x  ==  (x == 0), result i32
    int a = gen_expr(node->lhs);
    int r = nv++;
    fprintf(o, "  v%d = %s.eqz v%d\n", r, irty(node->lhs->ty), a);
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
  case ND_COMMA:
    gen_expr(node->lhs);
    return gen_expr(node->rhs);
  case ND_VAR:
    return gen_load(node->ty, gen_addr(node));
  case ND_DEREF:
    return gen_load(node->ty, gen_addr(node));
  case ND_ADDR:
    return gen_addr(node->lhs);
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
    // Zero-initialize the variable's whole window region (§3d data stack).
    int sz = node->var->ty->size;
    int base = node->var->offset;
    for (int i = 0; i < sz;) {
      int a = nv++;
      fprintf(o, "  v%d = i64.const %d\n", a, base + i);
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
  fprintf(o, "  br_if v%d block%d() block%d()\n", c, t, e);
  term = true;

  open_block(t);
  gen_stmt(node->then);
  if (!term)
    fprintf(o, "  br block%d()\n", end);

  open_block(e);
  if (node->els)
    gen_stmt(node->els);
  if (!term)
    fprintf(o, "  br block%d()\n", end);

  open_block(end);
}

// `for (init; cond; inc) body` (and `while`, with init/inc absent): cond/body/end blocks
// with a back-edge. (break/continue, which chibicc lowers to gotos, are not yet handled.)
static void gen_for(Node *node) {
  if (node->init)
    gen_stmt(node->init);
  int cond = nb++, body = nb++, end = nb++;
  fprintf(o, "  br block%d()\n", cond);
  term = true;

  open_block(cond);
  if (node->cond) {
    int c = gen_expr(node->cond);
    fprintf(o, "  br_if v%d block%d() block%d()\n", c, body, end);
  } else {
    fprintf(o, "  br block%d()\n", body); // `for(;;)` — unconditional
  }
  term = true;

  open_block(body);
  gen_stmt(node->then);
  if (!term) {
    if (node->inc)
      gen_expr(node->inc);
    fprintf(o, "  br block%d()\n", cond);
  }

  open_block(end);
}

static void gen_stmt(Node *node) {
  switch (node->kind) {
  case ND_BLOCK:
    for (Node *n = node->body; n; n = n->next) {
      if (term)
        break; // unreachable after a terminator (drop dead code)
      gen_stmt(n);
    }
    return;
  case ND_EXPR_STMT:
    gen_expr(node->lhs); // value discarded
    return;
  case ND_IF:
    gen_if(node);
    return;
  case ND_FOR:
    gen_for(node);
    return;
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

// Lay this function's locals out in the data-stack frame (the window), from STACK_BASE.
// (One fixed frame per function for now — fine until calls share the window via a
// data-SP, §3d.) Sets each local's `offset` and the frame `stack_size`.
static void assign_offsets(Obj *fn) {
  int off = STACK_BASE;
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
  if (fn->params)
    error_tok(fn->tok, "codegen_ir: function parameters not supported yet");

  nb = 0;
  Type *ret = fn->ty->return_ty;
  if (ret->kind == TY_VOID)
    fprintf(o, "func () -> () {\n");
  else
    fprintf(o, "func () -> (%s) {\n", irty(ret));
  open_block(nb++); // the entry block (index 0)
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

void codegen_ir(Obj *prog, FILE *out) {
  o = out;

  // Assign data-stack offsets and decide whether the module needs a window.
  bool need_mem = false;
  for (Obj *fn = prog; fn; fn = fn->next)
    if (fn->is_function && fn->is_definition) {
      assign_offsets(fn);
      if (fn->locals)
        need_mem = true;
    }
  // A 2^16-byte window is ample for the data stack of the small programs we lower today
  // (the size becomes program-driven once calls/heap land).
  if (need_mem)
    fprintf(o, "memory 16\n\n");

  for (Obj *fn = prog; fn; fn = fn->next)
    if (fn->is_function)
      gen_func(fn);
}
