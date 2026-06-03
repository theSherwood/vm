// codegen_ir.c — SVM text IR backend for chibicc (DESIGN.md §3d).
//
// Walks chibicc's typed AST and emits the SVM text IR our verifier/interpreter/JIT
// consume, instead of x86-64 assembly. This is the C frontend's lowering: source UB is
// resolved here into the IR's total semantics (§3b), and the *verifier* — not this code
// — is what enforces escape-freedom (§2a), so the frontend is outside the escape-TCB.
//
// Status (grown incrementally): a single `int`/`long` function whose body is integer
// expressions + `return` — constants, +/-/*/'/'%, bitwise, shifts, comparisons, unary
// minus/not/bitnot, and integer casts. Locals, control flow, calls, pointers/memory,
// and floats are added in later passes; anything unsupported is a hard error (so we
// never emit IR we cannot stand behind).

#include "chibicc.h"

static FILE *o;
static int nv; // next SSA value index in the current function

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
  default:
    error_tok(node->tok, "codegen_ir: unsupported expression");
  }
}

static void gen_stmt(Node *node) {
  switch (node->kind) {
  case ND_BLOCK:
    for (Node *n = node->body; n; n = n->next)
      gen_stmt(n);
    return;
  case ND_EXPR_STMT:
    gen_expr(node->lhs); // value discarded
    return;
  case ND_RETURN:
    if (node->lhs) {
      int v = gen_expr(node->lhs);
      fprintf(o, "  return v%d\n", v);
    } else {
      fprintf(o, "  return\n");
    }
    return;
  default:
    error_tok(node->tok, "codegen_ir: unsupported statement");
  }
}

static void gen_func(Obj *fn) {
  if (!fn->is_definition)
    return;
  if (fn->params)
    error_tok(fn->tok, "codegen_ir: function parameters not supported yet");

  nv = 0;
  Type *ret = fn->ty->return_ty;
  if (ret->kind == TY_VOID)
    fprintf(o, "func () -> () {\n");
  else
    fprintf(o, "func () -> (%s) {\n", irty(ret));
  fprintf(o, "block0():\n");
  gen_stmt(fn->body);
  fprintf(o, "}\n\n");
}

void codegen_ir(Obj *prog, FILE *out) {
  o = out;
  for (Obj *fn = prog; fn; fn = fn->next)
    if (fn->is_function)
      gen_func(fn);
}
