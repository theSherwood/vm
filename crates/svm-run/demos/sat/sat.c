/* sat.c — a small DPLL SAT solver, as a self-validating pure-compute correctness indicator for the
 * LLVM on-ramp (LLVM.md "Suggested ladder": "a SAT solver / perft slot in here too — pure compute,
 * self-validating"). No OS surface — the whole thing runs on the bare powerbox, output via `printf`.
 *
 * Why it is a good translator exercise: backtracking search with unit propagation is **branchy,
 * pointer-and-array heavy, and deeply recursive** — a different shape from the corpus's hashers and
 * parsers. Why it is self-validating: (1) a **planted** random 3-SAT instance is satisfiable by
 * construction, and the solver's model is re-checked against every clause (`verify`), so a wrong
 * model or a mis-propagation is caught in-guest; (2) **pigeonhole** PHP(n+1, n) is a classic UNSAT
 * family, so "UNSAT" is a known answer. The decision heuristic is fixed (lowest unassigned var,
 * true-first), so the model is deterministic and the on-ramp output is byte-identical to native. */

#include <stdio.h>

#define MAXV 64      /* max variables (1-indexed) */
#define MAXLIT 8192  /* max total literals across all clauses (flat, 0-terminated per clause) */

/* A CNF: literals packed into `lits` with a trailing 0 after each clause; `nclauses`/`nvars` count. */
typedef struct {
  int lits[MAXLIT];
  int nlit;
  int nclauses;
  int nvars;
} Cnf;

static void cnf_init(Cnf *f, int nvars) {
  f->nlit = 0;
  f->nclauses = 0;
  f->nvars = nvars;
}
/* add a clause from a 0-terminated literal list */
static void cnf_add(Cnf *f, const int *ls) {
  while (*ls) f->lits[f->nlit++] = *ls++;
  f->lits[f->nlit++] = 0;
  f->nclauses++;
}

static int g_val[MAXV + 1];       /* -1 unassigned, else 0/1 */
static int g_trail[MAXV + 1];     /* vars assigned, for undo */
static int g_tn;

static int lit_var(int l) { return l < 0 ? -l : l; }
static int lit_want(int l) { return l > 0 ? 1 : 0; }

/* Unit propagation: assign forced literals until fixpoint. Returns 0 on conflict, 1 otherwise. */
static int propagate(const Cnf *f) {
  int again = 1;
  while (again) {
    again = 0;
    const int *p = f->lits;
    for (int ci = 0; ci < f->nclauses; ci++) {
      int nun = 0, unit = 0, sat = 0;
      const int *c = p;
      for (; *c; c++) {
        int v = lit_var(*c);
        if (g_val[v] == -1) { nun++; unit = *c; }
        else if (g_val[v] == lit_want(*c)) { sat = 1; }
      }
      p = c + 1; /* next clause starts past the 0 */
      if (sat) continue;
      if (nun == 0) return 0;            /* all literals false → conflict */
      if (nun == 1) {                    /* forced assignment */
        int v = lit_var(unit);
        g_val[v] = lit_want(unit);
        g_trail[g_tn++] = v;
        again = 1;
      }
    }
  }
  return 1;
}

/* DPLL: returns 1 with `g_val` a model, or 0 (UNSAT under the current partial assignment). */
static int solve(const Cnf *f) {
  int mark = g_tn;
  if (!propagate(f)) {
    while (g_tn > mark) g_val[g_trail[--g_tn]] = -1;
    return 0;
  }
  int v = 0;
  for (int i = 1; i <= f->nvars; i++) {
    if (g_val[i] == -1) { v = i; break; }
  }
  if (!v) return 1; /* every variable assigned, no conflict → SAT */
  for (int b = 1; b >= 0; b--) {
    int m2 = g_tn;
    g_val[v] = b;
    g_trail[g_tn++] = v;
    if (solve(f)) return 1;
    while (g_tn > m2) g_val[g_trail[--g_tn]] = -1;
  }
  while (g_tn > mark) g_val[g_trail[--g_tn]] = -1;
  return 0;
}

/* Re-check a model against every clause (the self-validation): returns 1 iff all clauses satisfied. */
static int verify(const Cnf *f) {
  const int *p = f->lits;
  for (int ci = 0; ci < f->nclauses; ci++) {
    int sat = 0;
    const int *c = p;
    for (; *c; c++) {
      if (g_val[lit_var(*c)] == lit_want(*c)) sat = 1;
    }
    p = c + 1;
    if (!sat) return 0;
  }
  return 1;
}

static int run(Cnf *f) {
  for (int i = 0; i <= MAXV; i++) g_val[i] = -1;
  g_tn = 0;
  return solve(f);
}

/* An FNV-1a digest of the model (so the whole assignment is compared, compactly). */
static unsigned model_digest(const Cnf *f) {
  unsigned h = 2166136261u;
  for (int i = 1; i <= f->nvars; i++) {
    h = (h ^ (unsigned)(g_val[i] & 1)) * 16777619u;
  }
  return h;
}

/* A tiny deterministic LCG, so a "random" instance is identical every run and vs native. */
static unsigned g_rng;
static unsigned rnd(void) {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}

/* Build a **planted** random 3-SAT: pick a hidden assignment, then add clauses each of which the
 * hidden assignment satisfies (so the instance is SAT by construction). `ratio` clauses per var. */
static void build_planted(Cnf *f, int nvars, int nclauses, unsigned seed) {
  cnf_init(f, nvars);
  g_rng = seed;
  int hidden[MAXV + 1];
  for (int i = 1; i <= nvars; i++) hidden[i] = (int)(rnd() & 1);
  for (int k = 0; k < nclauses; k++) {
    int cl[4];
    for (;;) {
      int a = 1 + (int)(rnd() % (unsigned)nvars);
      int b = 1 + (int)(rnd() % (unsigned)nvars);
      int c = 1 + (int)(rnd() % (unsigned)nvars);
      if (a == b || a == c || b == c) continue; /* three distinct vars */
      /* random polarities, then force at least one literal true under `hidden` */
      int pa = (int)(rnd() & 1), pb = (int)(rnd() & 1), pc = (int)(rnd() & 1);
      int la = pa ? a : -a, lb = pb ? b : -b, lc = pc ? c : -c;
      int ok = (hidden[a] == lit_want(la)) || (hidden[b] == lit_want(lb)) || (hidden[c] == lit_want(lc));
      if (!ok) { /* flip the first literal so the clause holds under `hidden` */
        la = (hidden[a] == 1) ? a : -a;
      }
      cl[0] = la; cl[1] = lb; cl[2] = lc; cl[3] = 0;
      break;
    }
    cnf_add(f, cl);
  }
}

/* Pigeonhole PHP(pigeons, holes): each pigeon in some hole, no two pigeons share a hole. UNSAT
 * whenever pigeons > holes. Variable x(p,h) = (p-1)*holes + h, for p in 1..pigeons, h in 1..holes. */
static void build_php(Cnf *f, int pigeons, int holes) {
  cnf_init(f, pigeons * holes);
  int cl[MAXV + 1];
  for (int p = 1; p <= pigeons; p++) { /* pigeon p is in at least one hole */
    int n = 0;
    for (int h = 1; h <= holes; h++) cl[n++] = (p - 1) * holes + h;
    cl[n] = 0;
    cnf_add(f, cl);
  }
  for (int h = 1; h <= holes; h++) { /* at most one pigeon per hole */
    for (int p = 1; p <= pigeons; p++) {
      for (int q = p + 1; q <= pigeons; q++) {
        cl[0] = -((p - 1) * holes + h);
        cl[1] = -((q - 1) * holes + h);
        cl[2] = 0;
        cnf_add(f, cl);
      }
    }
  }
}

static Cnf f;

int main(void) {
  /* 1–3: planted random 3-SAT instances of growing size — each must be SAT, and the model must verify. */
  unsigned seeds[3] = {12345u, 0xC0FFEEu, 987654321u};
  int sizes[3][2] = {{12, 48}, {20, 84}, {30, 120}}; /* {vars, clauses} */
  for (int i = 0; i < 3; i++) {
    build_planted(&f, sizes[i][0], sizes[i][1], seeds[i]);
    int sat = run(&f);
    int ok = sat && verify(&f);
    printf("planted[%d]: vars=%d clauses=%d -> %s verify=%s digest=%08x\n",
           i, f.nvars, f.nclauses, sat ? "SAT" : "UNSAT", ok ? "OK" : "BAD",
           sat ? model_digest(&f) : 0u);
  }

  /* 4–6: pigeonhole — PHP(3,2) and PHP(4,3) are UNSAT; PHP(2,3) is trivially SAT. */
  int php[3][2] = {{3, 2}, {4, 3}, {2, 3}};
  for (int i = 0; i < 3; i++) {
    build_php(&f, php[i][0], php[i][1]);
    int sat = run(&f);
    int ok = !sat || verify(&f); /* a claimed SAT model must verify */
    printf("php(%d,%d): vars=%d clauses=%d -> %s check=%s\n",
           php[i][0], php[i][1], f.nvars, f.nclauses, sat ? "SAT" : "UNSAT", ok ? "OK" : "BAD");
  }
  return 0;
}
