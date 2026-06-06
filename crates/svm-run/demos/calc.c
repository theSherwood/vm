// A recursive-descent arithmetic calculator — a real, self-contained program run sandboxed:
//
//   svm-run crates/svm-run/demos/calc.c
//
// It deliberately exercises the whole stack: a **global array of string pointers** and a
// **global struct array holding function pointers** (both are relocations, §3a), **indirect
// calls** through that dispatch table (§3c), **recursion** (expr → term → factor), pointers,
// char arithmetic, and stack arrays. Output goes through `write` — a powerbox builtin on the
// VM (lowered to the Stream capability) and the real `write(2)` when compiled with `cc`, so the
// same source is directly comparable against a native build.

int write(int fd, char *buf, long n);

static void puts_(char *s) {
  long n = 0;
  while (s[n]) n++;
  write(1, s, n);
}

static void putint(int v) {
  char b[16];
  int i = 0;
  if (v < 0) {
    char m = '-';
    write(1, &m, 1);
    v = -v;
  }
  if (v == 0) {
    char z = '0';
    write(1, &z, 1);
    return;
  }
  while (v) {
    b[i++] = '0' + v % 10;
    v /= 10;
  }
  while (i) {
    char c = b[--i];
    write(1, &c, 1);
  }
}

// The operator dispatch table: a global array of structs, each pairing a character with a
// function pointer. Both the entries' function pointers are relocations resolved at compile
// time, and `apply` dispatches through them with an indirect call.
static int add(int a, int b) { return a + b; }
static int sub(int a, int b) { return a - b; }
static int mul(int a, int b) { return a * b; }
static int dvd(int a, int b) { return b ? a / b : 0; }

struct Op {
  char c;
  int (*fn)(int, int);
};
struct Op ops[] = {{'+', add}, {'-', sub}, {'*', mul}, {'/', dvd}};

static int apply(char c, int a, int b) {
  for (int i = 0; i < 4; i++)
    if (ops[i].c == c)
      return ops[i].fn(a, b);
  return 0;
}

// A global parse cursor + a recursive-descent grammar with the usual precedence:
//   expr   = term   (('+' | '-') term)*
//   term   = factor (('*' | '/') factor)*
//   factor = number | '(' expr ')'
static char *cur;

static int peek(void) {
  while (*cur == ' ')
    cur++;
  return *cur;
}

static int parse_expr(void);

static int parse_factor(void) {
  if (peek() == '(') {
    cur++; // '('
    int v = parse_expr();
    peek();
    cur++; // ')'
    return v;
  }
  int n = 0;
  while (*cur >= '0' && *cur <= '9') {
    n = n * 10 + (*cur - '0');
    cur++;
  }
  return n;
}

static int parse_term(void) {
  int v = parse_factor();
  for (;;) {
    int c = peek();
    if (c == '*' || c == '/') {
      cur++;
      v = apply((char)c, v, parse_factor());
    } else {
      return v;
    }
  }
}

static int parse_expr(void) {
  int v = parse_term();
  for (;;) {
    int c = peek();
    if (c == '+' || c == '-') {
      cur++;
      v = apply((char)c, v, parse_term());
    } else {
      return v;
    }
  }
}

static int eval(char *s) {
  cur = s;
  return parse_expr();
}

// The program drives itself from a global table of expressions (string-literal relocations),
// printing "<expr> = <result>" for each — no stdin needed.
char *exprs[] = {
    "1 + 2 * 3",      "(1 + 2) * 3",   "100 / 7",        "2 * (3 + 4) - 5",
    "((1 + 2) * 3)",  "10 - 2 - 3",    "1000 / 5 / 2",   "42",
};

int main(void) {
  int n = (int)(sizeof(exprs) / sizeof(exprs[0]));
  for (int i = 0; i < n; i++) {
    puts_(exprs[i]);
    puts_(" = ");
    putint(eval(exprs[i]));
    puts_("\n");
  }
  return 0;
}
