// A exact-rational arithmetic evaluator — a second, more demanding sandboxed program:
//
//   svm-run crates/svm-run/demos/rational.c
//
// Where calc.c stresses recursion + a function-pointer table, this hammers the **by-value
// aggregate ABI** (§3d D39): every operation takes two `struct Rat` *by value* and returns one
// *by value* (the hidden-sret path), composed with **recursion** (Euclid's `gcd`) and an
// **indirect call returning a struct by value** through a global dispatch table (sret + a
// function-pointer relocation + a struct-valued indirect call, all at once). Output is via
// `write`, so it compiles with `cc` too and the two builds are directly comparable.

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
    b[i++] = (char)('0' + v % 10);
    v /= 10;
  }
  while (i) {
    char c = b[--i];
    write(1, &c, 1);
  }
}

struct Rat {
  int n, d;
};

static int gcd(int a, int b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  return b ? gcd(b, a % b) : a;
}

// Returns a struct *by value* (sret), taking one by value — and recurses through gcd.
static struct Rat reduce(struct Rat r) {
  if (r.d < 0) {
    r.n = -r.n;
    r.d = -r.d;
  }
  int g = gcd(r.n, r.d);
  if (g == 0) g = 1;
  struct Rat o;
  o.n = r.n / g;
  o.d = r.d / g;
  return o;
}

static struct Rat radd(struct Rat a, struct Rat b) {
  struct Rat r;
  r.n = a.n * b.d + b.n * a.d;
  r.d = a.d * b.d;
  return reduce(r);
}
static struct Rat rsub(struct Rat a, struct Rat b) {
  struct Rat r;
  r.n = a.n * b.d - b.n * a.d;
  r.d = a.d * b.d;
  return reduce(r);
}
static struct Rat rmul(struct Rat a, struct Rat b) {
  struct Rat r;
  r.n = a.n * b.n;
  r.d = a.d * b.d;
  return reduce(r);
}

// A global dispatch table of struct-returning operators (function-pointer relocations).
struct Binop {
  char c;
  struct Rat (*fn)(struct Rat, struct Rat);
};
struct Binop binops[] = {{'+', radd}, {'-', rsub}, {'*', rmul}};

// An indirect call that both passes and returns a `struct Rat` by value.
static struct Rat apply(char c, struct Rat a, struct Rat b) {
  for (int i = 0; i < 3; i++)
    if (binops[i].c == c)
      return binops[i].fn(a, b);
  struct Rat z = {0, 1};
  return z;
}

static void print_rat(struct Rat r) {
  putint(r.n);
  if (r.d != 1) {
    char slash = '/';
    write(1, &slash, 1);
    putint(r.d);
  }
}

static struct Rat rat(int n, int d) {
  struct Rat r = {n, d};
  return reduce(r);
}

int main(void) {
  struct Rat a = rat(1, 2), b = rat(1, 3);
  char *labels[] = {"1/2 + 1/3 = ", "1/2 - 1/3 = ", "1/2 * 1/3 = "};
  char ops[] = {'+', '-', '*'};
  for (int i = 0; i < 3; i++) {
    puts_(labels[i]);
    print_rat(apply(ops[i], a, b));
    puts_("\n");
  }

  // Sum 1/1 + 1/2 + ... + 1/6 as an exact rational (a chain of by-value struct returns).
  struct Rat sum = rat(0, 1);
  for (int k = 1; k <= 6; k++)
    sum = radd(sum, rat(1, k));
  puts_("sum(1/1..1/6) = ");
  print_rat(sum);
  puts_("\n");
  return 0;
}
