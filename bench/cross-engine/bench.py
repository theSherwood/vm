import time
# CPython mirror of kernels.c — i32-LCG (masked to 32 bits), same computations as the C/SVM kernels.
def now(): return time.perf_counter_ns()
MASK = 0xffffffff
def lcg(a, i): return (a * 1103515245 + 12345 + i) & MASK

def alu(n):
    a = 0
    for i in range(n): a = lcg(a, i)
    return a
def xorshift(n):
    a = 1
    for i in range(n):
        a ^= (a << 13) & MASK; a ^= a >> 17; a = (a + i) & MASK
    return a
def step(a, i): return lcg(a, i)
def call(n):
    a = 0
    for i in range(n): a = step(a, i)
    return a
fp = step
def call_indirect(n):
    a = 0; f = fp
    for i in range(n): a = f(a, i)
    return a
def mem(n):
    cell = 0; a = 0
    for i in range(n): cell = a; a = lcg(cell, i)
    return a

CN = 4096
def chase(n):
    carr = [(i + 1789) & (CN - 1) for i in range(CN)]
    x = 0; h = 0
    for _ in range(n): x = carr[x]; h = (h + x) & MASK
    return h
RN = 1 << 20
def chase_rand(n):
    rarr = [(i * 1103515245 + 12345) & (RN - 1) for i in range(RN)]
    x = 0; h = 0
    for _ in range(n): x = rarr[x]; h = (h + x) & MASK
    return h
FBUF = 4096
def fnv(n):
    fbuf = bytearray((i * 7 + 1) & 0xff for i in range(FBUF))
    h = 2166136261
    for k in range(n): h = ((h ^ fbuf[k & (FBUF - 1)]) * 16777619) & MASK
    return h
def fma(n):
    a = 1.0
    for _ in range(n): a = a * 0.9999999 + 1.0
    return int(a)
def vadd(n):
    seed = (n * 2654435761) & MASK
    s = 0
    for k in range(n): s = (s + (k ^ seed)) & MASK
    return s

def min_run(k, n):
    k(n)
    best = float('inf')
    for _ in range(7):
        a = now(); k(n); b = now()
        if b - a < best: best = b - a
    return best
for name, k in [("alu", alu), ("xorshift", xorshift), ("call", call), ("call_indirect", call_indirect),
                ("mem", mem), ("chase", chase), ("chase_rand", chase_rand), ("fnv", fnv), ("fma", fma), ("vadd", vadd)]:
    s = min_run(k, 1000); l = min_run(k, 201000)
    print(f"python,{name},{(l - s) / 200000.0:.4f}")
