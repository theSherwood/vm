#!/usr/bin/env python3
"""Python reference for the interp_perf kernels — a calibration point next to the reference
interpreter and the JIT. The functions mirror the hand-written IR in `interp_perf.rs` (wrapping
i64 arithmetic, a leaf-call recurrence, and a load/store loop), so `cargo test ... interp_perf`
can print "how does our interpreter compare to plain CPython?" alongside the interp/JIT numbers.

Usage (driven by the Rust bench; also runnable directly):
    python3 interp_perf.py <alu|call|mem> <big_iters>
prints a single float: steady-state ns/iter (isolated by the same big-small subtraction the Rust
bench uses).
"""
import sys
import time

MASK = (1 << 64) - 1
C1 = 6364136223846793005
C2 = 1442695040888963407


def alu(n):
    # acc = acc*C1 + C2 + i, wrapping i64 — the ALU recurrence.
    acc = 0
    for i in range(n):
        acc = (acc * C1 + C2 + i) & MASK
    return acc


def _leaf(a, b):
    return (a + b) & MASK


def callret(n):
    # acc = leaf(acc, i) per iteration — a direct call + return each step.
    acc = 0
    for i in range(n):
        acc = _leaf(acc, i)
    return acc


def memloop(n):
    # store acc, load it back, acc = loaded + i — one store + one load per iteration.
    mem = [0, 0]
    acc = 0
    for i in range(n):
        mem[1] = acc
        acc = (mem[1] + i) & MASK
    return acc


KERNELS = {"alu": alu, "call": callret, "mem": memloop}


def ns_per_iter(fn, big, small, reps=5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(big)
        tb = time.perf_counter() - t
        t = time.perf_counter()
        fn(small)
        ts = time.perf_counter() - t
        best = min(best, (tb - ts) / (big - small))
    return best * 1e9


if __name__ == "__main__":
    kernel = sys.argv[1] if len(sys.argv) > 1 else "alu"
    big = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    print("%.3f" % ns_per_iter(KERNELS[kernel], big, 1_000))
