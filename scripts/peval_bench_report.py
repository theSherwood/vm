#!/usr/bin/env python3
"""Regenerate the partial-evaluation performance report (PEVAL_BENCH.md).

Runs the CSV-emitting partial-evaluation benches in `svm-peval` and `svm-llvm`
with `SVM_BENCH_CSV=1`, collects the `CSV,<bench>,<case>,<metric>,<value>` rows
they print, and renders one consolidated markdown report.

    python3 scripts/peval_bench_report.py [output.md]   # default: PEVAL_BENCH.md

It is slow (~2 min): it runs the `--ignored` timing benches, including the
end-to-end Futamura ROI loop. Timings are single-run and machine-dependent —
the report records the host so numbers are interpreted in context. `svm-llvm`
needs `clang` on PATH (it compiles the real BF/Lisp interpreters from C).
"""

import os
import sys
import subprocess
import datetime
import platform
import pathlib
import collections

REPO = pathlib.Path(__file__).resolve().parent.parent

# (label, cwd, argv) — each command prints CSV rows on stdout when SVM_BENCH_CSV is set.
BENCHES = [
    (
        "size_corpus",
        REPO,
        ["cargo", "test", "-p", "svm-peval", "--test", "bench", "size_corpus",
         "--", "--nocapture"],
    ),
    (
        "gain_spectrum",
        REPO,
        ["cargo", "test", "-p", "svm-peval", "--test", "bench", "gain_spectrum",
         "--", "--ignored", "--nocapture"],
    ),
    (
        "roi_futamura_loop",
        REPO,
        ["cargo", "test", "-p", "svm-peval", "--test", "bench", "roi_futamura_loop",
         "--", "--ignored", "--nocapture"],
    ),
    (
        "peval_corpus",
        REPO / "crates" / "svm-llvm",
        ["cargo", "test", "--test", "peval_corpus", "corpus_metric_matrix",
         "--", "--ignored", "--nocapture"],
    ),
]

# Human-readable one-liners for each bench section.
DESCRIPTIONS = {
    "size_corpus": "Static size reduction across toy interpreter shapes (svm-peval).",
    "gain_spectrum": "Toy gain gradient: folding programs, then loops with growing per-iteration "
                     "work — JIT run-time speedup, compile excluded (svm-peval).",
    "roi_futamura_loop": "End-to-end Futamura ROI: sum 1..N as a register-machine loop, all four "
                         "execution configs (svm-peval).",
    "peval_corpus": "Real clang-compiled interpreters (Brainfuck + Lisp) across a range of guest "
                    "programs: size, PE/compile time, and run-time speedup (svm-llvm).",
}


def run_all():
    """Run every bench; return rows [(bench, case, metric, value)] and any warnings."""
    rows, warnings = [], []
    env = dict(os.environ, SVM_BENCH_CSV="1")
    for label, cwd, argv in BENCHES:
        print(f"  running {label} (cwd={cwd.name}) …", file=sys.stderr)
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-12:])
            warnings.append(f"{label} exited {proc.returncode}:\n{tail}")
            print(f"  WARN: {label} exited {proc.returncode}", file=sys.stderr)
        n = 0
        for line in proc.stdout.splitlines():
            if line.startswith("CSV,"):
                parts = line.split(",", 4)
                if len(parts) == 5:
                    rows.append(tuple(parts[1:]))
                    n += 1
        print(f"    {n} CSV rows", file=sys.stderr)
    return rows, warnings


def metric_sort_key(metric):
    """Group metrics: sizes, then %, then times, then speedups (speedup@N by N)."""
    if metric.endswith("bytes"):
        return (0, metric)
    if metric.endswith("pct"):
        return (1, metric)
    if metric.endswith("_ms") or metric == "time_ms":
        return (2, metric)
    if metric.startswith("speedup@"):
        try:
            return (3, f"{int(metric.split('@')[1]):020d}")
        except ValueError:
            return (3, metric)
    if "speedup" in metric:
        return (4, metric)
    return (5, metric)


def fmt(metric, value):
    try:
        x = float(value)
    except ValueError:
        return value
    if metric.endswith("bytes"):
        return str(int(round(x)))
    if metric.endswith("pct"):
        return f"{x:.0f}%"
    if metric.endswith("_ms") or metric == "time_ms":
        return f"{x:.3f}"
    if "speedup" in metric:
        return f"{x:.1f}×"
    return f"{x:g}"


def render(rows, warnings):
    # bench -> case -> metric -> value, preserving first-seen order of cases/metrics.
    benches = collections.OrderedDict()
    for bench, case, metric, value in rows:
        benches.setdefault(bench, collections.OrderedDict()).setdefault(case, {})[metric] = value

    commit = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip() or "unknown"
    except OSError:
        pass
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out = []
    out.append("# Partial-evaluation performance report\n")
    out.append(
        "_Generated by `scripts/peval_bench_report.py` — regenerate with "
        "`python3 scripts/peval_bench_report.py`._\n"
    )
    out.append(
        f"- generated: {now}\n"
        f"- commit: `{commit}`\n"
        f"- host: {platform.platform()} / {platform.processor() or platform.machine()}\n"
    )
    out.append(
        "\nTimings are JIT, **compile-once/run-many** (compilation excluded), single-run and "
        "machine-dependent. Specialization pays off on three axes: **size** (residual bytes vs the "
        "interpreter), **compile time** (a smaller residual JITs faster), and **run time** (only for "
        "programs that do real runtime work — folding programs collapse to a trivial residual, so "
        "their win is size + compile). `speedup@N` is the run-time speedup at workload size N.\n")

    for bench, cases in benches.items():
        out.append(f"\n## {bench}\n")
        if bench in DESCRIPTIONS:
            out.append(f"\n{DESCRIPTIONS[bench]}\n")
        metrics = sorted({m for c in cases.values() for m in c}, key=metric_sort_key)
        out.append("\n| case | " + " | ".join(metrics) + " |")
        out.append("|" + "---|" * (len(metrics) + 1))
        for case, mv in cases.items():
            cells = [fmt(m, mv[m]) if m in mv else "—" for m in metrics]
            out.append(f"| {case} | " + " | ".join(cells) + " |")
        out.append("")

    if warnings:
        out.append("\n## warnings\n")
        for w in warnings:
            out.append("```\n" + w + "\n```")

    # Machine-readable appendix: the raw CSV, so the report is self-contained and diffable.
    out.append("\n## raw CSV\n")
    out.append("```")
    out.append("bench,case,metric,value")
    out.extend(",".join(r) for r in rows)
    out.append("```")
    return "\n".join(out) + "\n"


def main():
    out_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "PEVAL_BENCH.md"
    rows, warnings = run_all()
    if not rows:
        print("ERROR: no CSV rows collected — did the benches build/run?", file=sys.stderr)
        return 1
    out_path.write_text(render(rows, warnings))
    print(f"wrote {out_path} ({len(rows)} metric rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
