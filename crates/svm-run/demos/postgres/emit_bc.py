#!/usr/bin/env python3
"""Emit per-TU LLVM bitcode for the Postgres backend, reusing the makefile's exact compile
flags. For each native `.o`, ask `make -n` for its clang command (after removing the up-to-date
`.o` so the rule prints), rewrite it to `-emit-llvm -o X.bc`, and run it. See README.md.

    SVM_PG_SRC=/path/to/postgresql-XX.Y python3 emit_bc.py

Writes bc_manifest.txt next to the source tree's parent (SVM_PG_CACHE)."""
import os, subprocess, shlex, concurrent.futures

SRC = os.environ.get("SVM_PG_SRC")
CACHE = os.environ.get("SVM_PG_CACHE", "/tmp/svm_pg_cache")
assert SRC and os.path.isdir(SRC), "set SVM_PG_SRC to the extracted postgres tree"
DIRS = ["src/backend", "src/common", "src/port", "src/timezone"]
EXTRA = ["-emit-llvm", "-fno-vectorize", "-fno-slp-vectorize"]


def find_objs():
    objs = []
    for d in DIRS:
        for root, _, files in os.walk(os.path.join(SRC, d)):
            objs += [os.path.join(root, f) for f in files if f.endswith(".o")]
    return objs


def compile_cmd(obj):
    d, base = os.path.dirname(obj), os.path.basename(obj)
    # Remove the up-to-date .o so `make -n` (no -B, so no config.status recheck) prints the rule.
    try:
        os.remove(obj)
    except OSError:
        pass
    out = subprocess.run(["make", "-n", base], cwd=d, capture_output=True, text=True, timeout=120).stdout
    line = next((l.strip() for l in out.splitlines()
                 if l.strip().startswith("clang ") and " -c " in l and (" -o " + base) in l), None)
    if not line:
        return (obj, None)
    toks, out_toks, bc, i = shlex.split(line), [], obj[:-2] + ".bc", 0
    while i < len(toks):
        t = toks[i]
        if t == "-o":
            out_toks += ["-o", bc]; i += 2; continue
        if t == "-ftree-vectorize":
            i += 1; continue
        out_toks.append(t); i += 1
    out_toks[1:1] = EXTRA
    return (obj, (d, out_toks, bc))


def main():
    objs = find_objs()
    print(f"backend link set: {len(objs)} objects", flush=True)
    cmds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for _obj, cmd in ex.map(compile_cmd, objs):
            if cmd:
                cmds.append(cmd)
    print(f"compile commands recovered: {len(cmds)}", flush=True)

    def run(c):
        d, toks, bc = c
        return subprocess.run(toks, cwd=d, capture_output=True, text=True).returncode

    fails, done = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        for rc in ex.map(run, cmds):
            done += 1
            fails += rc != 0
    print(f"bitcode: {len(cmds) - fails}/{len(cmds)} ok, {fails} failed", flush=True)
    with open(os.path.join(CACHE, "bc_manifest.txt"), "w") as f:
        for _d, _t, bc in cmds:
            if os.path.exists(bc):
                f.write(bc + "\n")


main()
