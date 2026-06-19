import subprocess, time, sys
import os
WT=os.environ.get("WASMTIME","/tmp/wasmtime-v45.0.2-x86_64-linux/wasmtime")
NS, NL, REPS = 1_000_000, 100_000_000, 3
def t(args):
    best=1e18
    for _ in range(REPS):
        a=time.perf_counter()
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        b=time.perf_counter()
        best=min(best,b-a)
    return best
def bench(label, module, extra):
    for k in ["alu","call","call_indirect","mem","chase","chase_rand","fnv","fma"]:  # vsum omitted: too fast for CLI timing + needs n<=VBUF
        s=t([WT,"run",*extra,"--invoke",k,module,str(NS)])
        l=t([WT,"run",*extra,"--invoke",k,module,str(NL)])
        print(f"{label},{k},{(l-s)/(NL-NS)*1e9:.4f}", flush=True)
bench("wasm32(wasmtime)","k32.wasm",[])
bench("wasm64(wasmtime)","k64.wasm",["-W","memory64"])
