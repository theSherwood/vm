import time, sys
def now(): return time.perf_counter_ns()
def alu(n):
    acc=0
    while n: acc=(acc+n)&0xffffffff; n-=1
    return acc
def leaf(x): return (x+1)&0xffffffff
def call(n):
    acc=0
    while n: acc=leaf(acc); n-=1
    return acc
TABLE=[leaf]
def call_indirect(n):
    acc=0
    while n: acc=TABLE[0](acc); n-=1
    return acc
def mem(n):
    cell=[0]
    acc=0
    while n: cell[0]=acc; acc=cell[0]+1; n-=1
    return acc
CN=4096; RN=1<<20; M32=0xffffffff
def chase(n):
    carr=[(i+1789)&(CN-1) for i in range(CN)]
    idx=0; hops=0
    while n: idx=carr[idx]; hops+=idx; n-=1
    return hops
def chase_rand(n):
    rarr=[(i*1103515245+12345)&(RN-1) for i in range(RN)]
    idx=0; hops=0
    while n: idx=rarr[idx]; hops+=idx; n-=1
    return hops
def min_run(k,n):
    k(n)
    best=float('inf')
    for _ in range(7):
        a=now(); k(n); b=now()
        if b-a<best: best=b-a
    return best
for name,k in [("alu",alu),("call",call),("call_indirect",call_indirect),("mem",mem),("chase",chase),("chase_rand",chase_rand)]:
    s=min_run(k,1000); l=min_run(k,201000)
    print(f"python,{name},{(l-s)/200000.0:.4f}")
