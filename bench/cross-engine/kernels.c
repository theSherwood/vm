#include <stdint.h>
// Per-iteration optimization barrier: emits NO instructions, but blocks the compiler from
// closed-form-folding or eliding the loop, so native AND wasm honestly execute all n iterations
// (the same work the SVM engines do). Works on x86-64 and the wasm targets.
#define DNO(x) __asm__ __volatile__("" : "+r"(x))
#if defined(__wasm__)
#define EXPORT(n) __attribute__((export_name(n)))
#else
#define EXPORT(n)
#endif

// All arithmetic is int32 to mirror the SVM kernels' i32 ops.

EXPORT("alu")
int32_t alu(int32_t n){ int32_t acc=0; while(n){ acc+=n; n-=1; DNO(acc); DNO(n); } return acc; }

__attribute__((noinline))
static int32_t leaf(int32_t x){ return x+1; }

EXPORT("call")
int32_t call(int32_t n){ int32_t acc=0; while(n){ acc=leaf(acc); n-=1; DNO(acc); DNO(n); } return acc; }

typedef int32_t (*fp)(int32_t);
static volatile fp table_slot = leaf;  // opaque → a real indirect call each iteration

EXPORT("call_indirect")
int32_t call_indirect(int32_t n){ int32_t acc=0; while(n){ fp f=table_slot; acc=f(acc); n-=1; DNO(acc); DNO(n); } return acc; }

static int32_t cell;  // plain: matches SVM mem IR (optimizers may forward store->load)

EXPORT("mem")
int32_t mem(int32_t n){ int32_t acc=0; while(n){ cell=acc; acc=cell+1; n-=1; DNO(acc); DNO(n); } return acc; }

// --- memory kernels that genuinely execute (a dependent-load pointer chase) -------------------
// Each load's ADDRESS is the previous load's VALUE, so the access is strictly serial: it can't be
// forwarded, hoisted, vectorized, or unrolled-for-ILP. The chain is rebuilt inside the function (a
// fixed O(size) prelude that cancels in the large/small-n subtraction), matching the SVM IR.
#define CN 4096u            // `chase`: 16 KiB → L1; constant stride (prefetcher-friendly: load-issue path)
#define RN (1u<<20)         // `chase_rand`: 4 MiB → L3; LCG permutation (prefetcher-defeating: cache latency)
static int32_t carr[CN];
static int32_t rarr[RN];

EXPORT("chase")
int64_t chase(int32_t n){
  for(uint32_t i=0;i<CN;i++) carr[i]=(int32_t)((i+1789u)&(CN-1u));   // constant-stride cycle
  uint32_t idx=0; int64_t hops=0;
  while(n){ idx=(uint32_t)carr[idx]; hops+=idx; n-=1; DNO(idx); DNO(n); }
  return hops;
}

EXPORT("chase_rand")
int64_t chase_rand(int32_t n){
  for(uint32_t i=0;i<RN;i++) rarr[i]=(int32_t)((i*1103515245u+12345u)&(RN-1u)); // full-period LCG perm
  uint32_t idx=0; int64_t hops=0;
  while(n){ idx=(uint32_t)rarr[idx]; hops+=idx; n-=1; DNO(idx); DNO(n); }
  return hops;
}
