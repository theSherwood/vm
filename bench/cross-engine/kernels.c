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
