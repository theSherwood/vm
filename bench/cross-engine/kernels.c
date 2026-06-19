// Cross-engine benchmark kernels — ONE source compiled to native (clang -O2), wasm32/64 (clang
// --target=wasm), and SVM IR (clang -O2 -emit-llvm -fno-*-vectorize -> svm-llvm). Fold-resistant
// *by construction* (multiplicative i32-LCG recurrences, data-dependent loads, opaque pointers)
// rather than inline-asm barriers, so the SAME source survives the LLVM->SVM on-ramp (which rejects
// inline asm); i32 throughout so JS can match via Math.imul. Each chase/array prelude is rebuilt
// inside the function (a fixed O(size) cost that cancels in the large/small-n subtraction).
#include <stdint.h>
#if defined(__wasm__)
#define EXPORT(n) __attribute__((export_name(n)))
#else
#define EXPORT(n)
#endif
// i32 LCG: multiplicative => clang can't closed-form; i32 => JS can match via Math.imul.
#define LCG(a,i) ((a)*1103515245 + 12345 + (i))
EXPORT("alu") int32_t alu(int32_t n){ int32_t a=0; for(int32_t i=0;i<n;i++) a=LCG(a,i); return a; }
__attribute__((noinline)) static int32_t step(int32_t a,int32_t i){ return LCG(a,i); }
EXPORT("call") int32_t call(int32_t n){ int32_t a=0; for(int32_t i=0;i<n;i++) a=step(a,i); return a; }
static int32_t (* volatile fp)(int32_t,int32_t)=step;
EXPORT("call_indirect") int32_t call_indirect(int32_t n){ int32_t a=0; int32_t(*f)(int32_t,int32_t)=fp; for(int32_t i=0;i<n;i++) a=f(a,i); return a; }
static int32_t cell;
EXPORT("mem") int32_t mem(int32_t n){ int32_t a=0; for(int32_t i=0;i<n;i++){ cell=a; a=LCG(cell,i); } return a; }
#define CN 4096u
static int32_t carr[CN];
EXPORT("chase") int32_t chase(int32_t n){ for(uint32_t i=0;i<CN;i++) carr[i]=(int32_t)((i+1789u)&(CN-1u)); uint32_t x=0; int32_t h=0; for(int32_t k=0;k<n;k++){ x=(uint32_t)carr[x]; h+=(int32_t)x; } return h; }
#define RN (1u<<20)
static int32_t rarr[RN];
EXPORT("chase_rand") int32_t chase_rand(int32_t n){ for(uint32_t i=0;i<RN;i++) rarr[i]=(int32_t)((i*1103515245u+12345u)&(RN-1u)); uint32_t x=0; int32_t h=0; for(int32_t k=0;k<n;k++){ x=(uint32_t)rarr[x]; h+=(int32_t)x; } return h; }
#define FBUF 4096u
static uint8_t fbuf[FBUF];
EXPORT("fnv") int32_t fnv(int32_t n){ for(uint32_t i=0;i<FBUF;i++) fbuf[i]=(uint8_t)((i*7u+1u)&0xffu); uint32_t h=2166136261u; for(int32_t k=0;k<n;k++) h=(h^fbuf[(uint32_t)k&(FBUF-1u)])*16777619u; return (int32_t)h; }
EXPORT("fma") int32_t fma_k(int32_t n){ double a=1.0; for(int32_t k=0;k<n;k++) a=a*0.9999999+1.0; return (int32_t)a; }
#define VBUF 262144u
static int32_t vbuf[VBUF];
static int32_t * volatile vptr=vbuf;
EXPORT("vsum") int32_t vsum(int32_t n){ for(uint32_t i=0;i<VBUF;i++) vbuf[i]=(int32_t)(i+1u); int32_t* p=vptr; int32_t s=0; for(int32_t k=0;k<n;k++) s+=p[k]; return s; }
