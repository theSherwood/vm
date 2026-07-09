// Faithful embench matmult-int shape — the kernel `matmul.c` should have been. Where `matmul.c` is a
// clean `long`-index, register-accumulator multiply (which measures ~parity, 1.05x), this reproduces
// the real embench `matmult-int` inner structure: `int` (32-bit) indices, a memory-accumulated
// `Res[Outer][Inner] +=`, and — the ingredient that matters — a per-iteration bulk array copy.
//
// That copy is why svm-jit trails Wasmtime-w64 here (~1.3x). LLVM lowers `da[k]=sa[k]` over the whole
// array to a `memcpy`; the wasm64 lane (built with -mbulk-memory) turns it into a single `memory.copy`
// that is range-checked once, while svm-llvm lowers the same memcpy to an *inline chunked copy* where
// every 8-byte chunk is an individually confinement-masked load/store (see MAX_MEM_UNROLL in
// crates/svm-llvm/src/lib.rs). Two 3200-byte copies/iter => ~1600 masked accesses vs ~2 range checks.
#define N 20
typedef long matrix[N][N];
static matrix ArrayA_ref, ArrayA, ArrayB_ref, ArrayB, ResultArray;

__attribute__((noinline)) static void Multiply(matrix A, matrix B, matrix Res){
  register int Outer, Inner, Index;
  for (Outer = 0; Outer < N; Outer++)
    for (Inner = 0; Inner < N; Inner++){
      Res[Outer][Inner] = 0;
      for (Index = 0; Index < N; Index++)
        Res[Outer][Inner] += A[Outer][Index] * B[Index][Inner];
    }
}

long run(long n){
  long seed = 0;
  for (int i=0;i<N;i++) for (int j=0;j<N;j++){ seed=((seed*133)+81)%8095; ArrayA_ref[i][j]=seed; }
  for (int i=0;i<N;i++) for (int j=0;j<N;j++){ seed=((seed*133)+81)%8095; ArrayB_ref[i][j]=seed; }
  long *da=(long*)ArrayA,*sa=(long*)ArrayA_ref,*db=(long*)ArrayB,*sb=(long*)ArrayB_ref;
  long h=0;
  for (long t=0;t<n;t++){
    for (int k=0;k<N*N;k++){ da[k]=sa[k]; db[k]=sb[k]; }  // LLVM idiom-recognizes this as memcpy
    Multiply(ArrayA, ArrayB, ResultArray);
    h += ResultArray[t%N][t%N];
  }
  return h;
}
