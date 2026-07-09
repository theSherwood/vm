// embench matmult-int shape: 20x20 i64 matrices — dense array loop, unbounded-base accesses.
#define N 20
typedef long mat[N][N];
static mat A, B, C;
__attribute__((noinline)) static void mul(mat a, mat b, mat r){
  for (int i=0;i<N;i++) for (int j=0;j<N;j++){ long s=0; for (int k=0;k<N;k++) s+=a[i][k]*b[k][j]; r[i][j]=s; }
}
long run(long n){ long h=0; for (long t=0;t<n;t++){ for(int i=0;i<N;i++)for(int j=0;j<N;j++){A[i][j]=i*N+j+t;B[i][j]=(i^j)+t;} mul(A,B,C); h+=C[t%N][t%N]; } return h; }
