// edn-shaped FIR/dot: two streaming loads per iter off an unbounded base — confinement-dense, ILP.
#define M 4096
static long xa[M], xb[M];
long run(long n){ for(int i=0;i<M;i++){xa[i]=i*7+1;xb[i]=i*13+5;} long s=0; for(long k=0;k<n;k++){ unsigned i=(unsigned)k&(M-1); s+=xa[i]*xb[i]; } return s; }
