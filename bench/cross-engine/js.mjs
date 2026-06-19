// Pure-JS (V8) mirror of kernels.c — i32-LCG arithmetic via Math.imul, same computations as the C/SVM
// kernels. Per-iteration = (min time at n=201000 − at n=1000) / 200000, min over reps, warmed up.
const now = () => Number(process.hrtime.bigint());
const M = 1103515245, K = 12345;

function alu(n){ let a=0; for(let i=0;i<n;i++) a=(Math.imul(a,M)+K+i)|0; return a; }
function step(a,i){ return (Math.imul(a,M)+K+i)|0; }
function call(n){ let a=0; for(let i=0;i<n;i++) a=step(a,i); return a; }
let fp = step;
function call_indirect(n){ let a=0; const f=fp; for(let i=0;i<n;i++) a=f(a,i); return a; }
function mem(n){ let cell=0,a=0; for(let i=0;i<n;i++){ cell=a; a=(Math.imul(cell,M)+K+i)|0; } return a; }

const CN=4096, carr=new Int32Array(CN);
function chase(n){ for(let i=0;i<CN;i++) carr[i]=(i+1789)&(CN-1); let x=0,h=0; for(let k=0;k<n;k++){ x=carr[x]>>>0; h=(h+x)|0; } return h; }
const RN=1<<20, rarr=new Int32Array(RN);
function chase_rand(n){ for(let i=0;i<RN;i++) rarr[i]=(Math.imul(i,1103515245)+12345)&(RN-1); let x=0,h=0; for(let k=0;k<n;k++){ x=rarr[x]>>>0; h=(h+x)|0; } return h; }
const FBUF=4096, fbuf=new Uint8Array(FBUF);
function fnv(n){ for(let i=0;i<FBUF;i++) fbuf[i]=(i*7+1)&0xff; let h=2166136261>>>0; for(let k=0;k<n;k++) h=Math.imul(h^fbuf[k&(FBUF-1)],16777619)>>>0; return h|0; }
function fma(n){ let a=1.0; for(let k=0;k<n;k++) a=a*0.9999999+1.0; return a|0; }
const VBUF=262144, vbuf=new Int32Array(VBUF);
function vsum(n){ for(let i=0;i<VBUF;i++) vbuf[i]=i+1; let s=0; for(let k=0;k<n;k++) s=(s+vbuf[k])|0; return s; }

function minRun(k,n){ for(let w=0;w<50;w++) k(n); let best=1e18; for(let r=0;r<25;r++){ let a=now(); k(n); let b=now(); if(b-a<best)best=b-a; } return best; }
for(const [name,k] of [["alu",alu],["call",call],["call_indirect",call_indirect],["mem",mem],["chase",chase],["chase_rand",chase_rand],["fnv",fnv],["fma",fma],["vsum",vsum]]){
  const s=minRun(k,1000), l=minRun(k,201000);
  console.log(`js(v8),${name},${((l-s)/200000).toFixed(4)}`);
}
