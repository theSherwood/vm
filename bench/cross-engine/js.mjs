const now=()=>Number(process.hrtime.bigint());
function alu(n){let acc=0;while(n){acc=(acc+n)|0;n-=1;}return acc;}
function leaf(x){return (x+1)|0;}
function call(n){let acc=0;while(n){acc=leaf(acc);n-=1;}return acc;}
const T=[leaf];
function call_indirect(n){let acc=0;while(n){acc=T[0](acc);n-=1;}return acc;}
const cell=new Int32Array(1);
function mem(n){let acc=0;while(n){cell[0]=acc;acc=cell[0]+1|0;n-=1;}return acc;}
const CN=4096, RN=1<<20;
const carr=new Int32Array(CN), rarr=new Int32Array(RN);
function chase(n){ for(let i=0;i<CN;i++) carr[i]=(i+1789)&(CN-1); let idx=0,hops=0; while(n){idx=carr[idx]>>>0;hops+=idx;n-=1;} return hops; }
function chase_rand(n){ for(let i=0;i<RN;i++) rarr[i]=(Math.imul(i,1103515245)+12345)&(RN-1); let idx=0,hops=0; while(n){idx=rarr[idx]>>>0;hops+=idx;n-=1;} return hops; }
function minRun(k,n){for(let w=0;w<50;w++)k(n);let best=1e18;for(let r=0;r<25;r++){let a=now();k(n);let b=now();if(b-a<best)best=b-a;}return best;}
for(const [name,k] of [["alu",alu],["call",call],["call_indirect",call_indirect],["mem",mem],["chase",chase],["chase_rand",chase_rand]]){
  const s=minRun(k,1000),l=minRun(k,201000);
  console.log(`js(v8),${name},${((l-s)/200000).toFixed(4)}`);
}
