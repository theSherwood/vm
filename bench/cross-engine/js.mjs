const now=()=>Number(process.hrtime.bigint());
function alu(n){let acc=0;while(n){acc=(acc+n)|0;n-=1;}return acc;}
function leaf(x){return (x+1)|0;}
function call(n){let acc=0;while(n){acc=leaf(acc);n-=1;}return acc;}
const T=[leaf];
function call_indirect(n){let acc=0;while(n){acc=T[0](acc);n-=1;}return acc;}
const cell=new Int32Array(1);
function mem(n){let acc=0;while(n){cell[0]=acc;acc=cell[0]+1|0;n-=1;}return acc;}
function minRun(k,n){for(let w=0;w<50;w++)k(n);let best=1e18;for(let r=0;r<25;r++){let a=now();k(n);let b=now();if(b-a<best)best=b-a;}return best;}
for(const [name,k] of [["alu",alu],["call",call],["call_indirect",call_indirect],["mem",mem]]){
  const s=minRun(k,1000),l=minRun(k,201000);
  console.log(`js(v8),${name},${((l-s)/200000).toFixed(4)}`);
}
