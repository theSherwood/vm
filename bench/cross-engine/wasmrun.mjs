import {readFileSync} from 'fs';
function load(f){ return new WebAssembly.Instance(new WebAssembly.Module(readFileSync(f)),{}).exports; }
const now=()=>Number(process.hrtime.bigint());
function minRun(k,n){
  // warm up generously so V8 tiers up to optimized (TurboFan) before timing
  for(let w=0;w<50;w++) k(n);
  let best=1e18;
  for(let r=0;r<25;r++){ let a=now(); k(n); let b=now(); if(b-a<best)best=b-a; }
  return best;
}
function bench(label, ex){
  for(const name of ["alu","call","call_indirect","mem","chase","chase_rand","fnv","fma","vsum"]){
    const k=ex[name];
    const s=minRun(k,1000), l=minRun(k,201000);
    console.log(`${label},${name},${((l-s)/200000).toFixed(4)}`);
  }
}
bench("wasm32(v8)", load(process.argv[2]));
bench("wasm64(v8)", load(process.argv[3]));
