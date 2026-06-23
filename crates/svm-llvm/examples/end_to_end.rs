//! **End-to-end real programs.** The micro-kernels (`cross_engine`) and the backend differential
//! (`corpus_diff`) are single-algorithm loops. This driver runs *whole small programs* — branchy
//! state machines over generated inputs — through the real LLVM frontend across **all four engines**
//! (native `clang -O2`, svm tree-walk / bytecode / JIT), to see how the stack holds up on realistic,
//! control-flow-heavy code rather than tight arithmetic. Each program is self-contained (no libc,
//! deterministic input built from a seed) and returns an `i64` checksum, so every engine's result is
//! **cross-checked bit-exact** against native — this is simultaneously a benchmark and a whole-stack
//! differential test.
//!
//! Programs (`long run(long n)` loops the whole workload `n` times, folding a per-iteration checksum):
//! - `json` — build a small JSON object from the seed, then tokenize it (string/number scan, nesting
//!   depth) and sum the integer values: a realistic parser inner loop.
//! - `dfa` — generate text and count substrings matching `[a-z]+@[a-z]+\.[a-z]+` via a hand-coded
//!   scanner state machine (the shape of a regex/lexer).
//! - `lz` — an LZ77-style compressor: longest-match search in a 16-byte sliding window over a
//!   semi-repetitive buffer, counting output tokens (memory + match-search heavy).
//! - `vm` — a tiny stack-machine **bytecode interpreter** executing a generated program
//!   (push/add/mul/sub/xor/dup): an interpreter running on the SVM.
//!
//! Run: cd crates/svm-llvm && cargo run --release --example end_to_end

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use svm_interp::{bytecode, Value};

// (name, large_fast, large_bc, large_tw, src). The JIT recompiles every call (~6 ms), so it needs a
// big `n` for the compile to wash out of the per-iter subtraction; the interpreters are 20–150× slower
// per op, so they need a *small* `n` to finish — hence per-engine iteration counts (native uses the
// fast count too). small_n is fixed at 100 for all.
const PROGRAMS: &[(&str, i64, i64, i64, &str)] = &[
    (
        "json",
        2_000_000,
        50_000,
        20_000,
        r#"
long run(long n){ char buf[512]; long acc=0;
  for(long k=0;k<n;k++){
    unsigned s=(unsigned)k*2654435761u; int p=0; buf[p++]='{';
    int fields=3+(int)(s&3);
    for(int f=0;f<fields;f++){ if(f) buf[p++]=',';
      buf[p++]='"'; buf[p++]=(char)('a'+(int)((s>>(f*3))&7)); buf[p++]='"'; buf[p++]=':';
      int v=(int)((s>>(f*2))&0x3ff); char t[8]; int ti=0;
      if(v==0)t[ti++]='0'; while(v){t[ti++]=(char)('0'+v%10); v/=10;}
      while(ti) buf[p++]=t[--ti]; }
    buf[p++]='}';
    long sum=0; int depth=0; int i=0;
    while(i<p){ char c=buf[i++];
      if(c=='{'||c=='[') depth++;
      else if(c=='}'||c==']') depth--;
      else if(c==':'){ int v=0; while(i<p&&buf[i]>='0'&&buf[i]<='9'){v=v*10+(buf[i]-'0');i++;} sum+=v; } }
    acc+=sum+depth; }
  return acc; }"#,
    ),
    (
        "dfa",
        2_000_000,
        30_000,
        8_000,
        r#"
long run(long n){ char buf[256]; long acc=0;
  for(long k=0;k<n;k++){
    unsigned s=(unsigned)k*2654435761u; int len=0;
    for(int i=0;i<128;i++){ unsigned r=(s=s*1664525u+1013904223u);
      int m=(int)(r%16); char c; if(m==0)c='@'; else if(m==1)c='.'; else c=(char)('a'+(int)(r%26));
      buf[len++]=c; }
    long matches=0; int i=0;
    while(i<len){ int j=i,a=0; while(j<len&&buf[j]>='a'&&buf[j]<='z'){j++;a++;}
      if(a>0&&j<len&&buf[j]=='@'){ j++; int b=0; while(j<len&&buf[j]>='a'&&buf[j]<='z'){j++;b++;}
        if(b>0&&j<len&&buf[j]=='.'){ j++; int c=0; while(j<len&&buf[j]>='a'&&buf[j]<='z'){j++;c++;}
          if(c>0){ matches++; i=j; continue; } } }
      i++; }
    acc+=matches; }
  return acc; }"#,
    ),
    (
        "lz",
        1_000_000,
        8_000,
        2_000,
        r#"
long run(long n){ unsigned char src[64]; long acc=0;
  for(long k=0;k<n;k++){
    unsigned s=(unsigned)k*2654435761u;
    for(int i=0;i<64;i++){ unsigned r=(s=s*1664525u+1013904223u);
      src[i]=(unsigned char)((r>>16)&((i&15)<4?3u:0xffu)); }
    long out=0; int i=0;
    while(i<64){ int bestlen=0; int wstart=i>16?i-16:0;
      for(int j=wstart;j<i;j++){ int l=0; while(i+l<64&&l<8&&src[j+l]==src[i+l]) l++;
        if(l>bestlen) bestlen=l; }
      if(bestlen>=3){ out++; i+=bestlen; } else { out++; i++; } }
    acc+=out; }
  return acc; }"#,
    ),
    (
        "vm",
        2_000_000,
        40_000,
        15_000,
        r#"
long run(long n){ long acc=0;
  for(long k=0;k<n;k++){
    unsigned s=(unsigned)k*2654435761u; int op[64],arg[64],plen=0;
    for(int i=0;i<48;i++){ unsigned r=(s=s*1664525u+1013904223u); op[plen]=(int)(r%6); arg[plen]=(int)((r>>8)&0xff); plen++; }
    long st[32]; int sp=0;
    for(int pc=0;pc<plen;pc++){ int o=op[pc];
      if(o==0){ if(sp<32) st[sp++]=arg[pc]; }
      else if(o==5){ if(sp>=1&&sp<32){ st[sp]=st[sp-1]; sp++; } }
      else if(sp>=2){ long b=st[--sp],a=st[--sp],r;
        switch(o){case 1:r=a+b;break;case 2:r=a*b;break;case 3:r=a-b;break;default:r=a^b;} st[sp++]=r; } }
    acc+=sp>0?st[sp-1]:0; }
  return acc; }"#,
    ),
];

const SMALL: i64 = 100;
const VERIFY_N: i64 = 777; // a fixed n for the cross-engine bit-exact correctness check

// Native driver: prints "<per_iter_ns>\n<checksum_at_VERIFY_N>".
const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
long run(long);
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
int main(int argc,char**argv){
  long small=atol(argv[1]), large=atol(argv[2]), vn=atol(argv[3]);
  volatile long sink=0; sink+=run(large);
  double bs=1e18,bl=1e18;
  for(int r=0;r<15;r++){ double a=now(); sink+=run(small); double e=now(); if(e-a<bs)bs=e-a; }
  for(int r=0;r<15;r++){ double a=now(); sink+=run(large); double e=now(); if(e-a<bl)bl=e-a; }
  printf("%.6f\n%ld\n",(bl-bs)/(double)(large-small), run(vn));
  return (int)sink; }"#;

/// Native: (per_iter_ns, checksum).
fn native(name: &str, src: &str, large: i64) -> Option<(f64, i64)> {
    let dir = std::env::temp_dir();
    let kf = dir.join(format!("e2e_{name}.c"));
    let df = dir.join(format!("e2e_{name}_drv.c"));
    let exe = dir.join(format!("e2e_{name}.exe"));
    std::fs::write(&kf, format!("#include <stdint.h>\n{src}\n")).unwrap();
    std::fs::write(&df, DRIVER).unwrap();
    let ok = Command::new("clang")
        .args(["-O2", "-march=native"])
        .args([&kf, &df])
        .arg("-o")
        .arg(&exe)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let out = Command::new(&exe)
        .args([SMALL.to_string(), large.to_string(), VERIFY_N.to_string()])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.lines();
    let ns = it.next()?.trim().parse::<f64>().ok()?;
    let chk = it.next()?.trim().parse::<i64>().ok()?;
    Some((ns, chk))
}

/// Per-iteration ns via the two-point `(T(large) − T(small)) / Δn` min-of-reps fit.
fn per_iter(large: i64, run_one: impl Fn(i64)) -> f64 {
    let m = |n: i64| {
        run_one(n);
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t = Instant::now();
            run_one(n);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        best
    };
    (m(large) - m(SMALL)) / (large - SMALL) as f64
}

fn as_i64(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("unexpected {other:?}"),
    }
}

fn main() {
    println!(
        "{:<6} {:>11} {:>11} {:>11} {:>11} | {:>7} {:>7} {:>9}  checksum",
        "prog", "native", "jit", "bytecode", "tree-walk", "jit/nat", "bc/jit", "tw/jit"
    );
    println!(
        "{:<6} {:>11} {:>11} {:>11} {:>11}",
        "", "(ns)", "(ns)", "(ns)", "(ns)"
    );
    let dir = std::env::temp_dir();
    let mut jit_ratios = Vec::new();
    for &(name, large_fast, large_bc, large_tw, src) in PROGRAMS {
        let kf = dir.join(format!("e2e_{name}.c"));
        let bc = dir.join(format!("e2e_{name}.bc"));
        std::fs::write(&kf, format!("#include <stdint.h>\n{src}\n")).unwrap();
        let ok = Command::new("clang")
            .args(["-O2", "-emit-llvm", "-c"])
            .arg(&kf)
            .arg("-o")
            .arg(&bc)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            println!("{name:<6}  (skipped: clang -emit-llvm failed)");
            continue;
        }
        let Some((nat_ns, nat_chk)) = native(name, src, large_fast) else {
            println!("{name:<6}  (skipped: native compile/run failed)");
            continue;
        };
        let t = match svm_llvm::translate_bc_path(&bc) {
            Ok(t) => t,
            Err(e) => {
                println!("{name:<6}  (skipped: translate failed: {e:?})");
                continue;
            }
        };
        let sp = t.entry_sp as i64;
        let e = t.exports.iter().find(|(s, _)| s == "run").unwrap().1;

        // Correctness: every engine must equal native at VERIFY_N (bit-exact).
        let mut fuel = u64::MAX;
        let tw_chk = as_i64(
            svm_interp::run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I64(VERIFY_N)],
                &mut fuel,
            )
            .unwrap()[0],
        );
        let mut fuel = u64::MAX;
        let bc_chk = as_i64(
            bytecode::compile_and_run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I64(VERIFY_N)],
                &mut fuel,
            )
            .expect("bytecode supports program")
            .unwrap()[0],
        );
        let jit_chk = match svm_jit::compile_and_run(&t.module, e, &[sp, VERIFY_N]).unwrap() {
            svm_jit::JitOutcome::Returned(v) => v[0],
            o => panic!("jit: {o:?}"),
        };
        let ok = nat_chk == tw_chk && nat_chk == bc_chk && nat_chk == jit_chk;
        assert!(
            ok,
            "{name}: checksum mismatch native={nat_chk} tw={tw_chk} bc={bc_chk} jit={jit_chk}"
        );

        // Timing (per-engine iteration counts; per-iter is normalized so they're comparable).
        let jit_ns = per_iter(large_fast, |n| {
            black_box(svm_jit::compile_and_run(&t.module, e, &[sp, n]).unwrap());
        });
        let bc_ns = per_iter(large_bc, |n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(
                &t.module,
                e,
                &[Value::I64(sp), Value::I64(n)],
                &mut fuel,
            );
            black_box(&r);
        });
        let tw_ns = per_iter(large_tw, |n| {
            let mut fuel = u64::MAX;
            let r = svm_interp::run(&t.module, e, &[Value::I64(sp), Value::I64(n)], &mut fuel);
            black_box(&r);
        });

        jit_ratios.push(jit_ns / nat_ns);
        println!(
            "{name:<6} {nat_ns:>11.1} {jit_ns:>11.1} {bc_ns:>11.1} {tw_ns:>11.1} | {:>6.2}x {:>6.2}x {:>8.2}x  {} {}",
            jit_ns / nat_ns,
            bc_ns / jit_ns,
            tw_ns / jit_ns,
            nat_chk,
            if ok { "OK" } else { "MISMATCH" }
        );
    }
    if !jit_ratios.is_empty() {
        let geo = (jit_ratios.iter().map(|r| r.ln()).sum::<f64>() / jit_ratios.len() as f64).exp();
        println!(
            "\nsvm-jit vs native: geomean {geo:.2}x over {} end-to-end programs (all engines bit-exact with native)",
            jit_ratios.len()
        );
    }
}
