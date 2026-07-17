//! Backend code-quality differential: how far is svm-jit from native `clang -O2` on *realistic*
//! compute kernels (not the synthetic micro-benchmarks)? For each kernel — real algorithms spanning
//! crypto, hashing, sorting, search, matmul, and float — compile the SAME C two ways: native via
//! `clang -O2` (an executable that self-times `run(n)` by large/small-n subtraction, min reps), and
//! svm-jit via `clang -O2 -emit-llvm` → `svm_llvm::translate_ll_path` → `svm_jit::compile_and_run`,
//! timed in-process the same way.
//!
//! It reports the per-kernel ratio (svm-jit / native) plus the distribution — quantifying "how many
//! optimizations like the LCG-collapse are we missing by ingesting mid-end IR + our own backend?". A
//! tight cluster near 1x with a thin tail validates the architecture; a fat tail flags real gaps.
//!
//! Run: cd crates/svm-llvm && cargo run --release --example corpus_diff

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

// (name, large_n) — small_n is fixed at 100; large_n is sized per kernel so the large run is a few ms
// (so it dominates both native process spawn and svm-jit's per-call recompile jitter).
const KERNELS: &[(&str, i64, &str)] = &[
    (
        "fnv",
        2_000_000,
        r#"
static unsigned char buf[256];
long run(long n){ for(int i=0;i<256;i++) buf[i]=(unsigned char)(i*31+7);
  unsigned h=2166136261u; for(long k=0;k<n;k++) h=(h^buf[k&255])*16777619u; return (long)h; }"#,
    ),
    (
        "crc32",
        1_000_000,
        r#"
static unsigned tab[256];
long run(long n){ for(unsigned i=0;i<256;i++){unsigned c=i;for(int j=0;j<8;j++)c=(c&1)?(0xEDB88320u^(c>>1)):(c>>1);tab[i]=c;}
  unsigned crc=0xffffffffu; for(long k=0;k<n;k++) crc=tab[(crc^(unsigned)(k&0xff))&0xff]^(crc>>8); return (long)(crc^0xffffffffu); }"#,
    ),
    (
        "xxhash",
        1_000_000,
        r#"
long run(long n){ unsigned long h=0x9E3779B185EBCA87ul;
  for(long k=0;k<n;k++){ unsigned long x=(unsigned long)k; x*=0xC2B2AE3D27D4EB4Ful; x=(x<<31)|(x>>33); x*=0x9E3779B185EBCA87ul;
    h^=x; h=(h<<27)|(h>>37); h=h*0x100000001B3ul+0x9E3779B97F4A7C15ul; } return (long)h; }"#,
    ),
    (
        "sha256_round",
        500_000,
        r#"
static unsigned ror(unsigned x,int r){return (x>>r)|(x<<(32-r));}
long run(long n){ unsigned a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,hh=0x5be0cd19;
  for(long k=0;k<n;k++){ unsigned s1=ror(e,6)^ror(e,11)^ror(e,25); unsigned ch=(e&f)^(~e&g); unsigned t1=hh+s1+ch+0x428a2f98u+(unsigned)k;
    unsigned s0=ror(a,2)^ror(a,13)^ror(a,22); unsigned maj=(a&b)^(a&c)^(b&c); unsigned t2=s0+maj;
    hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; } return (long)(a^b^c^d^e^f^g^hh); }"#,
    ),
    (
        "insertion_sort",
        30_000,
        r#"
long run(long n){ int arr[32]; long acc=0;
  for(long k=0;k<n;k++){ unsigned s=(unsigned)k*2654435761u; for(int i=0;i<32;i++){s^=s<<13;s^=s>>17;s^=s<<5;arr[i]=(int)s;}
    for(int i=1;i<32;i++){int v=arr[i],j=i-1; while(j>=0&&arr[j]>v){arr[j+1]=arr[j];j--;} arr[j+1]=v;} acc+=arr[0]+arr[31]; } return acc; }"#,
    ),
    (
        "binsearch",
        500_000,
        r#"
static int sa[1024];
long run(long n){ NOVEC for(int i=0;i<1024;i++) sa[i]=i*3; long acc=0;
  for(long k=0;k<n;k++){ int target=(int)(((unsigned)k*2654435761u)%3072); int lo=0,hi=1023,res=-1;
    while(lo<=hi){int mid=(lo+hi)>>1; if(sa[mid]==target){res=mid;break;} else if(sa[mid]<target)lo=mid+1; else hi=mid-1;} acc+=res; } return acc; }"#,
    ),
    (
        "matmul8",
        20_000,
        r#"
long run(long n){ int A[64],B[64],C[64]; long acc=0;
  for(long k=0;k<n;k++){ for(int i=0;i<64;i++){A[i]=(int)(k+i);B[i]=(int)(k-i);}
    for(int i=0;i<8;i++)for(int j=0;j<8;j++){int s=0;for(int l=0;l<8;l++)s+=A[i*8+l]*B[l*8+j];C[i*8+j]=s;} acc+=C[0]+C[63]; } return acc; }"#,
    ),
    (
        "dotprod_f64",
        100_000,
        r#"
long run(long n){ double X[64],Y[64]; long acc=0;
  for(long k=0;k<n;k++){ for(int i=0;i<64;i++){X[i]=(double)(k+i)*0.5;Y[i]=(double)(k-i)*0.25;}
    double s=0; for(int i=0;i<64;i++)s+=X[i]*Y[i]; acc+=(long)s; } return acc; }"#,
    ),
    (
        "mandelbrot",
        50_000,
        r#"
long run(long n){ long acc=0;
  for(long k=0;k<n;k++){ double cr=((double)(k%100)/50.0)-2.0, ci=((double)((k/100)%100)/50.0)-1.0; double zr=0,zi=0; int it=0;
    while(it<100 && zr*zr+zi*zi<4.0){ double t=zr*zr-zi*zi+cr; zi=2.0*zr*zi+ci; zr=t; it++; } acc+=it; } return acc; }"#,
    ),
    (
        "popcount",
        1_000_000,
        r#"
long run(long n){ long acc=0; unsigned long x=0x123456789abcdef0ul;
  for(long k=0;k<n;k++){ x^=(unsigned long)k; x*=0x2545F4914F6CDD1Dul; acc+=__builtin_popcountll(x); } return acc; }"#,
    ),
];

const SMALL: i64 = 100;

const DRIVER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
long run(long);
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
int main(int argc,char**argv){
  long small=atol(argv[1]), large=atol(argv[2]);
  volatile long sink=0; sink+=run(large);
  double bs=1e18,bl=1e18;
  for(int r=0;r<15;r++){ double a=now(); sink+=run(small); double e=now(); if(e-a<bs)bs=e-a; }
  for(int r=0;r<15;r++){ double a=now(); sink+=run(large); double e=now(); if(e-a<bl)bl=e-a; }
  printf("%.6f\n",(bl-bs)/(double)(large-small)); return (int)sink;
}"#;

fn native_ns(name: &str, src: &str, large: i64) -> Option<f64> {
    let dir = std::env::temp_dir();
    let kf = dir.join(format!("cd_{name}.c"));
    let df = dir.join(format!("cd_{name}_drv.c"));
    let exe = dir.join(format!("cd_{name}.exe"));
    std::fs::write(&kf, format!("#include <stdint.h>\n#if defined(__clang__)\n#define NOVEC _Pragma(\"clang loop vectorize(disable)\")\n#else\n#define NOVEC\n#endif\n{src}\n")).unwrap();
    std::fs::write(&df, DRIVER).unwrap();
    let ok = Command::new("clang")
        .args(["-O2", "-march=native"])
        .arg(&kf)
        .arg(&df)
        .arg("-o")
        .arg(&exe)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let out = Command::new(&exe)
        .arg(SMALL.to_string())
        .arg(large.to_string())
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .ok()
}

fn svmjit_ns(name: &str, src: &str, large: i64) -> Option<f64> {
    let dir = std::env::temp_dir();
    let kf = dir.join(format!("cd_{name}.c"));
    let bc = dir.join(format!("cd_{name}.ll"));
    std::fs::write(&kf, format!("#include <stdint.h>\n#if defined(__clang__)\n#define NOVEC _Pragma(\"clang loop vectorize(disable)\")\n#else\n#define NOVEC\n#endif\n{src}\n")).unwrap();
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&kf)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = svm_llvm::translate_ll_path(&bc).ok()?;
    let sp = t.entry_sp as i64;
    let e = t.exports.iter().find(|(n, _)| n == "run")?.1;
    // sanity: the module must actually run on the JIT (else skip)
    svm_jit::compile_and_run(&t.module, e, &[sp, SMALL]).ok()?;
    let run = |n: i64| {
        black_box(svm_jit::compile_and_run(&t.module, e, &[sp, n]).unwrap());
    };
    let m = |n: i64| {
        run(n);
        let mut best = f64::MAX;
        for _ in 0..9 {
            let t0 = Instant::now();
            run(n);
            best = best.min(t0.elapsed().as_nanos() as f64);
        }
        best
    };
    Some((m(large) - m(SMALL)) / (large - SMALL) as f64)
}

fn main() {
    println!(
        "{:<16} {:>12} {:>12} {:>8}",
        "kernel", "native(ns)", "svm-jit(ns)", "ratio"
    );
    let mut ratios = Vec::new();
    for &(name, large, src) in KERNELS {
        match (native_ns(name, src, large), svmjit_ns(name, src, large)) {
            (Some(nat), Some(jit)) => {
                let r = jit / nat;
                ratios.push((name, r));
                println!("{name:<16} {nat:>11.3} {jit:>11.3} {r:>7.2}x");
            }
            _ => println!(
                "{name:<16} {:>12} (skipped: compile/translate/run failed)",
                ""
            ),
        }
    }
    if !ratios.is_empty() {
        let geo = (ratios.iter().map(|(_, r)| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        let (wname, wr) = ratios
            .iter()
            .cloned()
            .fold(("", 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let near = ratios.iter().filter(|(_, r)| *r <= 1.5).count();
        println!("\nsummary: {} kernels | geomean ratio {geo:.2}x | within 1.5x of native: {}/{} | worst: {wname} {wr:.2}x",
            ratios.len(), near, ratios.len());
    }
}
