//! Host-side **differential corpus** generator. For each guest module it (1) encodes the module to
//! its `svm-encode` binary form under `corpus/`, and (2) computes the **native** bytecode-engine
//! result for a set of args — the ground truth `corpus.mjs` checks the wasm `svm_run` against.
//!
//! The native run here uses the *exact same* `bytecode::compile_and_run` the wasm entry calls, so a
//! mismatch isolates a wasm-compilation / sandbox effect (not an engine difference). The repo
//! already gates the bytecode engine against the tree-walker oracle (`bytecode_diff.rs`); this gates
//! the *wasm build of it* against the native build.
//!
//! Status codes mirror `lib.rs`: 0 OK · 2 UNSUPPORTED (`None`) · 3 TRAP (`Err`) · 4 BAD_RESULT.

use std::io::Write;

use svm_interp::{bytecode, Value};

// Three op-family kernels lifted verbatim from `crates/svm/tests/bytecode_diff.rs` (known parseable
// and engine-supported), plus a divide-by-zero trap kernel.
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

const CALL: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br block1(v7, v10, v12)
block3(v13: i64):
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.add v0 v1
  return v2
}
"#;

const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"#;

const DIVTRAP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.div_s v0 v1
  return v2
}
"#;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× — total exactly 4000 on every interleaving.
// Lifted from `crates/svm/tests/bytecode_threads.rs`; exercises `thread.spawn`/`join` + atomics on
// the bytecode engine's cooperative `drive` (the browser concurrency model). Takes no args.
const THREADS: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

/// Args fed to each kernel (all `(i64) -> (i64)`), incl. negatives and a large value.
const ARGS: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000];

fn native(m: &svm_ir::Module, args: &[Value]) -> (i32, i64) {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, 0, args, &mut fuel) {
        None => (2, 0),
        Some(Err(_)) => (3, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (0, *x),
            _ => (4, 0),
        },
    }
}

fn main() {
    // (name, source, nargs): nargs==1 sweeps `ARGS`; nargs==0 runs once with no argument.
    let modules = [
        ("alu", ALU, 1u32),
        ("call", CALL, 1),
        ("mem", MEM, 1),
        ("divtrap", DIVTRAP, 1),
        ("threads", THREADS, 0),
    ];
    std::fs::create_dir_all("corpus").expect("mkdir corpus");

    let mut json = String::from("[\n");
    for (i, (name, src, nargs)) in modules.iter().enumerate() {
        let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
        let bytes = svm_encode::encode_module(&m);
        let file = format!("corpus/{name}.svmbc");
        std::fs::File::create(&file)
            .and_then(|mut f| f.write_all(&bytes))
            .expect("write module");

        // args sweep for 1-arg kernels; a single no-arg case otherwise.
        let args: &[i64] = if *nargs == 1 { ARGS } else { &[0] };
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":{nargs},\"cases\":["
        ));
        for (j, &arg) in args.iter().enumerate() {
            let call_args: &[Value] = if *nargs == 1 { &[Value::I64(arg)] } else { &[] };
            let (status, value) = native(&m, call_args);
            // i64s as JSON strings so JS keeps full precision (BigInt).
            json.push_str(&format!(
                "{}{{\"arg\":\"{arg}\",\"status\":{status},\"value\":\"{value}\"}}",
                if j == 0 { "" } else { "," }
            ));
        }
        json.push_str(if i + 1 == modules.len() { "]}\n" } else { "]},\n" });
        eprintln!("{name}: {} bytes, {} cases", bytes.len(), args.len());
    }
    json.push_str("]\n");
    std::fs::write("corpus.json", json).expect("write corpus.json");
    eprintln!("wrote corpus.json");
}
