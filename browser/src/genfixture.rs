//! Host-side fixture generator: parse an SVM IR text module and write its `svm-encode` binary form
//! to the path in `argv[1]`. `run.mjs` feeds the bytes to the wasm `svm_run` entry, so the wasm
//! build is exercised on the **real decode path** (not an embedded module).

use std::io::Write;

/// Same "alu" LCG recurrence as the embedded smoke kernel in `lib.rs` — so the encoded-module path
/// can be checked against the same hand-derived anchors.
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

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "alu.svmbc".into());
    let m = svm_text::parse_module(ALU).expect("parse ALU module");
    let bytes = svm_encode::encode_module(&m);
    let mut f = std::fs::File::create(&out).expect("create fixture file");
    f.write_all(&bytes).expect("write fixture");
    eprintln!("wrote {} bytes of encoded IR to {out}", bytes.len());
}
