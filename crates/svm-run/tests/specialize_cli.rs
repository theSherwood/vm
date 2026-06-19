//! The `--specialize` CLI path (§20c): library entry (`specialize_module`) plus the binary
//! end-to-end (specialize → run, and specialize → `-o` artifact → run the artifact).

use std::process::Command;

use svm_ir::{BinOp, Block, Func, Inst, IntTy, Module, Terminator, ValType};
use svm_run::{run_kernel, specialize_module, SpecArg, SpecializeOpts, Value};

/// `g(a, b) = a * 2 + b`.
fn g() -> Module {
    Module {
        funcs: vec![Func {
            params: vec![ValType::I64, ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64, ValType::I64], // 0: a, 1: b
                insts: vec![
                    Inst::ConstI64(2), // 2
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Mul,
                        a: 0,
                        b: 2,
                    }, // 3: a * 2
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Add,
                        a: 3,
                        b: 1,
                    }, // 4: + b
                ],
                term: Terminator::Return(vec![4]),
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn specialize_module_binds_a_static_arg() {
    // a = 5 (static), b dynamic -> residual(b) = 10 + b, a single-parameter function.
    let opts = SpecializeOpts {
        args: vec![SpecArg::ConstI64(5), SpecArg::Dynamic],
        optimize: true,
        ..SpecializeOpts::default()
    };
    let r = specialize_module(&g(), &opts).expect("specializes");
    assert_eq!(r.funcs[0].params.len(), 1, "only the dynamic arg remains");
    assert_eq!(run_kernel(&r, &[3]).unwrap(), vec![Value::I64(13)]);
    assert_eq!(run_kernel(&r, &[-10]).unwrap(), vec![Value::I64(0)]);
}

#[test]
fn specialize_module_missing_bindings_default_to_dynamic() {
    // No --arg given: every parameter is dynamic, so the residual matches the original.
    let r = specialize_module(&g(), &SpecializeOpts::default()).expect("specializes");
    assert_eq!(r.funcs[0].params.len(), 2);
    assert_eq!(run_kernel(&r, &[4, 1]).unwrap(), vec![Value::I64(9)]);
}

#[test]
fn cli_specialize_runs_and_emits_artifact() {
    let bin = env!("CARGO_BIN_EXE_svm-run");
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let src = dir.join(format!("peval_cli_{pid}.svmb"));
    std::fs::write(&src, svm_encode::encode_module(&g())).unwrap();

    // Specialize a=5, run with b=3 -> 10 + 3 = 13.
    let run = Command::new(bin)
        .arg(&src)
        .args(["--specialize", "--arg", "i64:5", "--run-args", "3"])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "[I64(13)]");

    // Emit the residual as a binary artifact, then run the artifact as a bare kernel (b defaults
    // to 0 -> 10).
    let art = dir.join(format!("peval_cli_{pid}_out.svmb"));
    let emit = Command::new(bin)
        .arg(&src)
        .args(["--specialize", "--arg", "i64:5", "-o"])
        .arg(&art)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(art.exists(), "artifact was written");

    let run_art = Command::new(bin).arg(&art).output().unwrap();
    assert!(
        run_art.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run_art.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run_art.stdout).trim(), "[I64(10)]");

    // --emit-text prints text IR.
    let text = Command::new(bin)
        .arg(&src)
        .args(["--specialize", "--arg", "i64:5", "--emit-text"])
        .output()
        .unwrap();
    assert!(text.status.success());
    assert!(String::from_utf8_lossy(&text.stdout).contains("func"));

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&art);
}
