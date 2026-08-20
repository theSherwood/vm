//! The `--link` CLI path (D-LINK, v9 object dialect): binary `.svmo` units carrying data
//! exports, `data.ptr` relocations, and the link-form data-address instructions link into one
//! runnable `.svmb` — the end-to-end for the wire dialect that `svm-encode` v9 added.

use std::path::PathBuf;
use std::process::Command;

use svm_ir::{
    Block, Data, DataExport, DataPtr, DataPtrTarget, Func, Inst, LoadOp, Memory, Module,
    Terminator, ValType,
};

/// Unit A: data-only — exports `g_val`, an 8-byte little-endian 7 at unit-local offset 0.
fn unit_a() -> Module {
    Module {
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: false,
            bytes: 7u64.to_le_bytes().to_vec(),
        }],
        data_exports: vec![DataExport {
            name: "g_val".into(),
            offset: 0,
        }],
        ..Default::default()
    }
}

/// Unit B: an 8-byte pointer slot whose bytes a `data.ptr` relocation fills with `&g_val`, and
/// a kernel that chases it — `data.self` → load the slot (yielding `&g_val`) → load the value.
/// Exercises all three cross-unit data mechanisms through the binary object form at once.
fn unit_b() -> Module {
    Module {
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: false,
            bytes: vec![0u8; 8], // placeholder, overwritten by the relocation
        }],
        data_ptrs: vec![DataPtr {
            at: 0,
            target: DataPtrTarget::Sym {
                name: "g_val".into(),
                addend: 0,
            },
        }],
        data_funcrefs: vec![],
        funcs: vec![Func {
            params: vec![],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![],
                insts: vec![
                    Inst::DataSelf { offset: 0 }, // v0 = &slot (this unit's data base + 0)
                    Inst::Load {
                        op: LoadOp::I64,
                        addr: 0,
                        offset: 0,
                    }, // v1 = *v0 = &g_val (written by the data.ptr relocation)
                    Inst::Load {
                        op: LoadOp::I64,
                        addr: 1,
                        offset: 0,
                    }, // v2 = *v1 = 7
                ],
                term: Terminator::Return(vec![2]),
            }],
        }],
        ..Default::default()
    }
}

fn write_unit(dir: &std::path::Path, name: &str, m: &Module) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, svm_encode::encode_unit(m)).unwrap();
    p
}

#[test]
fn link_svmo_units_into_runnable_svmb() {
    let bin = env!("CARGO_BIN_EXE_svm-run");
    let dir = std::env::temp_dir().join(format!("svm_link_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let a = write_unit(&dir, "a.svmo", &unit_a());
    let b = write_unit(&dir, "b.svmo", &unit_b());
    let out = dir.join("out.svmb");

    // Objects refuse to run directly — the error must point at --link.
    let run_obj = Command::new(bin).arg(&b).output().unwrap();
    assert!(!run_obj.status.success());
    assert!(
        String::from_utf8_lossy(&run_obj.stderr).contains("--link"),
        "direct .svmo run should point at --link"
    );

    // Link the two objects into one runnable binary.
    let link = Command::new(bin)
        .args(["--link"])
        .args([&a, &b])
        .args(["-o"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        link.status.success(),
        "link failed: {}",
        String::from_utf8_lossy(&link.stderr)
    );

    // The artifact is a flag-0 runnable with no surviving link scaffolding.
    let bytes = std::fs::read(&out).unwrap();
    let linked = svm_encode::decode_module(&bytes).expect("linked output is runnable dialect");
    assert!(linked.data_ptrs.is_empty());

    // Running it chases the relocated pointer: unit B's kernel returns unit A's 7.
    let run = Command::new(bin).arg(&out).output().unwrap();
    assert!(
        run.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains('7'),
        "expected the linked kernel to return 7, got: {stdout}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// `--assemble`: the tiny text→binary-object step for text-emitting frontends. A text unit
/// carrying link forms assembles to a `.svmo` that decodes as the identical module and links.
/// `.svmt` is the canonical text extension (no note); the deprecated `.svm` still works but
/// steers the user to `.svmt` on stderr.
#[test]
fn assemble_text_unit_to_svmo() {
    let bin = env!("CARGO_BIN_EXE_svm-run");
    let dir = std::env::temp_dir().join(format!("svm_asm_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let unit = unit_b();
    let text_path = dir.join("b.svmt");
    std::fs::write(&text_path, svm_text::print_module(&unit)).unwrap();
    let obj_path = dir.join("b.svmo");

    let asm = Command::new(bin)
        .args(["--assemble"])
        .arg(&text_path)
        .args(["-o"])
        .arg(&obj_path)
        .output()
        .unwrap();
    assert!(
        asm.status.success(),
        "assemble failed: {}",
        String::from_utf8_lossy(&asm.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&asm.stderr).contains("deprecated"),
        "canonical .svmt must not warn"
    );
    let decoded = svm_encode::decode_unit(&std::fs::read(&obj_path).unwrap()).expect("decode");
    assert_eq!(decoded, unit, "assembled object differs from the text unit");

    // Deprecated spelling: `.svm` still assembles identically but notes the rename.
    let old_path = dir.join("b.svm");
    std::fs::write(&old_path, svm_text::print_module(&unit)).unwrap();
    let obj2 = dir.join("b2.svmo");
    let asm2 = Command::new(bin)
        .args(["--assemble"])
        .arg(&old_path)
        .args(["-o"])
        .arg(&obj2)
        .output()
        .unwrap();
    assert!(asm2.status.success());
    assert!(
        String::from_utf8_lossy(&asm2.stderr).contains("deprecated"),
        "deprecated .svm should note the rename to .svmt"
    );
    assert_eq!(
        std::fs::read(&obj_path).unwrap(),
        std::fs::read(&obj2).unwrap(),
        ".svm and .svmt inputs assemble to identical objects"
    );

    std::fs::remove_dir_all(&dir).ok();
}
