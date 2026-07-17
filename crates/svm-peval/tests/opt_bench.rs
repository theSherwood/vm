//! **Optimizer ablation harness (OPT.md Phase 2(d) / Phase 5).** Measures how much each individual
//! svm-opt pass buys us, by *leave-one-out*: run the full pipeline, then the full pipeline with one
//! pass disabled, and attribute the difference to that pass. Two views:
//!
//!   * `opt_size_ablation` (cheap, assertion-backed — **not** `#[ignore]`, so it also guards against
//!     size regressions): for a corpus of realistic + pass-targeted modules, report the encoded-byte /
//!     instruction / block size under `none` (baseline canonicalization only), `all` (full pipeline),
//!     and `all − pass` for each pass. The `all − pass` minus `all` delta is what that pass removes.
//!   * `opt_runtime_ablation` (`#[ignore]` — perf): for the one module with a genuine runtime loop (a
//!     specialized register-machine interpreter), JIT compile-once/run-many under `none` / `all` /
//!     `all − pass` and print the run-time each pass is worth.
//!
//! Every ablation variant is re-verified and differential-tested against the unoptimized input on the
//! reference interpreter, so this doubles as a correctness check over the whole toggle space.
//!
//!   cargo test -p svm-peval --test opt_bench -- --nocapture              # size table (+ guards)
//!   cargo test -p svm-peval --test opt_bench -- --ignored --nocapture    # + run-time table
//!
//! Set `SVM_BENCH_CSV=1` to also emit `CSV,<bench>,<case>,<metric>,<value>` rows for the report script.

use std::hint::black_box;
use std::time::{Duration, Instant};

use svm_interp::Value;
use svm_ir::{
    BinOp, Block, CmpOp, Data, Func, FuncType, Inst, IntTy, LoadOp, Memory, Module, Terminator,
    ValType, DEFAULT_RESERVED_LOG2,
};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_peval::{optimize_module_with, specialize, OptConfig, SpecArg};
use svm_verify::verify_module;

// ---------------------------------------------------------------------------------------
// Sizes + CSV (same shape as tests/bench.rs, kept local so this file stands alone).
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Sizes {
    insts: usize,
    bytes: usize,
}

fn sizes(m: &Module) -> Sizes {
    Sizes {
        insts: m
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| b.insts.len())
            .sum(),
        bytes: svm_encode::encode_module(m).len(),
    }
}

fn csv(bench: &str, case: &str, metric: &str, value: f64) {
    if std::env::var_os("SVM_BENCH_CSV").is_some() {
        // To stderr, so the human-readable tables on stdout stay clean under `--nocapture`.
        eprintln!("CSV,{bench},{case},{metric},{value}");
    }
}

/// The optional passes, each as `all()` with exactly one turned off — the leave-one-out configs. The
/// `all − pass` output minus the `all` output is the size/time that pass is responsible for.
fn leave_one_out() -> Vec<(&'static str, OptConfig)> {
    let a = OptConfig::all();
    vec![
        ("sccp", OptConfig { sccp: false, ..a }),
        (
            "reassociate",
            OptConfig {
                reassociate: false,
                ..a
            },
        ),
        ("gvn", OptConfig { gvn: false, ..a }),
        ("licm", OptConfig { licm: false, ..a }),
        (
            "local_cse",
            OptConfig {
                local_cse: false,
                ..a
            },
        ),
        (
            "jump_thread",
            OptConfig {
                jump_thread: false,
                ..a
            },
        ),
        ("devirt", OptConfig { devirt: false, ..a }),
        ("inline", OptConfig { inline: false, ..a }),
        ("dfe", OptConfig { dfe: false, ..a }),
        ("mem", OptConfig { mem: false, ..a }),
        (
            "load_elim",
            OptConfig {
                load_elim: false,
                ..a
            },
        ),
    ]
}

fn interp_run(m: &Module, args: &[Value]) -> Result<Vec<Value>, svm_interp::Trap> {
    let mut fuel = 50_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

// ---------------------------------------------------------------------------------------
// Corpus. Each case is an *unoptimized* module (a good optimizer input) plus reference args. The mix
// spans a realistic specialization residual (all passes fire) and pass-targeted micro-modules so every
// pass has at least one case where removing it visibly hurts.
// ---------------------------------------------------------------------------------------

struct Case {
    name: &'static str,
    module: Module,
    args: Vec<Vec<Value>>,
}

fn i32s(xs: &[i32]) -> Vec<Value> {
    xs.iter().map(|&x| Value::I32(x)).collect()
}
fn i64s(xs: &[i64]) -> Vec<Value> {
    xs.iter().map(|&x| Value::I64(x)).collect()
}

fn bin32(op: BinOp, a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op,
        a,
        b,
    }
}

// ---- realistic: a specialized register-machine sum-loop interpreter (shape shared with the ROI
// bench in tests/bench.rs). The residual is a genuinely compiled runtime loop with dispatch folded
// away — the meatiest optimizer input we have. ----

const HALT: u8 = 0;
const SETACC: u8 = 1;
const SETI_INPUT: u8 = 2;
const ADD_I: u8 = 3;
const DEC_I: u8 = 4;
const JNZ: u8 = 5;

fn encode_program(program: &[(u8, i64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for &(op, imm) in program {
        bytes.push(op);
        bytes.extend_from_slice(&imm.to_le_bytes());
    }
    bytes
}

fn build_interpreter(program: &[(u8, i64)]) -> Module {
    let t = || ValType::I64;
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let load = |op, addr, offset| Inst::Load {
        op,
        addr,
        offset,
        align: 0,
    };

    let entry = Block {
        params: vec![t()],
        insts: vec![Inst::ConstI64(0), Inst::ConstI64(0), Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![1, 2, 3, 0],
        },
    };
    let header = Block {
        params: vec![t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(0),
            add(4, 2),
            load(LoadOp::I32_8U, 5, 0),
            load(LoadOp::I64, 5, 1),
        ],
        term: Terminator::BrTable {
            idx: 6,
            targets: vec![
                (2, vec![0]),
                (3, vec![0, 1, 2, 7, 3]),
                (4, vec![0, 1, 2, 7, 3]),
                (5, vec![0, 1, 2, 7, 3]),
                (6, vec![0, 1, 2, 7, 3]),
                (7, vec![0, 1, 2, 7, 3]),
            ],
            default: (2, vec![0]),
        },
    };
    let halt = Block {
        params: vec![t()],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    };
    let setacc = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![Inst::ConstI64(9), add(2, 5)],
        term: Terminator::Br {
            target: 1,
            args: vec![3, 1, 6, 4],
        },
    };
    let seti_input = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![Inst::ConstI64(9), add(2, 5)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 4, 6, 4],
        },
    };
    let add_i = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![add(0, 1), Inst::ConstI64(9), add(2, 6)],
        term: Terminator::Br {
            target: 1,
            args: vec![5, 1, 7, 4],
        },
    };
    let dec_i = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(1),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 1,
                b: 5,
            },
            Inst::ConstI64(9),
            add(2, 7),
        ],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 6, 8, 4],
        },
    };
    let jnz = Block {
        params: vec![t(), t(), t(), t(), t()],
        insts: vec![
            Inst::ConstI64(0),
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Ne,
                a: 1,
                b: 5,
            },
            Inst::ConstI64(9),
            add(2, 7),
        ],
        term: Terminator::BrIf {
            cond: 6,
            then_blk: 1,
            then_args: vec![0, 1, 3, 4],
            else_blk: 1,
            else_args: vec![0, 1, 8, 4],
        },
    };

    Module {
        funcs: vec![Func {
            params: vec![t()],
            results: vec![t()],
            blocks: vec![entry, header, halt, setacc, seti_input, add_i, dec_i, jnz],
        }],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: true,
            bytes: encode_program(program),
        }],
        ..Default::default()
    }
}

fn sum_program() -> [(u8, i64); 6] {
    [
        (SETACC, 0),
        (SETI_INPUT, 0),
        (ADD_I, 0),
        (DEC_I, 0),
        (JNZ, 18),
        (HALT, 0),
    ]
}

/// The realistic residual: specialize the sum-loop interpreter against its program, input dynamic.
fn reg_sum_residual() -> Module {
    let interp = build_interpreter(&sum_program());
    verify_module(&interp).expect("interp verifies");
    let residual = specialize(&interp, 0, &[SpecArg::Dynamic]).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    residual
}

/// A counted loop whose body recomputes an invariant `a*b` twice and sums it — targets LICM (hoist the
/// invariant), GVN/local-CSE (the two `a*b` are congruent), and the cleanup fixpoint.
///   b0(n,a,b): br b1(n, 0, a, b)
///   b1(i,acc,a,b): t1=a*b; t2=a*b; s=t1+t2; acc2=acc+s; i2=i-1; brif i2 → b1(i2,acc2,a,b) else b2(acc2)
///   b2(r): return r
fn licm_cse_kernel() -> Module {
    let f = Func {
        params: vec![ValType::I32, ValType::I32, ValType::I32], // n, a, b
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32, ValType::I32, ValType::I32],
                insts: vec![Inst::ConstI32(0)], // v3 = acc0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 3, 1, 2],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32], // i,acc,a,b
                insts: vec![
                    bin32(BinOp::Mul, 2, 3), // v4 = a*b
                    bin32(BinOp::Mul, 2, 3), // v5 = a*b (redundant)
                    bin32(BinOp::Add, 4, 5), // v6 = t1+t2
                    bin32(BinOp::Add, 1, 6), // v7 = acc + s
                    Inst::ConstI32(1),       // v8
                    bin32(BinOp::Sub, 0, 8), // v9 = i-1
                ],
                term: Terminator::BrIf {
                    cond: 9,
                    then_blk: 1,
                    then_args: vec![9, 7, 2, 3],
                    else_blk: 2,
                    else_args: vec![7],
                },
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    module1(f, None)
}

/// A loop whose accumulator is provably the constant 0 around the back edge (entry passes 0, the
/// back edge passes it unchanged) — SCCP resolves `acc == 0` to 1 through the phi; plain block-local
/// folding cannot, since `acc` is a parameter. The folded compare then feeds the returned value.
///   b0(n): br b1(n, 0)
///   b1(i,acc): c = (acc == 0); i2 = i-1; brif i2 → b1(i2, acc) else b2(acc + c)
///   b2(r): return r
fn sccp_const_loop() -> Module {
    let f = Func {
        params: vec![ValType::I32], // n
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![Inst::ConstI32(0)], // v1
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32], // i, acc
                insts: vec![
                    Inst::ConstI32(0), // v2
                    Inst::IntCmp {
                        ty: IntTy::I32,
                        op: CmpOp::Eq,
                        a: 1,
                        b: 2,
                    }, // v3 = (acc == 0)  -> SCCP: always 1
                    Inst::ConstI32(1), // v4
                    bin32(BinOp::Sub, 0, 4), // v5 = i - 1
                    bin32(BinOp::Add, 1, 3), // v6 = acc + c
                ],
                term: Terminator::BrIf {
                    cond: 5,
                    then_blk: 1,
                    then_args: vec![5, 1], // loop with (i-1, acc)
                    else_blk: 2,
                    else_args: vec![6], // exit with acc + c
                },
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    module1(f, None)
}

/// A straight-line constant chain `((((x + 1) + 2) + 3) + 4)` — reassociation collapses the constant
/// tail to a single `x + 10`, then the fixpoint DCEs the intermediate adds.
fn reassoc_chain() -> Module {
    let f = Func {
        params: vec![ValType::I32], // x = v0
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I32],
            insts: vec![
                Inst::ConstI32(1),       // v1
                bin32(BinOp::Add, 0, 1), // v2 = x+1
                Inst::ConstI32(2),       // v3
                bin32(BinOp::Add, 2, 3), // v4
                Inst::ConstI32(3),       // v5
                bin32(BinOp::Add, 4, 5), // v6
                Inst::ConstI32(4),       // v7
                bin32(BinOp::Add, 6, 7), // v8
            ],
            term: Terminator::Return(vec![8]),
        }],
    };
    module1(f, None)
}

/// A correlated branch through an empty forwarder — targets jump threading (the flag passed into the
/// forwarder is a different constant on each edge, so SCCP cannot resolve it).
///   b0(x): z=(x==5); brif z → b1(x,1) else b1(x,0)
///   b1(x,flag): brif flag → b2(x) else b3(x)
///   b2(y): return y+100 ; b3(y): return y+200
fn correlated_branch() -> Module {
    let f = Func {
        params: vec![ValType::I32],
        results: vec![ValType::I32],
        blocks: vec![
            Block {
                params: vec![ValType::I32],
                insts: vec![
                    Inst::ConstI32(5),
                    Inst::IntCmp {
                        ty: IntTy::I32,
                        op: CmpOp::Eq,
                        a: 0,
                        b: 1,
                    },
                    Inst::ConstI32(1),
                    Inst::ConstI32(0),
                ],
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![0, 3],
                    else_blk: 1,
                    else_args: vec![0, 4],
                },
            },
            Block {
                params: vec![ValType::I32, ValType::I32],
                insts: vec![],
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 2,
                    then_args: vec![0],
                    else_blk: 3,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![Inst::ConstI32(100), bin32(BinOp::Add, 0, 1)],
                term: Terminator::Return(vec![2]),
            },
            Block {
                params: vec![ValType::I32],
                insts: vec![Inst::ConstI32(200), bin32(BinOp::Add, 0, 1)],
                term: Terminator::Return(vec![2]),
            },
        ],
    };
    module1(f, None)
}

fn module1(f: Func, memory: Option<Memory>) -> Module {
    Module {
        funcs: vec![f],
        memory,
        ..Default::default()
    }
}

/// A memory-access shape that exercises the Phase-4 passes: a redundant same-address load
/// (intra-block `mem`) in the head, and a **join reload** across a diamond that only `load_elim` can
/// remove (a multi-predecessor block cannot be merged, so intra-block forwarding never sees it).
///   b0(p,sel): a = mem[p+0]; b = mem[p+0]; br_if sel -> b1(p, a+b) else b2(p, a+b)   // b redundant → mem
///   b1/b2: pass (p, s) through to b3
///   b3(p3, s): d = mem[p3+0]; return s + d                                            // d → load_elim
fn mem_case() -> Module {
    let ld = |addr: u32, off: u64| Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: off,
        align: 0,
    };
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let thru = |t: u32| Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![],
        term: Terminator::Br {
            target: t,
            args: vec![0, 1],
        },
    };
    let f = Func {
        params: vec![ValType::I64, ValType::I32], // p, sel
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I32], // v0=p, v1=sel
                insts: vec![
                    ld(0, 0),  // v2 = mem[p+0]
                    ld(0, 0),  // v3 = mem[p+0]  (redundant → mem)
                    add(2, 3), // v4 = a + b
                ],
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 1,
                    then_args: vec![0, 4],
                    else_blk: 2,
                    else_args: vec![0, 4],
                },
            },
            thru(3), // b1 -> b3
            thru(3), // b2 -> b3
            Block {
                params: vec![ValType::I64, ValType::I64], // p3, s
                insts: vec![
                    ld(0, 0),  // v2 = mem[p3+0]  (join reload → load_elim)
                    add(1, 2), // v3 = s + d
                ],
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: false,
            bytes: 5i64.to_le_bytes().to_vec(),
        }],
        ..Default::default()
    }
}

/// An interprocedural shape: the entry `call_indirect`s a constant funcref, so **devirt** turns it
/// into a direct call, **inline** splices the small leaf in, and **dfe** then removes both the leaf
/// and an unused third function.
///   f0(a,b): call_indirect(ref.func(1), [a,b])       // -> devirt -> inline
///   f1(a,b): a*3 + b*5 + 7                            // leaf (inlined, then dead)
///   f2(a,b): dead (uncalled)
fn interproc_case() -> Module {
    let mul = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Mul,
        a,
        b,
    };
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let sig = FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    };
    let entry = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts: vec![
                Inst::RefFunc { func: 1 }, // v2 = funcref(1)
                Inst::CallIndirect {
                    ty: sig,
                    idx: 2,
                    args: vec![0, 1],
                }, // v3
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    let leaf = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts: vec![
                Inst::ConstI64(3),
                mul(0, 2), // a*3
                Inst::ConstI64(5),
                mul(1, 4), // b*5
                add(3, 5),
                Inst::ConstI64(7),
                add(6, 7), // a*3 + b*5 + 7
            ],
            term: Terminator::Return(vec![8]),
        }],
    };
    let dead = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts: vec![add(0, 1)],
            term: Terminator::Return(vec![2]),
        }],
    };
    Module {
        funcs: vec![entry, leaf, dead],
        ..Default::default()
    }
}

/// A caller with a **multi-block** callee (internal control flow), so the inliner must splice the
/// callee's CFG in and thread a captured value across the call — the shape single-block inlining can't
/// touch. `entry(a,b): k=a+b; t=max(a,b); return t+k`; `max` is a three-block branch (two returns).
/// After inline + DFE the whole thing collapses into one function with no call.
fn multiblock_inline_case() -> Module {
    let add = |a, b| Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    };
    let entry = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64], // a, b
            insts: vec![
                add(0, 1), // v2 = k = a + b  (captured across the call)
                Inst::Call {
                    func: 1,
                    args: vec![0, 1],
                }, // v3 = max(a, b)
                add(3, 2), // v4 = t + k
            ],
            term: Terminator::Return(vec![4]),
        }],
    };
    // max(a, b): b0 tests a<b and branches; b1 returns b; b2 returns a.
    let max = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I64], // a, b
                insts: vec![Inst::IntCmp {
                    ty: IntTy::I64,
                    op: CmpOp::LtS,
                    a: 0,
                    b: 1,
                }], // v2 = a < b
                term: Terminator::BrIf {
                    cond: 2,
                    then_blk: 1,
                    then_args: vec![1],
                    else_blk: 2,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![ValType::I64], // b
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
            Block {
                params: vec![ValType::I64], // a
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    Module {
        funcs: vec![entry, max],
        ..Default::default()
    }
}

fn corpus() -> Vec<Case> {
    vec![
        Case {
            name: "reg-sum residual (loop, all passes)",
            module: reg_sum_residual(),
            args: vec![
                vec![Value::I64(1)],
                vec![Value::I64(7)],
                vec![Value::I64(64)],
            ],
        },
        Case {
            name: "licm+cse kernel",
            module: licm_cse_kernel(),
            args: vec![i32s(&[1, 3, 4]), i32s(&[5, -2, 7]), i32s(&[8, 0, 9])],
        },
        Case {
            name: "sccp const-loop",
            module: sccp_const_loop(),
            args: vec![i32s(&[1]), i32s(&[4]), i32s(&[9])],
        },
        Case {
            name: "reassoc chain",
            module: reassoc_chain(),
            args: vec![i32s(&[0]), i32s(&[10]), i32s(&[-7])],
        },
        Case {
            name: "correlated branch",
            module: correlated_branch(),
            args: vec![i32s(&[-3]), i32s(&[4]), i32s(&[5]), i32s(&[100])],
        },
        Case {
            name: "memory (mem + load_elim)",
            module: mem_case(),
            args: vec![
                vec![Value::I64(0), Value::I32(1)],
                vec![Value::I64(0), Value::I32(0)],
            ],
        },
        Case {
            name: "interproc (devirt+inline+dfe)",
            module: interproc_case(),
            args: vec![i64s(&[3, 4]), i64s(&[-2, 7]), i64s(&[10, 10])],
        },
        Case {
            name: "multiblock inline (inline+dfe)",
            module: multiblock_inline_case(),
            args: vec![i64s(&[3, 4]), i64s(&[7, -2]), i64s(&[10, 10])],
        },
    ]
}

/// Optimize under `cfg`, re-verify, and assert behavior is preserved against the unoptimized input on
/// every reference-arg tuple. Returns the optimized module's sizes.
fn check_and_size(case: &Case, cfg: &OptConfig, label: &str) -> Sizes {
    let opt = optimize_module_with(&case.module, cfg);
    verify_module(&opt)
        .unwrap_or_else(|e| panic!("{}/{label}: re-verify failed: {e:?}", case.name));
    for args in &case.args {
        assert_eq!(
            interp_run(&case.module, args),
            interp_run(&opt, args),
            "{}/{label}: divergence at args {args:?}",
            case.name
        );
    }
    sizes(&opt)
}

#[test]
fn opt_size_ablation() {
    let cases = corpus();
    println!("\n=== optimizer size ablation (encoded bytes; leave-one-out) ===");
    println!(
        "For each case: input → none (canonicalization only) → all (full pipeline), as insts/bytes.\n\
         Each pass column is the *byte delta if that pass is removed* from the full pipeline:\n\
           +N  the output is N bytes larger without the pass — it shrinks code by N (what it buys).\n\
           -N  the output is N bytes *smaller* without the pass — the pass grows static size (e.g.\n\
               LICM/GVN hoist or thread values through new block params, a size cost paid for a\n\
               run-time win; see opt_runtime_ablation).\n"
    );
    let passes = leave_one_out();
    print!(
        "{:<38} {:>9} {:>9} {:>9} | ",
        "case", "input i/b", "none i/b", "all i/b"
    );
    for (name, _) in &passes {
        print!("{name:>11} ");
    }
    println!();

    for case in &cases {
        let input = sizes(&case.module);
        let none = check_and_size(case, &OptConfig::none(), "none");
        let full = check_and_size(case, &OptConfig::all(), "all");

        let ib = |s: Sizes| format!("{}/{}", s.insts, s.bytes);
        print!(
            "{:<38} {:>9} {:>9} {:>9} | ",
            case.name,
            ib(input),
            ib(none),
            ib(full)
        );
        csv(case.name, "size", "input_bytes", input.bytes as f64);
        csv(case.name, "size", "none_bytes", none.bytes as f64);
        csv(case.name, "size", "all_bytes", full.bytes as f64);
        csv(case.name, "size", "all_insts", full.insts as f64);
        for (name, cfg) in &passes {
            let without = check_and_size(case, cfg, name);
            let delta = without.bytes as i64 - full.bytes as i64;
            print!("{:>+11} ", delta);
            csv(case.name, name, "delta_bytes", delta as f64);
        }
        println!();
    }
    println!(
        "\n(A realistic residual leans on several passes at once; the micro-cases isolate one. The\n\
         correctness of every variant is asserted against the interpreter — this table is also a guard.)"
    );
}

// ---------------------------------------------------------------------------------------
// Run-time ablation: the register-machine sum loop is the only case with a genuine runtime loop, so
// it is where a pass changes *execution* time, not just code size. Compile-once/run-many via the JIT.
// ---------------------------------------------------------------------------------------

fn jit_compile(m: &Module) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compile")
}

fn jit_call(cm: &mut CompiledModule, args: &[i64]) -> i64 {
    match cm.run(args, None, None, None) {
        Ok((JitOutcome::Returned(v), _)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

fn best_of(reps: usize, mut f: impl FnMut() -> i64) -> Duration {
    f(); // warm up
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        let r = f();
        best = best.min(start.elapsed());
        black_box(r);
    }
    best
}

/// A counted loop (`i64`) whose body recomputes a *heavy, loop-invariant* expression of the runtime
/// parameters `a`/`b` — `inv = (a*b + a) * (b + 7) + a*b` — and accumulates it `n` times. The `a`/`b`
/// are runtime params, so nothing folds the chain at compile time; only **LICM** can move the five
/// invariant ops out of the loop (GVN/CSE additionally dedupe the two `a*b`). This is where a pass
/// changes *run time*, not just code size — without LICM the JIT re-runs the whole chain every trip.
fn heavy_invariant_loop() -> Module {
    let bin = |op, a, b| Inst::IntBin {
        ty: IntTy::I64,
        op,
        a,
        b,
    };
    let f = Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64], // n, a, b
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I64, ValType::I64],
                insts: vec![Inst::ConstI64(0)], // v3 = acc0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 3, 1, 2], // b1(i=n, acc=0, a, b)
                },
            },
            Block {
                params: vec![ValType::I64, ValType::I64, ValType::I64, ValType::I64], // i,acc,a,b
                insts: vec![
                    bin(BinOp::Mul, 2, 3),  // v4 = a*b
                    bin(BinOp::Add, 4, 2),  // v5 = a*b + a
                    Inst::ConstI64(7),      // v6
                    bin(BinOp::Add, 3, 6),  // v7 = b + 7
                    bin(BinOp::Mul, 5, 7),  // v8 = (a*b+a)*(b+7)
                    bin(BinOp::Mul, 2, 3),  // v9 = a*b (redundant — GVN/CSE)
                    bin(BinOp::Add, 8, 9),  // v10 = inv
                    bin(BinOp::Add, 1, 10), // v11 = acc + inv
                    Inst::ConstI64(1),      // v12
                    bin(BinOp::Sub, 0, 12), // v13 = i - 1
                    Inst::ConstI64(0),      // v14
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::Ne,
                        a: 13,
                        b: 14,
                    }, // v15 = (i-1) != 0  (i32 cond)
                ],
                term: Terminator::BrIf {
                    cond: 15,
                    then_blk: 1,
                    then_args: vec![13, 11, 2, 3],
                    else_blk: 2,
                    else_args: vec![11],
                },
            },
            Block {
                params: vec![ValType::I64],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    module1(f, None)
}

/// One run-time ablation case: an unoptimized input module, the JIT/interp args to drive it, and a
/// human label for the workload.
struct RtCase {
    name: &'static str,
    module: Module,
    jit_args: Vec<i64>,
    interp_args: Vec<Value>,
}

fn i64_result(vs: &[Value]) -> i64 {
    match vs {
        [Value::I64(x)] => *x,
        o => panic!("expected one i64 result, got {o:?}"),
    }
}

fn interp_i64(m: &Module, args: &[Value]) -> i64 {
    i64_result(&interp_run(m, args).expect("interp ok"))
}

#[test]
#[ignore = "perf benchmark — run with --ignored --nocapture"]
fn opt_runtime_ablation() {
    let reps = 5;
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    let n: i64 = 1_000_000;
    let cases = [
        RtCase {
            name: "reg-machine sum 1..=N (already-tight loop)",
            module: reg_sum_residual(),
            jit_args: vec![n],
            interp_args: vec![Value::I64(n)],
        },
        RtCase {
            name: "heavy-invariant loop (LICM showcase)",
            module: heavy_invariant_loop(),
            jit_args: vec![n, 3, 5],
            interp_args: vec![Value::I64(n), Value::I64(3), Value::I64(5)],
        },
    ];

    let variants: Vec<(&str, OptConfig)> = {
        let mut v = vec![("none", OptConfig::none()), ("all", OptConfig::all())];
        v.extend(leave_one_out());
        v
    };

    println!(
        "\nTwo backends, because they answer different questions. The **JIT** runs its own optimizer\n\
         over the IR, so svm-opt's scalar passes are largely redundant there (native run time barely\n\
         moves). The reference **interpreter** executes the IR as-is, so removing a pass that trims\n\
         per-iteration work (LICM, CSE) shows a real run-time delta — that is the run-time value\n\
         svm-opt adds on the interp path (and inside svm, DESIGN.md §20c)."
    );

    for case in &cases {
        verify_module(&case.module).expect("input module verifies");
        let expect = interp_i64(&case.module, &case.interp_args);
        println!(
            "\n=== optimizer run-time ablation: {} (N={n}) ===",
            case.name
        );
        println!(
            "{:<14} {:>7} {:>10} {:>9} {:>11} {:>9}",
            "variant", "bytes", "compile_ms", "jit_ms", "interp_ms", "interp/all"
        );

        // Baseline `all` interpreter run time for the "interp/all" column — the pass-attribution axis.
        let all_interp = {
            let opt = optimize_module_with(&case.module, &OptConfig::all());
            assert_eq!(interp_i64(&opt, &case.interp_args), expect);
            ms(best_of(reps, || interp_i64(&opt, &case.interp_args)))
        };

        for (name, cfg) in &variants {
            let opt = optimize_module_with(&case.module, cfg);
            verify_module(&opt).expect("re-verify");
            let bytes = sizes(&opt).bytes;
            let t_compile = ms(best_of(reps, || {
                black_box(jit_compile(&opt));
                0
            }));
            let mut cm = jit_compile(&opt);
            assert_eq!(
                jit_call(&mut cm, &case.jit_args),
                expect,
                "{name}: wrong jit result"
            );
            let t_jit = ms(best_of(reps, || jit_call(&mut cm, &case.jit_args)));
            let t_interp = ms(best_of(reps, || interp_i64(&opt, &case.interp_args)));
            println!(
                "{:<14} {:>7} {:>10.3} {:>9.3} {:>11.3} {:>8.2}x",
                name,
                bytes,
                t_compile,
                t_jit,
                t_interp,
                t_interp / all_interp
            );
            csv(case.name, name, "run_bytes", bytes as f64);
            csv(case.name, name, "compile_ms", t_compile);
            csv(case.name, name, "jit_ms", t_jit);
            csv(case.name, name, "interp_ms", t_interp);
        }
    }
    println!(
        "\n(interp/all > 1 when a pass is removed = that pass matters for interpreter run time. On the\n\
         heavy-invariant loop, removing LICM re-runs the invariant chain every iteration; `none` vs\n\
         `all` on interp_ms is the whole optimizer's interpreter run-time win.)"
    );
}
