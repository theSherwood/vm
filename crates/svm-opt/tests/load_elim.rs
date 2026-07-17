//! Spec for cross-block redundant-load elimination (OPT.md Phase 4, `svm_opt::load_elim`). A load
//! whose location a *dominating* access already established, with no memory write on any path between,
//! must be removed and forwarded — and, crucially, must **not** be when a clobber sits between or a
//! loop encloses them. Every case is differential-tested against the reference interpreter, including
//! aliasing/looping inputs that would expose an unsound forward.

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, CmpOp, Data, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator,
    ValType,
};
use svm_opt::{optimize_module, optimize_module_with, OptConfig};
use svm_verify::verify_module;

/// A module with linear memory pre-initialized so `mem[0..8]` holds `v` (little-endian i64), so a load
/// from address 0 reads a known nonzero value without needing a store.
fn module(f: Func, v: i64) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        data: vec![Data {
            offset: 0,
            readonly: false,
            bytes: v.to_le_bytes().to_vec(),
        }],
        ..Default::default()
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

fn loadi(addr: u32) -> Inst {
    Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: 0,
        align: 0,
    }
}
fn storei(addr: u32, value: u32) -> Inst {
    Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    }
}
fn store_off(op: StoreOp, addr: u32, value: u32, offset: u64) -> Inst {
    Inst::Store {
        op,
        addr,
        value,
        offset,
        align: 0,
    }
}
fn addi(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    }
}
fn subi(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a,
        b,
    }
}

fn n_loads(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Load { .. }))
        .count()
}

/// Just the cross-block load pass over the always-on canonicalization — nothing else — so an
/// eliminated load is attributable to `load_elim` (not intra-block `mem_forward` or GVN).
fn only_load_elim() -> OptConfig {
    OptConfig {
        load_elim: true,
        ..OptConfig::none()
    }
}

fn check(m: &Module, args: &[Vec<Value>]) {
    verify_module(m).expect("input verifies");
    let opt = optimize_module(m);
    verify_module(&opt).expect("optimized re-verifies");
    let iso = optimize_module_with(m, &only_load_elim());
    verify_module(&iso).expect("load_elim-only re-verifies");
    for a in args {
        assert_eq!(run(m, a), run(&opt, a), "full pipeline divergence at {a:?}");
        assert_eq!(
            run(m, a),
            run(&iso, a),
            "load_elim-only divergence at {a:?}"
        );
    }
}

#[test]
fn sequential_reload_is_forwarded() {
    // b0(addr): x = mem[addr]; br b1(addr, x)
    // b1(addr2, x2): y = mem[addr2]; return x2 + y     // y redundant with b0's load
    let f = Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64], // v0 = addr
                insts: vec![loadi(0)],      // v1 = x
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1],
                },
            },
            Block {
                params: vec![ValType::I64, ValType::I64], // v0 = addr2, v1 = x2
                insts: vec![loadi(0), addi(1, 2)],        // v2 = y, v3 = x2 + y
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    let m = module(f, 42);
    check(&m, &[vec![Value::I64(0)]]);
    // Isolated: b1's reload is gone (b0's load remains, since intra-block forwarding is off here).
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(n_loads(&iso), 1, "the dominated reload is eliminated");
    assert_eq!(run(&iso, &[Value::I64(0)]), Ok(vec![Value::I64(84)]));
}

#[test]
fn diamond_join_reload_is_forwarded() {
    // b0(addr,sel): x = mem[addr]; br_if sel -> b1(addr,x) else b2(addr,x)
    // b1 / b2: pass through to b3
    // b3(addr3, x3): y = mem[addr3]; return x3 + y     // y redundant — b0 dominates b3, no clobber
    let thru = |t: u32| Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![],
        term: Terminator::Br {
            target: t,
            args: vec![0, 1],
        },
    };
    let f = Func {
        params: vec![ValType::I64, ValType::I32], // addr, sel (br_if cond is i32)
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I32], // v0=addr, v1=sel
                insts: vec![loadi(0)],                    // v2 = x
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 1,
                    then_args: vec![0, 2],
                    else_blk: 2,
                    else_args: vec![0, 2],
                },
            },
            thru(3), // b1 -> b3
            thru(3), // b2 -> b3
            Block {
                params: vec![ValType::I64, ValType::I64], // v0=addr3, v1=x3
                insts: vec![loadi(0), addi(1, 2)],        // v2 = y, v3 = x3 + y
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    let m = module(f, 7);
    let args: Vec<Vec<Value>> = [0i32, 1]
        .iter()
        .map(|&s| vec![Value::I64(0), Value::I32(s)])
        .collect();
    check(&m, &args);
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(
        n_loads(&iso),
        1,
        "the join reload is eliminated across the diamond"
    );
    assert_eq!(
        run(&iso, &[Value::I64(0), Value::I32(1)]),
        Ok(vec![Value::I64(14)])
    );
}

#[test]
fn a_store_between_blocks_forwarding() {
    // b0(addr,other): x = mem[addr]; br b1(addr, other, x)
    // b1(addr2, other2, x2): mem[other2] = 99; br b2(addr2, x2)   // may-alias store — clobbers
    // b2(addr3, x3): y = mem[addr3]; return x3 + y                // must NOT forward
    let f = Func {
        params: vec![ValType::I64, ValType::I64], // addr, other
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I64], // v0=addr, v1=other
                insts: vec![loadi(0)],                    // v2 = x
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1, 2],
                },
            },
            Block {
                params: vec![ValType::I64, ValType::I64, ValType::I64], // addr2, other2, x2
                insts: vec![Inst::ConstI64(99), storei(1, 3)],          // mem[other2] = 99
                term: Terminator::Br {
                    target: 2,
                    args: vec![0, 2],
                },
            },
            Block {
                params: vec![ValType::I64, ValType::I64], // addr3, x3
                insts: vec![loadi(0), addi(1, 2)],        // v2 = y, v3 = x3 + y
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    let m = module(f, 42);
    // Aliasing (addr == other) is the case that would miscompile if the store were ignored.
    let args = vec![
        vec![Value::I64(0), Value::I64(0)], // alias: y reads 99, so 42 + 99 = 141
        vec![Value::I64(0), Value::I64(64)], // disjoint: y reads 42, so 84
    ];
    check(&m, &args);
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(
        n_loads(&iso),
        2,
        "a store between the accesses must block cross-block forwarding"
    );
    assert_eq!(
        run(&iso, &[Value::I64(0), Value::I64(0)]),
        Ok(vec![Value::I64(141)])
    );
}

#[test]
fn load_in_a_loop_with_a_store_is_not_forwarded() {
    // b0(n, addr): x0 = mem[addr]; br b1(n, addr, x0)
    // b1(i, addr2, prev): y = mem[addr2]; mem[addr2] = i; i2 = i - 1; if i2 -> b1(i2, addr2, y) else b2(y)
    // b2(r): return r
    // The loop stores to addr each iteration, so mem[addr] changes; forwarding y to the dominating x0
    // (or across the back edge) would be unsound. The loop between them must make the pass bail.
    let f = Func {
        params: vec![ValType::I64, ValType::I64], // n, addr
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I64], // v0=n, v1=addr
                insts: vec![loadi(1)],                    // v2 = x0
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 1, 2],
                },
            },
            Block {
                params: vec![ValType::I64, ValType::I64, ValType::I64], // i, addr2, prev
                insts: vec![
                    loadi(1),          // v3 = y = mem[addr2]
                    storei(1, 0),      // mem[addr2] = i
                    Inst::ConstI64(1), // v4
                    subi(0, 4),        // v5 = i - 1
                    Inst::ConstI64(0), // v6
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: CmpOp::Ne,
                        a: 5,
                        b: 6,
                    }, // v7 = i2 != 0
                ],
                term: Terminator::BrIf {
                    cond: 7,
                    then_blk: 1,
                    then_args: vec![5, 1, 3], // (i-1, addr2, y)
                    else_blk: 2,
                    else_args: vec![3], // exit with y
                },
            },
            Block {
                params: vec![ValType::I64], // r
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    let m = module(f, 42);
    // addr = 0 (the initialized cell); a few trip counts.
    let args: Vec<Vec<Value>> = [1i64, 2, 5]
        .iter()
        .map(|&n| vec![Value::I64(n), Value::I64(0)])
        .collect();
    check(&m, &args);
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(
        n_loads(&iso),
        2,
        "a load carried around a loop with a store must not be eliminated"
    );
}

/// A diamond whose one arm stores to `p + offset` (width from `op`); the join reloads `mem[p+0]`.
/// Used to pin the cross-block alias-precision boundary: a disjoint store (i64 at offset 8) leaves the
/// reload eliminable (1 load); an overlapping store (i32 at offset 4 → bytes [4,8) overlap the i64
/// [0,8)) blocks it (2 loads). The store's value is a const typed to match the op (i64 stores need an
/// i64 value), materialized at v2 before the store.
fn diamond_with_arm_store(op: StoreOp, offset: u64) -> Module {
    let thru = |t: u32| Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![],
        term: Terminator::Br {
            target: t,
            args: vec![0, 1],
        },
    };
    let konst = if matches!(op, StoreOp::I64) {
        Inst::ConstI64(0x1234)
    } else {
        Inst::ConstI32(0x1234)
    };
    let f = Func {
        params: vec![ValType::I64, ValType::I32], // p, sel
        results: vec![ValType::I64],
        blocks: vec![
            Block {
                params: vec![ValType::I64, ValType::I32],
                insts: vec![loadi(0)], // v2 = a = mem[p+0]
                term: Terminator::BrIf {
                    cond: 1,
                    then_blk: 1,
                    then_args: vec![0, 2],
                    else_blk: 2,
                    else_args: vec![0, 2],
                },
            },
            // b1: the arm carrying the store.
            Block {
                params: vec![ValType::I64, ValType::I64],        // p, a
                insts: vec![konst, store_off(op, 0, 2, offset)], // value v2, store addr = p = v0
                term: Terminator::Br {
                    target: 3,
                    args: vec![0, 1],
                },
            },
            thru(3), // b2: no store
            Block {
                params: vec![ValType::I64, ValType::I64], // p3, a3
                insts: vec![loadi(0), addi(1, 2)],        // v2 = d = mem[p3+0], v3 = a3 + d
                term: Terminator::Return(vec![3]),
            },
        ],
    };
    module(f, 5)
}

#[test]
fn disjoint_offset_store_in_an_arm_does_not_block_forwarding() {
    // The arm stores to mem[p+8] (i64) — disjoint from the reloaded mem[p+0] off the same base — so
    // cross-block forwarding still fires: the join reload is eliminated.
    let m = diamond_with_arm_store(StoreOp::I64, 8);
    let args: Vec<Vec<Value>> = [0i32, 1]
        .iter()
        .map(|&s| vec![Value::I64(0), Value::I32(s)])
        .collect();
    check(&m, &args);
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(
        n_loads(&iso),
        1,
        "a disjoint-offset store off the same base must not block cross-block forwarding"
    );
}

#[test]
fn overlapping_store_in_an_arm_blocks_forwarding() {
    // The arm stores 4 bytes at mem[p+4] — overlapping the reloaded i64 mem[p+0] ([4,8) ∩ [0,8)) — so
    // the join reload must NOT be forwarded. Checked at sel taking the storing arm, where the store
    // changes the high word of mem[p+0], so a wrong forward would diverge.
    let m = diamond_with_arm_store(StoreOp::I32, 4);
    let args: Vec<Vec<Value>> = [0i32, 1]
        .iter()
        .map(|&s| vec![Value::I64(0), Value::I32(s)])
        .collect();
    check(&m, &args);
    let iso = optimize_module_with(&m, &only_load_elim());
    assert_eq!(
        n_loads(&iso),
        2,
        "a store overlapping the reloaded bytes must block cross-block forwarding"
    );
}
