//! Spec for the Phase 4 memory pass (`mem_forward`): intra-block redundant-load elimination and
//! store-to-load forwarding. A load made redundant by an earlier identical load, or by a matching
//! store, must be removed and its result forwarded — but only when **no intervening memory write**
//! could have changed the location, since the alias model treats any store/call as clobbering
//! everything except the same-address cell. Every case is differential-tested against the reference
//! interpreter (including the aliasing `a == b` case that would expose unsound forwarding).

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, Func, Inst, IntTy, LoadOp, Memory, Module, StoreOp, Terminator, VBitBinOp,
    ValType,
};
use svm_opt::optimize_module;
use svm_verify::verify_module;

fn module(f: Func) -> Module {
    Module {
        funcs: vec![f],
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    }
}

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

fn load(addr: u32) -> Inst {
    Inst::Load {
        op: LoadOp::I64,
        addr,
        offset: 0,
        align: 0,
    }
}
fn store(addr: u32, value: u32) -> Inst {
    Inst::Store {
        op: StoreOp::I64,
        addr,
        value,
        offset: 0,
        align: 0,
    }
}
fn add(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Add,
        a,
        b,
    }
}
fn load_off(addr: u32, offset: u64) -> Inst {
    Inst::Load {
        op: LoadOp::I64,
        addr,
        offset,
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

fn count(m: &Module, pred: impl Fn(&Inst) -> bool) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| pred(i))
        .count()
}
fn n_loads(m: &Module) -> usize {
    count(m, |i| matches!(i, Inst::Load { .. }))
}
fn n_stores(m: &Module) -> usize {
    count(m, |i| matches!(i, Inst::Store { .. }))
}

/// Optimize, re-verify, and assert behavior is preserved on the interpreter for every arg tuple.
fn check(m: &Module, args_list: &[Vec<Value>]) -> Module {
    verify_module(m).expect("input verifies");
    let opt = optimize_module(m);
    verify_module(&opt).expect("optimized re-verifies");
    for args in args_list {
        assert_eq!(run(m, args), run(&opt, args), "divergence at {args:?}");
    }
    opt
}

#[test]
fn store_then_two_loads_forward_to_the_stored_value() {
    // f(a): mem[a] = 7; x = mem[a]; y = mem[a]; return x + y.  The two loads forward to the stored
    // constant (store-to-load, then redundant-load), leaving just the store; result is 14.
    let f = Func {
        params: vec![ValType::I64], // a = v0
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                Inst::ConstI64(7), // v1
                store(0, 1),       // mem[a] = 7
                load(0),           // v2 = mem[a]  → forwards to v1
                load(0),           // v3 = mem[a]  → forwards to v1
                add(2, 3),         // v4 = x + y
            ],
            term: Terminator::Return(vec![4]),
        }],
    };
    let m = module(f);
    let args: Vec<Vec<Value>> = [0i64, 8, 1024, 65000]
        .iter()
        .map(|&a| vec![Value::I64(a)])
        .collect();
    let opt = check(&m, &args);

    assert_eq!(n_loads(&opt), 0, "both loads forward to the stored value");
    assert_eq!(
        n_stores(&opt),
        1,
        "the store itself stays (it has an effect)"
    );
    for a in [0i64, 8, 1024] {
        assert_eq!(run(&opt, &[Value::I64(a)]), Ok(vec![Value::I64(14)]));
    }
}

#[test]
fn redundant_load_survives_a_pure_op_between() {
    // f(a): x = mem[a]; t = x + 1; y = mem[a]; return t + y.  A pure op does not clobber memory, so
    // the second load is redundant and forwards to the first — one load remains.
    let f = Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                load(0),           // v1 = mem[a]
                Inst::ConstI64(1), // v2
                add(1, 2),         // v3 = x + 1
                load(0),           // v4 = mem[a]  → forwards to v1
                add(3, 4),         // v5 = t + y
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = module(f);
    let args: Vec<Vec<Value>> = [0i64, 16, 4096]
        .iter()
        .map(|&a| vec![Value::I64(a)])
        .collect();
    let opt = check(&m, &args);
    assert_eq!(n_loads(&opt), 1, "the redundant second load is eliminated");
}

#[test]
fn store_to_an_unknown_address_blocks_forwarding() {
    // f(a, b): x = mem[a]; mem[b] = 99; y = mem[a]; return x + y.  The store to `b` might alias `a`
    // (they are distinct SSA values, so not provably disjoint), so it clobbers the cached load of `a`
    // — both loads must remain. Correctness is checked with a == b (where the store *does* overwrite
    // mem[a], so y must read 99, not the forwarded x): unsound forwarding would diverge here.
    let f = Func {
        params: vec![ValType::I64, ValType::I64], // a = v0, b = v1
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts: vec![
                load(0),            // v2 = mem[a]  (x)
                Inst::ConstI64(99), // v3
                store(1, 3),        // mem[b] = 99  → clobbers the cached mem[a]
                load(0),            // v4 = mem[a]  (y) — must NOT forward to x
                add(2, 4),          // v5 = x + y
            ],
            term: Terminator::Return(vec![5]),
        }],
    };
    let m = module(f);
    // Include aliasing (a == b) and non-aliasing (a != b) tuples.
    let args = vec![
        vec![Value::I64(0), Value::I64(0)], // a == b: store overwrites mem[a]
        vec![Value::I64(0), Value::I64(16)], // a != b
        vec![Value::I64(64), Value::I64(64)], // a == b again
        vec![Value::I64(8), Value::I64(4096)],
    ];
    let opt = check(&m, &args);
    assert_eq!(
        n_loads(&opt),
        2,
        "a store to an unprovably-disjoint address must block load forwarding"
    );
    // Spot-check the aliasing case explicitly: x=0 (init), then mem[0]=99, so y=99 → 99.
    assert_eq!(
        run(&opt, &[Value::I64(0), Value::I64(0)]),
        Ok(vec![Value::I64(99)])
    );
}

#[test]
fn disjoint_offset_store_does_not_block_forwarding() {
    // f(base): x = mem[base+0]; mem[base+8] = 5; y = mem[base+0]; return x + y.  The store writes bytes
    // [8,16), disjoint from the loaded [0,8) off the *same base*, so under trap-confinement it cannot
    // touch the cached load — the second load still forwards. Result is 2*mem[base].
    let f = Func {
        params: vec![ValType::I64], // base = v0
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                load_off(0, 0),                   // v1 = mem[base+0]
                Inst::ConstI64(5),                // v2
                store_off(StoreOp::I64, 0, 2, 8), // mem[base+8] = 5  (disjoint)
                load_off(0, 0),                   // v3 = mem[base+0]  → forwards to v1
                add(1, 3),                        // v4
            ],
            term: Terminator::Return(vec![4]),
        }],
    };
    let m = module(f);
    let args: Vec<Vec<Value>> = [0i64, 96, 4096]
        .iter()
        .map(|&b| vec![Value::I64(b)])
        .collect();
    let opt = check(&m, &args);
    assert_eq!(
        n_loads(&opt),
        1,
        "a store to a disjoint offset off the same base must not block forwarding"
    );
}

#[test]
fn overlapping_offset_store_blocks_forwarding() {
    // f(base): x = mem64[base+0]; mem32[base+4] = 0x7777; y = mem64[base+0]; return x + y.  The 4-byte
    // store at offset 4 overlaps the loaded [0,8), so the second load must NOT forward — it reads the
    // modified high word. Checked with base=0, where the store changes the value (y != x).
    let f = Func {
        params: vec![ValType::I64], // base = v0
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                load_off(0, 0),                   // v1 = mem64[base+0]  reads [0,8)
                Inst::ConstI32(0x7777),           // v2
                store_off(StoreOp::I32, 0, 2, 4), // mem32[base+4] = 0x7777  overlaps [4,8)
                load_off(0, 0),                   // v3 = mem64[base+0]  — must NOT forward
                add(1, 3),                        // v4
            ],
            term: Terminator::Return(vec![4]),
        }],
    };
    let m = module(f);
    let opt = check(&m, &[vec![Value::I64(0)], vec![Value::I64(4096)]]);
    assert_eq!(
        n_loads(&opt),
        2,
        "a store overlapping the loaded bytes must block forwarding"
    );
}

#[test]
fn v128_store_forwards_to_loads() {
    // f(base): mem128[base] = C; x = mem128[base]; y = mem128[base]; return x | y.  Both v128 loads
    // forward to the stored constant (x|y = C|C = C), so no v128.load survives.
    let c = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let f = Func {
        params: vec![ValType::I64], // base = v0
        results: vec![ValType::V128],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                Inst::ConstV128(c), // v1
                Inst::V128Store {
                    addr: 0,
                    value: 1,
                    offset: 0,
                    align: 0,
                },
                Inst::V128Load {
                    addr: 0,
                    offset: 0,
                    align: 0,
                }, // v2 → forwards to v1
                Inst::V128Load {
                    addr: 0,
                    offset: 0,
                    align: 0,
                }, // v3 → forwards to v1
                Inst::VBitBin {
                    op: VBitBinOp::Or,
                    a: 2,
                    b: 3,
                }, // v4 = x | y
            ],
            term: Terminator::Return(vec![4]),
        }],
    };
    let m = module(f);
    let args: Vec<Vec<Value>> = [0i64, 64, 4096]
        .iter()
        .map(|&b| vec![Value::I64(b)])
        .collect();
    let opt = check(&m, &args);
    assert_eq!(
        count(&opt, |i| matches!(i, Inst::V128Load { .. })),
        0,
        "both v128 loads forward to the stored value"
    );
    assert_eq!(run(&opt, &[Value::I64(0)]), Ok(vec![Value::V128(c)]));
}
