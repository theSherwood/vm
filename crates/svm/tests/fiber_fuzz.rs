//! Structured robustness fuzzing for §12 fibers (stack switching) on the reference
//! interpreter. The byte-level fuzzer (`fuzz_smoke`) already feeds the new opcodes
//! through decode→verify→interp, but random bytes almost never form a *valid, deep*
//! resume chain. This generates verifier-valid multi-function fiber programs — random
//! mixes of `cont.new`/`cont.resume`/`suspend`/`call` — and asserts the invariants that
//! the explicit-stack interpreter must uphold no matter how fibers are nested:
//!
//!   * **Never panics** — every generated, verified module interprets to either `Ok` or a
//!     defined `Trap` (bounded by fuel). A stack-switch driver is exactly the kind of code
//!     where an off-by-one in the resume chain would panic instead of trapping.
//!   * **Deterministic** — interpreting the same module twice yields the identical result
//!     (the single-vCPU determinism the differential oracle relies on, §12).
//!   * **Serialization round-trips** — text and binary encodings are identity, so the new
//!     ops survive the whole pipeline even in adversarially-shaped programs.
//!
//! The JIT bails `Unsupported` on fibers (step 4), so this is interpreter-only by design
//! and does not touch the interp↔JIT differential.

use svm_encode::{decode_module, encode_module};
use svm_interp::run;
use svm_ir::{Block, Func, Inst, IntTy, Module, Terminator, ValType};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// Tiny deterministic PRNG (xorshift64*) — mirrors `fuzz_smoke`, no external deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// Build one verifier-valid function body. Every function has type `(i64 sp, i64 arg) ->
/// (i64)` — the fiber entry signature (§12) — so any one may serve as a fiber body, a
/// callee, or the entry, and a `cont.resume` always finds a signature-matching target. A
/// single straight-line block keeps the module trivially well-typed (operands reference
/// strictly-earlier values) while letting `cont.new`/`resume`/`suspend`/`call` interleave.
fn gen_func(g: &mut Rng, nfuncs: usize) -> Func {
    // Indices 0,1 are the params `v0: i64` (data-SP) and `v1: i64` (arg). Track which
    // produced indices hold each value type.
    let mut next: u32 = 2;
    let mut i64s: Vec<u32> = vec![0, 1];
    let mut i32s: Vec<u32> = Vec::new();
    let mut insts: Vec<Inst> = Vec::new();

    // Ensure at least one i32 value exists (for handles / funcrefs), synthesizing a const.
    macro_rules! any_i32 {
        () => {{
            if i32s.is_empty() {
                insts.push(Inst::ConstI32(g.next_u64() as i32));
                i32s.push(next);
                next += 1;
            }
            i32s[g.range(i32s.len())]
        }};
    }
    macro_rules! any_i64 {
        () => {
            i64s[g.range(i64s.len())]
        };
    }

    let n = 1 + g.range(12);
    for _ in 0..n {
        match g.range(7) {
            0 => {
                insts.push(Inst::ConstI64(g.next_u64() as i64));
                i64s.push(next);
                next += 1;
            }
            1 => {
                insts.push(Inst::ConstI32(g.next_u64() as i32));
                i32s.push(next);
                next += 1;
            }
            2 => {
                let (a, b) = (any_i64!(), any_i64!());
                insts.push(Inst::IntBin {
                    ty: IntTy::I64,
                    op: svm_ir::BinOp::Add,
                    a,
                    b,
                });
                i64s.push(next);
                next += 1;
            }
            3 => {
                // cont.new(funcref, sp) -> i32 handle. The funcref is any i32 in scope (a
                // forgeable index, masked into the func table at first resume); sp is any
                // i64 (the fiber's data-stack base).
                let func = any_i32!();
                let sp = any_i64!();
                insts.push(Inst::ContNew { func, sp });
                i32s.push(next);
                next += 1;
            }
            4 => {
                // cont.resume(handle, arg) -> (status: i32, value: i64).
                let k = any_i32!();
                let arg = any_i64!();
                insts.push(Inst::ContResume { k, arg });
                i32s.push(next); // status
                i64s.push(next + 1); // value
                next += 2;
            }
            5 => {
                // suspend(value) -> i64. Traps at the root, succeeds inside a fiber.
                let value = any_i64!();
                insts.push(Inst::Suspend { value });
                i64s.push(next);
                next += 1;
            }
            _ => {
                // call a random function (all are `(i64, i64) -> (i64)`).
                let a0 = any_i64!();
                let a1 = any_i64!();
                insts.push(Inst::Call {
                    func: g.range(nfuncs) as u32,
                    args: vec![a0, a1],
                });
                i64s.push(next);
                next += 1;
            }
        }
    }

    // Return an in-scope i64 (always at least the params).
    let ret = i64s[g.range(i64s.len())];
    Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term: Terminator::Return(vec![ret]),
        }],
    }
}

fn gen_module(g: &mut Rng) -> Module {
    let nfuncs = 1 + g.range(4);
    Module {
        funcs: (0..nfuncs).map(|_| gen_func(g, nfuncs)).collect(),
        memory: None,
        data: Vec::new(),
    }
}

#[test]
fn generated_fiber_programs_never_panic_and_are_deterministic() {
    let mut rng = Rng(0xF1BE_5EED_1234_5678);
    let mut executed = 0u64;
    for _ in 0..2_000 {
        let m = gen_module(&mut rng);
        // The generator is constructed to always produce well-typed modules.
        verify_module(&m).expect("generated module must verify");

        // Serialization round-trips even for adversarially-shaped fiber programs.
        assert_eq!(
            decode_module(&encode_module(&m)),
            Ok(m.clone()),
            "binary round-trip changed a generated fiber module"
        );
        assert_eq!(
            parse_module(&print_module(&m)),
            Ok(m.clone()),
            "text round-trip changed a generated fiber module"
        );

        // Interpret every function: never panics, and is deterministic across two runs.
        for fi in 0..m.funcs.len() as u32 {
            let args = [svm_interp::Value::I64(4096), svm_interp::Value::I64(1)];
            let mut fuel_a = 8_000u64;
            let mut fuel_b = 8_000u64;
            let a = run(&m, fi, &args, &mut fuel_a);
            let b = run(&m, fi, &args, &mut fuel_b);
            assert_eq!(a, b, "interpretation was non-deterministic");
            executed += 1;
        }
    }
    // Guard against the whole corpus silently degenerating into no-ops.
    assert!(executed > 2_000, "expected to interpret many functions");
}
