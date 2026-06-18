//! Stage 1 — the **first Futamura projection** over the IR (see `PEVAL.md`).
//!
//! [`specialize`] takes a function (an *interpreter*) and a list of which parameters are
//! **static** (a known constant at specialization time) versus **dynamic** (a runtime value),
//! and produces a residual function specialized to the static inputs. Combined with a
//! **readonly data segment** holding a guest program — the IR's existing notion of immutable
//! "constant memory" — specializing the interpreter against that program folds the opcode
//! loads to constants, resolves the dispatch `br_table` to a single edge, and unrolls the
//! interpreter loop following the program. The dispatch loop disappears; what's left is the
//! *compiled* program. `spec(interp, program)(input) ≡ interp(program, input)`.
//!
//! The engine is **online polyvariant symbolic execution**, weval's shape:
//!
//! - Each SSA value is abstractly either a known [`Known`] constant or *dynamic* (a value in
//!   the residual block being built). Pure integer ops with all-constant operands fold (reusing
//!   the Stage-0 arithmetic, so it matches the interpreter exactly); a trapping fold (div/rem
//!   by zero) is emitted residually so it still traps. Anything with a dynamic operand is
//!   emitted into the residual.
//! - A **load from a constant address inside a readonly data segment** folds to the bytes
//!   there ([`read_const_mem`]) — the "constant memory" read. Any other load is emitted
//!   residually (faithful), so mutable memory is never wrongly folded.
//! - The **context** is `(source block, the constant valuation of its block parameters)`. One
//!   residual block is generated per context and memoized, so distinct constants (e.g. the
//!   program counter) drive loop unrolling, while repeated contexts (a real guest loop, or a
//!   dynamic-carried loop) reconnect — bounding termination. Dynamic block parameters become
//!   the residual block's parameters; constant ones are baked in.
//!
//! **Untrusted for escape** like the rest of the crate: the residual is meant to be
//! re-verified before it runs. The differential harness (`tests/specialize.rs`) is the spec —
//! the residual must equal the interpreter on the reference interpreter for every input.
//!
//! **Scope (Stage 1).** The engine specializes the integer/const/load/branch subset an
//! accumulator-style interpreter needs; an instruction outside that subset (stores, calls,
//! floats, SIMD, atomics, fibers, …) returns [`SpecError::Unsupported`] rather than guessing.
//! Lifting the interpreter's value stack out of memory into SSA (so memory-backed interpreters
//! specialize too) is Stage 2.

use std::collections::{HashMap, VecDeque};

use svm_ir::{ConvOp, Func, Inst, IntTy, LoadOp, Module, Terminator, ValType};

use crate::{fold_int_bin, fold_int_cmp, fold_int_un, Known};

/// How one parameter of the function being specialized is bound.
#[derive(Clone, Copy, Debug)]
pub enum SpecArg {
    /// Static: a known `i32` constant at specialization time (baked into the residual).
    ConstI32(i32),
    /// Static: a known `i64` constant at specialization time.
    ConstI64(i64),
    /// Dynamic: a runtime value (becomes a parameter of the residual function).
    Dynamic,
}

/// Why specialization could not produce a residual.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpecError {
    /// The requested function index does not exist.
    BadFunc,
    /// `args.len()` did not match the function's parameter count.
    ArityMismatch,
    /// An instruction outside the Stage-1 subset appeared (see the module scope note).
    Unsupported,
    /// The residual exceeded the block budget — a likely-divergent specialization.
    Budget,
}

/// The abstract value of an SSA value during symbolic execution.
#[derive(Clone, Copy)]
enum Abs {
    /// A compile-time constant.
    Const(Known),
    /// A runtime value, identified by its index in the residual block currently being built.
    Dyn(u32),
}

/// One residual block still to be generated: a source block plus the constant valuation of its
/// parameters (the dynamic ones, in order, become the residual block's parameters).
struct Task {
    src_block: u32,
    pattern: Vec<Option<Known>>,
}

/// The default ceiling on residual blocks before we declare likely divergence.
const DEFAULT_BUDGET: usize = 1 << 16;

/// Specialize `module.funcs[func]` against the static/dynamic binding in `args`, producing a
/// module with a single residual function (index 0). The original memory and data segments are
/// carried through, so any residual loads still resolve.
pub fn specialize(module: &Module, func: u32, args: &[SpecArg]) -> Result<Module, SpecError> {
    let f = module.funcs.get(func as usize).ok_or(SpecError::BadFunc)?;
    if args.len() != f.params.len() {
        return Err(SpecError::ArityMismatch);
    }

    // The entry context, and the residual function's parameters (the dynamic args, in order).
    let mut pattern = Vec::with_capacity(args.len());
    let mut residual_params = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        match arg {
            SpecArg::ConstI32(v) => pattern.push(Some(Known::I32(*v))),
            SpecArg::ConstI64(v) => pattern.push(Some(Known::I64(*v))),
            SpecArg::Dynamic => {
                pattern.push(None);
                residual_params.push(f.params[i]);
            }
        }
    }

    let mut spec = Spec {
        module,
        f,
        memo: HashMap::new(),
        queue: VecDeque::new(),
        next_id: 0,
    };
    spec.intern(0, pattern);

    let mut blocks = Vec::new();
    while let Some(task) = spec.queue.pop_front() {
        if blocks.len() >= DEFAULT_BUDGET {
            return Err(SpecError::Budget);
        }
        let block = spec.build_block(task)?;
        blocks.push(block);
    }

    Ok(Module {
        funcs: vec![Func {
            params: residual_params,
            results: f.results.clone(),
            blocks,
        }],
        memory: module.memory,
        data: module.data.clone(),
        imports: vec![],
        debug_info: None,
    })
}

struct Spec<'a> {
    module: &'a Module,
    f: &'a Func,
    /// `(source block, constant pattern) → residual block id`. The memo that makes the loop
    /// terminate and that closes residual loops.
    memo: HashMap<(u32, Vec<Option<Known>>), u32>,
    queue: VecDeque<Task>,
    next_id: u32,
}

impl Spec<'_> {
    /// Get (or create) the residual block id for a `(source block, pattern)` context, enqueuing
    /// it for generation the first time it is seen. Ids are assigned in enqueue order and blocks
    /// are produced in that same (FIFO) order, so id == position in the output `blocks`.
    fn intern(&mut self, src_block: u32, pattern: Vec<Option<Known>>) -> u32 {
        if let Some(&id) = self.memo.get(&(src_block, pattern.clone())) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.memo.insert((src_block, pattern.clone()), id);
        self.queue.push_back(Task { src_block, pattern });
        id
    }

    fn build_block(&mut self, task: Task) -> Result<svm_ir::Block, SpecError> {
        let f = self.f;
        let src = &f.blocks[task.src_block as usize];

        // Bind parameters: constants are baked in; dynamics become residual block parameters.
        let mut env: Vec<Abs> = Vec::with_capacity(src.params.len());
        let mut params: Vec<ValType> = Vec::new();
        let mut rnext: u32 = 0;
        for (i, slot) in task.pattern.iter().enumerate() {
            match slot {
                Some(k) => env.push(Abs::Const(*k)),
                None => {
                    env.push(Abs::Dyn(rnext));
                    rnext += 1;
                    params.push(src.params[i]);
                }
            }
        }

        // Symbolically execute the body, emitting residual instructions for dynamic computation.
        let mut out: Vec<Inst> = Vec::new();
        for inst in &src.insts {
            let res = self.eval_inst(inst, &env, &mut out, &mut rnext)?;
            env.push(res);
        }
        let term = self.eval_term(&src.term, &env, &mut out, &mut rnext)?;

        Ok(svm_ir::Block {
            params,
            insts: out,
            term,
        })
    }

    /// Abstractly evaluate one instruction, returning the abstract value of its (single) result
    /// and emitting any residual instruction needed. Errors on an out-of-subset instruction.
    fn eval_inst(
        &self,
        inst: &Inst,
        env: &[Abs],
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Abs, SpecError> {
        Ok(match *inst {
            Inst::ConstI32(v) => Abs::Const(Known::I32(v)),
            Inst::ConstI64(v) => Abs::Const(Known::I64(v)),

            Inst::IntBin { ty, op, a, b } => {
                let (av, bv) = (env[a as usize], env[b as usize]);
                if let (Abs::Const(x), Abs::Const(y)) = (av, bv) {
                    if let Some(k) = fold_int_bin(ty, op, x, y) {
                        return Ok(Abs::Const(k));
                    }
                }
                let a = materialize(av, out, rnext);
                let b = materialize(bv, out, rnext);
                out.push(Inst::IntBin { ty, op, a, b });
                Abs::Dyn(bump(rnext))
            }
            Inst::IntCmp { ty, op, a, b } => {
                let (av, bv) = (env[a as usize], env[b as usize]);
                if let (Abs::Const(x), Abs::Const(y)) = (av, bv) {
                    if let Some(k) = fold_int_cmp(ty, op, x, y) {
                        return Ok(Abs::Const(k));
                    }
                }
                let a = materialize(av, out, rnext);
                let b = materialize(bv, out, rnext);
                out.push(Inst::IntCmp { ty, op, a, b });
                Abs::Dyn(bump(rnext))
            }
            Inst::IntUn { ty, op, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    if let Some(k) = fold_int_un(ty, op, x) {
                        return Ok(Abs::Const(k));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::IntUn { ty, op, a });
                Abs::Dyn(bump(rnext))
            }
            Inst::Eqz { ty, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    let z = match ty {
                        IntTy::I32 => x.as_i32().map(|v| v == 0),
                        IntTy::I64 => x.as_i64().map(|v| v == 0),
                    };
                    if let Some(b) = z {
                        return Ok(Abs::Const(Known::I32(b as i32)));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::Eqz { ty, a });
                Abs::Dyn(bump(rnext))
            }
            Inst::Convert { op, a } => {
                let av = env[a as usize];
                if let Abs::Const(x) = av {
                    let folded = match op {
                        ConvOp::ExtendI32S => x.as_i32().map(|v| Known::I64(v as i64)),
                        ConvOp::ExtendI32U => x.as_i32().map(|v| Known::I64(v as u32 as i64)),
                        ConvOp::WrapI64 => x.as_i64().map(|v| Known::I32(v as i32)),
                    };
                    if let Some(k) = folded {
                        return Ok(Abs::Const(k));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::Convert { op, a });
                Abs::Dyn(bump(rnext))
            }
            // `select` with a constant condition forwards the chosen operand's abstract value
            // directly — no residual instruction, even if the chosen value is itself dynamic.
            Inst::Select { cond, a, b } => {
                if let Abs::Const(c) = env[cond as usize] {
                    if let Some(c) = c.as_i32() {
                        return Ok(if c != 0 {
                            env[a as usize]
                        } else {
                            env[b as usize]
                        });
                    }
                }
                let cond = materialize(env[cond as usize], out, rnext);
                let a = materialize(env[a as usize], out, rnext);
                let b = materialize(env[b as usize], out, rnext);
                out.push(Inst::Select { cond, a, b });
                Abs::Dyn(bump(rnext))
            }
            // A load from a constant address inside a readonly segment folds to those bytes;
            // anything else is emitted faithfully (mutable memory is never folded).
            Inst::Load {
                op,
                addr,
                offset,
                align,
            } => {
                if let Abs::Const(Known::I64(base)) = env[addr as usize] {
                    if let Some(k) = read_const_mem(self.module, base as u64, offset, op) {
                        return Ok(Abs::Const(k));
                    }
                }
                let addr = materialize(env[addr as usize], out, rnext);
                out.push(Inst::Load {
                    op,
                    addr,
                    offset,
                    align,
                });
                Abs::Dyn(bump(rnext))
            }

            _ => return Err(SpecError::Unsupported),
        })
    }

    fn eval_term(
        &mut self,
        term: &Terminator,
        env: &[Abs],
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Terminator, SpecError> {
        Ok(match term {
            Terminator::Return(vals) => {
                let vals = vals
                    .iter()
                    .map(|v| materialize(env[*v as usize], out, rnext))
                    .collect();
                Terminator::Return(vals)
            }
            Terminator::Br { target, args } => {
                let (target, args) = self.branch_to(*target, args, env, out, rnext);
                Terminator::Br { target, args }
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => match env[*cond as usize] {
                // Static condition: dispatch resolves to the one taken edge.
                Abs::Const(c) => {
                    let taken = c.as_i32().map(|c| c != 0).ok_or(SpecError::Unsupported)?;
                    let (blk, args) = if taken {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    let (target, args) = self.branch_to(blk, args, env, out, rnext);
                    Terminator::Br { target, args }
                }
                // Dynamic condition: specialize both successors and keep the branch.
                Abs::Dyn(cond) => {
                    let (then_blk, then_args) =
                        self.branch_to(*then_blk, then_args, env, out, rnext);
                    let (else_blk, else_args) =
                        self.branch_to(*else_blk, else_args, env, out, rnext);
                    Terminator::BrIf {
                        cond,
                        then_blk,
                        then_args,
                        else_blk,
                        else_args,
                    }
                }
            },
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => match env[*idx as usize] {
                // Static index: select the one edge (the interpreter's `targets[i] else default`).
                Abs::Const(c) => {
                    let i = c.as_i32().ok_or(SpecError::Unsupported)? as u32 as usize;
                    let (blk, args) = targets.get(i).unwrap_or(default);
                    let (target, args) = self.branch_to(*blk, args, env, out, rnext);
                    Terminator::Br { target, args }
                }
                Abs::Dyn(idx) => {
                    let targets = targets
                        .iter()
                        .map(|(blk, args)| self.branch_to(*blk, args, env, out, rnext))
                        .collect();
                    let default = self.branch_to(default.0, &default.1, env, out, rnext);
                    Terminator::BrTable {
                        idx,
                        targets,
                        default,
                    }
                }
            },
            Terminator::Unreachable => Terminator::Unreachable,
            // Tail calls are control transfers out of the function — out of Stage-1 subset.
            Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. } => {
                return Err(SpecError::Unsupported)
            }
        })
    }

    /// Resolve one outgoing edge: split the edge arguments into the constant ones (which join the
    /// successor's context) and the dynamic ones (which are passed as residual block arguments),
    /// intern the resulting context, and return `(residual block id, dynamic args)`.
    fn branch_to(
        &mut self,
        target: u32,
        args: &[u32],
        env: &[Abs],
        _out: &mut [Inst],
        _rnext: &mut u32,
    ) -> (u32, Vec<u32>) {
        let mut pattern = Vec::with_capacity(args.len());
        let mut dyn_args = Vec::new();
        for &a in args {
            match env[a as usize] {
                Abs::Const(k) => pattern.push(Some(k)),
                Abs::Dyn(i) => {
                    pattern.push(None);
                    dyn_args.push(i);
                }
            }
        }
        let id = self.intern(target, pattern);
        (id, dyn_args)
    }
}

/// Turn an abstract value into a concrete residual SSA index, emitting a `const` for a constant.
fn materialize(abs: Abs, out: &mut Vec<Inst>, rnext: &mut u32) -> u32 {
    match abs {
        Abs::Dyn(i) => i,
        Abs::Const(k) => {
            out.push(k.to_const_inst());
            bump(rnext)
        }
    }
}

/// Take the next residual value index.
fn bump(rnext: &mut u32) -> u32 {
    let i = *rnext;
    *rnext += 1;
    i
}

/// Read an integer load from constant memory: the effective address `base + offset` must lie
/// fully in range (so the interpreter would not fault) and inside a single **readonly** data
/// segment. Returns the loaded value, sign/zero-extended per `op`, matching the interpreter's
/// little-endian load exactly. Returns `None` (⇒ emit a residual load) for any non-integer load,
/// a possibly-faulting range, or an address not covered by readonly data.
fn read_const_mem(module: &Module, base: u64, offset: u64, op: LoadOp) -> Option<Known> {
    let (_, vt, width, signed) = op.info();
    if !matches!(vt, ValType::I32 | ValType::I64) {
        return None;
    }
    let mem = module.memory?;
    let eff = base.checked_add(offset)?;
    let end = eff.checked_add(width as u64)?;
    if end > mem.size() {
        return None; // could fault at the window top — let the residual load reproduce it
    }
    let seg = module
        .data
        .iter()
        .find(|d| d.readonly && eff >= d.offset && end <= d.offset + d.bytes.len() as u64)?;
    let start = (eff - seg.offset) as usize;
    let mut raw: u64 = 0;
    for (i, &byte) in seg.bytes[start..start + width as usize].iter().enumerate() {
        raw |= (byte as u64) << (8 * i);
    }
    Some(match (vt, width, signed) {
        (ValType::I32, 1, false) => Known::I32(raw as u8 as i32),
        (ValType::I32, 1, true) => Known::I32(raw as u8 as i8 as i32),
        (ValType::I32, 2, false) => Known::I32(raw as u16 as i32),
        (ValType::I32, 2, true) => Known::I32(raw as u16 as i16 as i32),
        (ValType::I32, 4, _) => Known::I32(raw as u32 as i32),
        (ValType::I64, 1, false) => Known::I64(raw as u8 as i64),
        (ValType::I64, 1, true) => Known::I64(raw as u8 as i8 as i64),
        (ValType::I64, 2, false) => Known::I64(raw as u16 as i64),
        (ValType::I64, 2, true) => Known::I64(raw as u16 as i16 as i64),
        (ValType::I64, 4, false) => Known::I64(raw as u32 as i64),
        (ValType::I64, 4, true) => Known::I64(raw as u32 as i32 as i64),
        (ValType::I64, 8, _) => Known::I64(raw as i64),
        _ => return None,
    })
}
