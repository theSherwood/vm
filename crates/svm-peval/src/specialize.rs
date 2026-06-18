//! Stage 1–2 — the **first Futamura projection** over the IR (see `PEVAL.md`).
//!
//! [`specialize`] / [`specialize_with`] / [`specialize_with_config`] take a function (an
//! *interpreter*) and a list of which parameters are **static** (a known constant at
//! specialization time) versus **dynamic** (a runtime value), and produce a residual function
//! specialized to the static inputs. Combined with a **program the caller declares constant** —
//! a readonly data segment, a caller-promised constant region, or explicit overlay bytes
//! ([`SpecConfig`]) — specializing the interpreter against that program folds the opcode loads to
//! constants, resolves the dispatch `br_table` to a single edge, and unrolls the interpreter loop
//! following the program. The dispatch loop disappears; what's left is the *compiled* program.
//! `spec(interp, program)(input) ≡ interp(program, input)`.
//!
//! The engine is **online polyvariant symbolic execution**, weval's shape:
//!
//! - Each SSA value is abstractly either a known [`Known`] constant or *dynamic* (a value in
//!   the residual block being built). Pure integer ops with all-constant operands fold (reusing
//!   the Stage-0 arithmetic, so it matches the interpreter exactly); a trapping fold (div/rem
//!   by zero) is emitted residually so it still traps. Anything with a dynamic operand is
//!   emitted into the residual.
//! - A **load from a constant address the caller declared constant** folds to those bytes
//!   ([`read_const_mem`]) — the "constant memory" read. By default that means a readonly data
//!   segment; a caller can also promise an arbitrary region or supply overlay bytes
//!   ([`SpecConfig`]). Constancy is a *caller contract*, not enforced here — a false promise is a
//!   miscompile, never an escape (the residual is re-verified). Any other load is emitted
//!   residually (faithful), so unpromised mutable memory is never folded.
//! - **Value-stack renaming (Stage 2).** A caller may designate a private byte range (the
//!   interpreter's operand stack / locals) as *renameable* ([`specialize_with`]). Stores into it
//!   at constant, full-width (`i32`/`i64`) addresses update an **abstract memory** instead of
//!   emitting a store; loads read that abstract memory instead of emitting a load — so the
//!   in-memory stack is lifted into SSA and disappears from the residual. Soundness is kept by
//!   construction: the region is assumed zero-initialized and private, every write to it is a
//!   tracked constant-address store, and any access that can't be resolved abstractly (a dynamic
//!   address that might alias the region, a partial-width overlap, a call) returns
//!   [`SpecError::Unsupported`] rather than guessing.
//! - The **context** threaded through the CFG is `(source block, the constant valuation of its
//!   block parameters, the constant valuation of the live abstract-memory cells)`. One residual
//!   block is generated per context and memoized, so distinct constants (e.g. the program
//!   counter / stack pointer) drive loop unrolling, while repeated contexts reconnect — bounding
//!   termination. Dynamic block parameters *and* dynamic memory cells become the residual block's
//!   parameters; constant ones are baked in.
//!
//! **Untrusted for escape** like the rest of the crate: the residual is meant to be
//! re-verified before it runs. The differential harness (`tests/specialize.rs`) is the spec —
//! the residual must equal the interpreter on the reference interpreter for every input.
//!
//! **Scope.** Integer / const / load / store / branch ops are specialized (folded where the
//! operands are constant). Other **pure, single-result** value ops — float and SIMD arithmetic,
//! casts, conversions, pointer ops — are emitted faithfully into the residual even though they are
//! not constant-folded (the engine tracks integer constants only), so dispatch is still eliminated
//! around them. Effectful, multi-result, or cross-function ops (calls, atomics, fibers/threads),
//! and memory accesses the engine can't resolve, return [`SpecError::Unsupported`] rather than
//! guessing.

use std::collections::{BTreeMap, HashMap, VecDeque};

use svm_ir::{ConvOp, Func, Inst, IntTy, LoadOp, Module, StoreOp, Terminator, ValType};

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
    /// An instruction (or a memory access) outside the supported subset appeared.
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

/// The constant valuation of one threaded lane (a block parameter or a memory cell): `Some` for a
/// baked-in constant, `None` for a dynamic value carried as a residual block parameter.
type ParamPattern = Vec<Option<Known>>;
/// Live abstract-memory cells at a program point, sorted by address: `(addr, width, value)`.
type MemPattern = Vec<(u64, u32, Option<Known>)>;

/// One residual block still to be generated: a source block plus the constant valuation of the
/// state threaded into it (parameters, then memory cells).
struct Task {
    src_block: u32,
    params: ParamPattern,
    mem: MemPattern,
}

/// The default ceiling on residual blocks before we declare likely divergence.
const DEFAULT_BUDGET: usize = 1 << 16;

/// What the caller promises about memory, to steer specialization. All fields default to empty,
/// which reproduces the plain Stage-1 behavior (readonly data segments still fold).
///
/// **These are caller contracts, not enforced invariants.** Declaring a region constant — or an
/// overlay's bytes — is a promise that those bytes do not change between specialization time and
/// every execution of the residual. If the promise is false (self-modifying code, a racing
/// thread, …) the residual computes the wrong answer. It is still *safe*: the residual is meant to
/// be re-verified, so confinement and capability checks hold regardless — a broken promise is a
/// miscompile, never an escape. (This mirrors weval's `assume_const_memory`.)
#[derive(Clone, Debug, Default)]
pub struct SpecConfig {
    /// A private, zero-initialized scratch range `[lo, hi)` (the interpreter's operand stack /
    /// locals) whose stores/loads are renamed into SSA and elided from the residual (Stage 2).
    pub rename: Option<(u64, u64)>,
    /// Window ranges `[lo, hi)` the caller promises are constant at specialization time. Loads from
    /// them fold to the module's initial data image (readonly **or not**); bytes not covered by any
    /// data segment read as zero (the demand-zeroed window).
    pub const_regions: Vec<(u64, u64)>,
    /// Explicit constant bytes at a base window address, for a program not described by a data
    /// segment (e.g. one written into the window before the call). Loads fully inside an overlay
    /// fold to its bytes. Overlays take precedence over data segments.
    pub const_overlays: Vec<(u64, Vec<u8>)>,
    /// Caller promise: the [`rename`](Self::rename) region is **private** — touched only by the
    /// constant-address accesses the engine renames, never by a dynamic-address load/store. With it
    /// set, a dynamic-address access (whose target the engine can't pin) is emitted as a faithful
    /// residual access instead of conservatively refusing — letting an interpreter use a renamed
    /// operand stack *and* a pointer-addressed heap at once. Unsound if violated (a dynamic write
    /// into the region would desync the elided renamed cells), so it is opt-in and off by default.
    pub rename_is_private: bool,
}

/// Specialize with no caller memory hints (only readonly data segments fold).
pub fn specialize(module: &Module, func: u32, args: &[SpecArg]) -> Result<Module, SpecError> {
    specialize_with_config(module, func, args, &SpecConfig::default())
}

/// Specialize with a renameable memory region (Stage 2 value-stack renaming), no other hints.
pub fn specialize_with(
    module: &Module,
    func: u32,
    args: &[SpecArg],
    rename: Option<(u64, u64)>,
) -> Result<Module, SpecError> {
    specialize_with_config(
        module,
        func,
        args,
        &SpecConfig {
            rename,
            ..SpecConfig::default()
        },
    )
}

/// Specialize `module.funcs[func]` against the static/dynamic binding in `args`, steered by
/// `config`. Produces a module with a single residual function (index 0); the original memory and
/// data segments are carried through, so any residual loads still resolve.
pub fn specialize_with_config(
    module: &Module,
    func: u32,
    args: &[SpecArg],
    config: &SpecConfig,
) -> Result<Module, SpecError> {
    let f = module.funcs.get(func as usize).ok_or(SpecError::BadFunc)?;
    if args.len() != f.params.len() {
        return Err(SpecError::ArityMismatch);
    }

    // The entry context, and the residual function's parameters (the dynamic args, in order).
    let mut params = Vec::with_capacity(args.len());
    let mut residual_params = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        match arg {
            SpecArg::ConstI32(v) => params.push(Some(Known::I32(*v))),
            SpecArg::ConstI64(v) => params.push(Some(Known::I64(*v))),
            SpecArg::Dynamic => {
                params.push(None);
                residual_params.push(f.params[i]);
            }
        }
    }

    let mut spec = Spec {
        module,
        f,
        config,
        memo: HashMap::new(),
        queue: VecDeque::new(),
        next_id: 0,
    };
    spec.intern(0, params, Vec::new());

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
    config: &'a SpecConfig,
    /// `(source block, param pattern, memory pattern) → residual block id`. The memo that makes
    /// the loop terminate and that closes residual loops.
    memo: HashMap<(u32, ParamPattern, MemPattern), u32>,
    queue: VecDeque<Task>,
    next_id: u32,
}

impl Spec<'_> {
    /// Get (or create) the residual block id for a context, enqueuing it the first time it is
    /// seen. Ids are assigned in enqueue order and blocks are produced in that same (FIFO) order,
    /// so id == position in the output `blocks`.
    fn intern(&mut self, src_block: u32, params: ParamPattern, mem: MemPattern) -> u32 {
        let key = (src_block, params, mem);
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Task {
            src_block: key.0,
            params: key.1.clone(),
            mem: key.2.clone(),
        });
        self.memo.insert(key, id);
        id
    }

    fn build_block(&mut self, task: Task) -> Result<svm_ir::Block, SpecError> {
        let f = self.f;
        let src = &f.blocks[task.src_block as usize];

        // Reconstruct the threaded state. Dynamic lanes become residual block parameters in a
        // canonical order: the dynamic block parameters first, then the dynamic memory cells (by
        // address). Constant lanes are baked back in.
        let mut env: Vec<Abs> = Vec::with_capacity(src.params.len());
        let mut params: Vec<ValType> = Vec::new();
        let mut rnext: u32 = 0;
        for (i, slot) in task.params.iter().enumerate() {
            match slot {
                Some(k) => env.push(Abs::Const(*k)),
                None => {
                    env.push(Abs::Dyn(rnext));
                    rnext += 1;
                    params.push(src.params[i]);
                }
            }
        }
        let mut mem: BTreeMap<u64, (u32, Abs)> = BTreeMap::new();
        for &(addr, width, slot) in &task.mem {
            match slot {
                Some(k) => {
                    mem.insert(addr, (width, Abs::Const(k)));
                }
                None => {
                    mem.insert(addr, (width, Abs::Dyn(rnext)));
                    rnext += 1;
                    params.push(cell_type(width));
                }
            }
        }

        // Symbolically execute the body.
        let mut out: Vec<Inst> = Vec::new();
        for inst in &src.insts {
            if let Some(res) = self.eval_inst(inst, &env, &mut mem, &mut out, &mut rnext)? {
                env.push(res);
            }
        }
        let term = self.eval_term(&src.term, &env, &mem, &mut out, &mut rnext)?;

        Ok(svm_ir::Block {
            params,
            insts: out,
            term,
        })
    }

    /// Abstractly evaluate one instruction. Returns the abstract value of its result (`None` for a
    /// result-less instruction such as a store), emitting any residual instruction needed.
    fn eval_inst(
        &self,
        inst: &Inst,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let abs = match *inst {
            Inst::ConstI32(v) => Abs::Const(Known::I32(v)),
            Inst::ConstI64(v) => Abs::Const(Known::I64(v)),

            Inst::IntBin { ty, op, a, b } => {
                let (av, bv) = (env[a as usize], env[b as usize]);
                if let (Abs::Const(x), Abs::Const(y)) = (av, bv) {
                    if let Some(k) = fold_int_bin(ty, op, x, y) {
                        return Ok(Some(Abs::Const(k)));
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
                        return Ok(Some(Abs::Const(k)));
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
                        return Ok(Some(Abs::Const(k)));
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
                        return Ok(Some(Abs::Const(Known::I32(b as i32))));
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
                        return Ok(Some(Abs::Const(k)));
                    }
                }
                let a = materialize(av, out, rnext);
                out.push(Inst::Convert { op, a });
                Abs::Dyn(bump(rnext))
            }
            // `select` with a constant condition forwards the chosen operand's abstract value.
            Inst::Select { cond, a, b } => {
                if let Abs::Const(c) = env[cond as usize] {
                    if let Some(c) = c.as_i32() {
                        return Ok(Some(if c != 0 {
                            env[a as usize]
                        } else {
                            env[b as usize]
                        }));
                    }
                }
                let cond = materialize(env[cond as usize], out, rnext);
                let a = materialize(env[a as usize], out, rnext);
                let b = materialize(env[b as usize], out, rnext);
                out.push(Inst::Select { cond, a, b });
                Abs::Dyn(bump(rnext))
            }

            Inst::Load {
                op,
                addr,
                offset,
                align,
            } => return self.eval_load(op, addr, offset, align, env, mem, out, rnext),
            Inst::Store {
                op,
                addr,
                value,
                offset,
                align,
            } => return self.eval_store(op, addr, value, offset, align, env, mem, out, rnext),

            // Any other pure, single-result value op (float arithmetic, casts, conversions, SIMD,
            // pointer ops, …) is emitted faithfully into the residual — folded constants flow in as
            // operands, dynamics pass through. Effectful / multi-result / memory / call ops are not
            // handled here and fall through to Unsupported.
            _ => {
                let abs =
                    emit_residual_pure(inst, env, out, rnext).ok_or(SpecError::Unsupported)?;
                return Ok(Some(abs));
            }
        };
        Ok(Some(abs))
    }

    /// A load: fold from a renameable cell, fold from readonly data, or emit a residual load.
    #[allow(clippy::too_many_arguments)]
    fn eval_load(
        &self,
        op: LoadOp,
        addr: u32,
        offset: u64,
        align: u8,
        env: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = op.info().2 as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(self.config.rename, eff, width) {
                // The renameable region must be resolved entirely abstractly.
                let rw = match renameable_load_width(op) {
                    Some(w) => w as u64,
                    None => return Err(SpecError::Unsupported), // narrow / float access into stack
                };
                if rw == width {
                    if let Some(&(wc, val)) = mem.get(&eff) {
                        if wc as u64 == width {
                            return Ok(Some(val));
                        }
                    }
                    if mem
                        .iter()
                        .any(|(&b, &(wc, _))| b < eff + width && eff < b + wc as u64)
                    {
                        return Err(SpecError::Unsupported); // partial overlap — can't resolve
                    }
                    // Untouched region cell ⇒ the zero-initialized backing.
                    let zero = match op.info().1 {
                        ValType::I32 => Known::I32(0),
                        ValType::I64 => Known::I64(0),
                        _ => return Err(SpecError::Unsupported),
                    };
                    return Ok(Some(Abs::Const(zero)));
                }
                return Err(SpecError::Unsupported);
            }
            // Outside the region: a readonly constant-memory read folds; otherwise residual.
            if let Some(k) = read_const_mem(self.config, self.module, base, offset, op) {
                return Ok(Some(Abs::Const(k)));
            }
            let addr = materialize(env[addr as usize], out, rnext);
            out.push(Inst::Load {
                op,
                addr,
                offset,
                align,
            });
            return Ok(Some(Abs::Dyn(bump(rnext))));
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if self.config.rename.is_some() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        out.push(Inst::Load {
            op,
            addr,
            offset,
            align,
        });
        Ok(Some(Abs::Dyn(bump(rnext))))
    }

    /// A store: rename into the abstract region, or emit a residual store outside it.
    #[allow(clippy::too_many_arguments)]
    fn eval_store(
        &self,
        op: StoreOp,
        addr: u32,
        value: u32,
        offset: u64,
        align: u8,
        env: &[Abs],
        mem: &mut BTreeMap<u64, (u32, Abs)>,
        out: &mut Vec<Inst>,
        rnext: &mut u32,
    ) -> Result<Option<Abs>, SpecError> {
        let width = store_width(op) as u64;
        if let Abs::Const(Known::I64(base)) = env[addr as usize] {
            let base = base as u64;
            let eff = base.wrapping_add(offset);
            if within_region(self.config.rename, eff, width) {
                let rw = match renameable_store_width(op) {
                    Some(w) => w as u64,
                    None => return Err(SpecError::Unsupported),
                };
                if rw != width {
                    return Err(SpecError::Unsupported);
                }
                // Invalidate any overlapping cell, then record this one. No residual store.
                mem.retain(|&b, &mut (wc, _)| !(b < eff + width && eff < b + wc as u64));
                mem.insert(eff, (width as u32, env[value as usize]));
                return Ok(None);
            }
            if disjoint_from_region(self.config.rename, eff, width) {
                let addr = materialize(env[addr as usize], out, rnext);
                let value = materialize(env[value as usize], out, rnext);
                out.push(Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                    align,
                });
                return Ok(None);
            }
            return Err(SpecError::Unsupported); // straddles the region boundary
        }
        // Dynamic address: with a region active it might alias the renamed stack, so refuse —
        // unless the caller has promised the region is private to the renamed accesses.
        if self.config.rename.is_some() && !self.config.rename_is_private {
            return Err(SpecError::Unsupported);
        }
        let addr = materialize(env[addr as usize], out, rnext);
        let value = materialize(env[value as usize], out, rnext);
        out.push(Inst::Store {
            op,
            addr,
            value,
            offset,
            align,
        });
        Ok(None)
    }

    fn eval_term(
        &mut self,
        term: &Terminator,
        env: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
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
                let (target, args) = self.branch_to(*target, args, env, mem);
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
                    let (target, args) = self.branch_to(blk, args, env, mem);
                    Terminator::Br { target, args }
                }
                // Dynamic condition: specialize both successors and keep the branch.
                Abs::Dyn(cond) => {
                    let (then_blk, then_args) = self.branch_to(*then_blk, then_args, env, mem);
                    let (else_blk, else_args) = self.branch_to(*else_blk, else_args, env, mem);
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
                Abs::Const(c) => {
                    let i = c.as_i32().ok_or(SpecError::Unsupported)? as u32 as usize;
                    let (blk, args) = targets.get(i).unwrap_or(default);
                    let (target, args) = self.branch_to(*blk, args, env, mem);
                    Terminator::Br { target, args }
                }
                Abs::Dyn(idx) => {
                    let targets = targets
                        .iter()
                        .map(|(blk, args)| self.branch_to(*blk, args, env, mem))
                        .collect();
                    let default = self.branch_to(default.0, &default.1, env, mem);
                    Terminator::BrTable {
                        idx,
                        targets,
                        default,
                    }
                }
            },
            Terminator::Unreachable => Terminator::Unreachable,
            Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. } => {
                return Err(SpecError::Unsupported)
            }
        })
    }

    /// Resolve one outgoing edge. The successor inherits the current abstract memory (memory
    /// persists across the branch). Constant lanes — block-argument constants and constant memory
    /// cells — join the context; dynamic lanes are passed as residual block arguments, in the
    /// canonical order (dynamic parameters first, then dynamic memory cells by address).
    fn branch_to(
        &mut self,
        target: u32,
        args: &[u32],
        env: &[Abs],
        mem: &BTreeMap<u64, (u32, Abs)>,
    ) -> (u32, Vec<u32>) {
        let mut params = Vec::with_capacity(args.len());
        let mut dyn_args = Vec::new();
        for &a in args {
            match env[a as usize] {
                Abs::Const(k) => params.push(Some(k)),
                Abs::Dyn(i) => {
                    params.push(None);
                    dyn_args.push(i);
                }
            }
        }
        let mut mem_pat = Vec::with_capacity(mem.len());
        for (&addr, &(width, val)) in mem.iter() {
            match val {
                Abs::Const(k) => mem_pat.push((addr, width, Some(k))),
                Abs::Dyn(i) => {
                    mem_pat.push((addr, width, None));
                    dyn_args.push(i);
                }
            }
        }
        let id = self.intern(target, params, mem_pat);
        (id, dyn_args)
    }
}

/// Emit a pure, single-result value op faithfully into the residual: materialize each operand
/// (a constant becomes a `const`; a dynamic reuses its residual value), then clone the op with its
/// operands rewritten. Returns `None` for anything not a pure value op (memory / call / effectful /
/// multi-result), which the caller turns into [`SpecError::Unsupported`].
///
/// "Pure value op" reuses the optimizer's [`crate::is_removable_if_dead`] whitelist (all such ops
/// are single-result and side-effect-free), plus the trapping-but-deterministic float→int
/// conversion, which is safe to emit residually (it traps at run time exactly as the source would).
fn emit_residual_pure(
    inst: &Inst,
    env: &[Abs],
    out: &mut Vec<Inst>,
    rnext: &mut u32,
) -> Option<Abs> {
    if !(crate::is_removable_if_dead(inst) || matches!(inst, Inst::FToITrap { .. })) {
        return None;
    }
    let mut clone = inst.clone();
    crate::map_operands(&mut clone, &mut |old| {
        materialize(env[old as usize], out, rnext)
    });
    out.push(clone);
    Some(Abs::Dyn(bump(rnext)))
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

/// The block-parameter type of a renameable memory cell of the given byte width.
fn cell_type(width: u32) -> ValType {
    match width {
        4 => ValType::I32,
        _ => ValType::I64,
    }
}

/// Whether `[eff, eff+width)` lies fully inside the renameable region.
fn within_region(region: Option<(u64, u64)>, eff: u64, width: u64) -> bool {
    match region {
        Some((lo, hi)) => eff >= lo && eff.checked_add(width).is_some_and(|end| end <= hi),
        None => false,
    }
}

/// Whether `[eff, eff+width)` is entirely outside the renameable region (vacuously true if none).
fn disjoint_from_region(region: Option<(u64, u64)>, eff: u64, width: u64) -> bool {
    match region {
        Some((lo, hi)) => eff
            .checked_add(width)
            .is_some_and(|end| end <= lo || eff >= hi),
        None => true,
    }
}

/// The byte width of a store op.
fn store_width(op: StoreOp) -> u32 {
    match op {
        StoreOp::I32 | StoreOp::F32 | StoreOp::I64_32 => 4,
        StoreOp::I64 | StoreOp::F64 => 8,
        StoreOp::I32_8 | StoreOp::I64_8 => 1,
        StoreOp::I32_16 | StoreOp::I64_16 => 2,
    }
}

/// Full-width integer load ops are renameable (4 or 8 bytes); narrow/float loads are not.
fn renameable_load_width(op: LoadOp) -> Option<u32> {
    match op {
        LoadOp::I32 => Some(4),
        LoadOp::I64 => Some(8),
        _ => None,
    }
}

/// Full-width integer store ops are renameable (4 or 8 bytes); narrow/float stores are not.
fn renameable_store_width(op: StoreOp) -> Option<u32> {
    match op {
        StoreOp::I32 => Some(4),
        StoreOp::I64 => Some(8),
        _ => None,
    }
}

/// Read an integer load from constant memory. The effective address `base + offset` must lie
/// fully in range (so the interpreter would not fault) and resolve to bytes the caller has
/// promised constant — a `const_overlay`, a `const_region`, or (the default) a **readonly** data
/// segment. Returns the loaded value, sign/zero-extended per `op`, matching the interpreter's
/// little-endian load exactly. Returns `None` (⇒ emit a residual load) otherwise.
fn read_const_mem(
    config: &SpecConfig,
    module: &Module,
    base: u64,
    offset: u64,
    op: LoadOp,
) -> Option<Known> {
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
    let bytes = const_bytes(config, module, eff, width)?;
    let mut raw: u64 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
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

/// Resolve `width` constant bytes at window address `eff`, if the caller has promised that range
/// constant. Precedence: an explicit overlay; then a caller `const_region` or a readonly data
/// segment, both read from the module's initial data image (uncovered bytes are zero).
fn const_bytes(config: &SpecConfig, module: &Module, eff: u64, width: u32) -> Option<Vec<u8>> {
    let width = width as u64;
    for (obase, bytes) in &config.const_overlays {
        if eff >= *obase {
            let rel = eff - *obase;
            if rel + width <= bytes.len() as u64 {
                let s = rel as usize;
                return Some(bytes[s..s + width as usize].to_vec());
            }
        }
    }
    let promised = config
        .const_regions
        .iter()
        .any(|&(lo, hi)| eff >= lo && eff + width <= hi)
        || module.data.iter().any(|d| {
            d.readonly && eff >= d.offset && eff + width <= d.offset + d.bytes.len() as u64
        });
    if !promised {
        return None;
    }
    Some((0..width).map(|i| image_byte(module, eff + i)).collect())
}

/// The byte at window address `addr` in the module's initial data image: the last data segment
/// covering it wins (segments are applied in order at instantiation), else the window is zero.
fn image_byte(module: &Module, addr: u64) -> u8 {
    let mut byte = 0u8;
    for d in &module.data {
        if addr >= d.offset && addr < d.offset + d.bytes.len() as u64 {
            byte = d.bytes[(addr - d.offset) as usize];
        }
    }
    byte
}
