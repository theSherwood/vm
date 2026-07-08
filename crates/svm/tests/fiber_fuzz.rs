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
//! Plus an **interp↔JIT differential** (`generated_fiber_programs_agree_on_interp_and_jit`): the
//! generated fiber programs run on *both* backends and must agree, hardening the `svm-fiber` native
//! stack-switch the JIT lowers fibers to — the exact asm a future migratable-fiber resume reuses
//! unchanged (DESIGN.md §23). It runs an **acyclic** generator (bounded call + fiber-spawn depth)
//! over a low fiber quota and only hands the JIT programs the interpreter proved terminate, so it is
//! hang-/bomb-proof by construction while exercising thousands of real resume chains.

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
/// Generate one fiber function. `acyclic_from` selects the call/fiber graph shape:
/// - `None` — a `call`/`cont.new` may target **any** function (a possibly-cyclic graph, incl. recursive
///   fiber creation). Fine for the interp-only fuzzer: recursion is heap-framed and fuel-bounded.
/// - `Some(self_idx)` — every direct `call` **and** every `cont.new` funcref targets a *strictly
///   higher* function index, so both the call graph and the fiber-spawn graph are **acyclic** (depth ≤
///   `nfuncs`). Required for the interp↔JIT differential: unbounded native recursion would overflow the
///   JIT's fixed fiber stack, and unbounded recursive *fiber* creation would fiber-bomb the two
///   backends at slightly different resource boundaries (heap frames vs `mmap`'d stacks). With the
///   spawn graph acyclic, the last function spawns no fibers and the whole tree is small, so a
///   `FiberFault` is once again a genuine *semantic* signal both backends must agree on.
fn gen_func(g: &mut Rng, nfuncs: usize, acyclic_from: Option<usize>) -> Func {
    // The lowest function index a `call`/`cont.new` may target (`0` ⇒ any).
    let call_floor = acyclic_from.map_or(0, |i| i + 1);
    // Indices 0,1 are the params `v0: i64` (data-SP) and `v1: i64` (arg). Track which
    // produced indices hold each value type.
    let mut next: u32 = 2;
    let mut i64s: Vec<u32> = vec![0, 1];
    let mut i32s: Vec<u32> = Vec::new();
    // Value indices that are *genuine* fiber handles (results of `cont.new`). The acyclic
    // differential mostly resumes these (keeping the corpus rich in real resume chains), with an
    // occasional forged i32 — since the D57 3b-i shared registry unified the handle namespace
    // (the interp's table, like the JIT's, holds only `cont.new`-created fibers), a forged handle
    // masks to the *same* slot on both backends, so forged resolution is differentially
    // comparable too. The interp-only fuzzer resumes arbitrary i32s throughout.
    let mut fiber_handles: Vec<u32> = Vec::new();
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
                // cont.new(funcref, sp) -> i64 handle; sp is any i64 (the fiber's data-stack base).
                // Interp-only: the funcref is any i32 in scope (a forgeable index, masked into the
                // func table at first resume). Acyclic differential: the funcref is a *const* equal to
                // a strictly-higher function index (`< nfuncs ≤ next_pow2`, so masking is the identity)
                // — the spawn graph stays acyclic, so a fiber-bomb can't arise. A last (leaf) function
                // has no higher target, so it emits a const i64 instead — spawning no fibers.
                let func = match call_floor {
                    0 => any_i32!(),
                    lo if lo < nfuncs => {
                        let target = lo + g.range(nfuncs - lo);
                        insts.push(Inst::ConstI32(target as i32));
                        i32s.push(next);
                        next += 1;
                        next - 1
                    }
                    _ => {
                        insts.push(Inst::ConstI64(g.next_u64() as i64));
                        i64s.push(next);
                        next += 1;
                        continue;
                    }
                };
                let sp = any_i64!();
                insts.push(Inst::ContNew { func, sp });
                // The handle is an i64 (16-bit slot + 48-bit generation); it lands directly in the
                // i64 pool so its *value* can flow into returns/args — the differential then observes
                // handle numbering itself (the D57 3b-i unified namespace: handles match across
                // backends, pinned per-program here).
                i64s.push(next);
                fiber_handles.push(next); // a genuine fiber handle
                next += 1;
            }
            4 => {
                // cont.resume(handle, arg) -> (status: i32, value: i64). The acyclic differential
                // resumes a genuine fiber handle (see `fiber_handles`) most of the time, a forged
                // i64 ~1-in-8 (comparable across backends since the 3b-i unified namespace; a
                // forged resume that traps just makes the interp skip that program). With no
                // genuine handle in scope it emits a const instead. The interp-only fuzzer
                // resumes any i64 throughout.
                let k = if acyclic_from.is_some() && g.range(8) != 0 {
                    if fiber_handles.is_empty() {
                        insts.push(Inst::ConstI64(g.next_u64() as i64));
                        i64s.push(next);
                        next += 1;
                        continue;
                    }
                    fiber_handles[g.range(fiber_handles.len())]
                } else {
                    any_i64!()
                };
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
                // call a function (all are `(i64, i64) -> (i64)`); targets are `[call_floor, nfuncs)`.
                // With no valid target (an acyclic last function), emit a const so the arm still
                // produces an i64 — keeping the corpus non-degenerate.
                if call_floor < nfuncs {
                    let a0 = any_i64!();
                    let a1 = any_i64!();
                    insts.push(Inst::Call {
                        func: (call_floor + g.range(nfuncs - call_floor)) as u32,
                        args: vec![a0, a1],
                    });
                } else {
                    insts.push(Inst::ConstI64(g.next_u64() as i64));
                }
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
        funcs: (0..nfuncs).map(|_| gen_func(g, nfuncs, None)).collect(),
        memory: None,
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
    }
}

/// Like [`gen_module`] but with an **acyclic** call graph (function `i` calls only functions `> i`),
/// so direct-call recursion depth is bounded by `nfuncs` — required for the interp↔JIT differential,
/// where the JIT runs fibers on a fixed-size native stack that unbounded recursion would overflow.
/// (Fiber-creation depth is bounded separately by only running JIT on interp-terminating programs.)
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn gen_module_acyclic(g: &mut Rng) -> Module {
    let nfuncs = 1 + g.range(4);
    Module {
        funcs: (0..nfuncs).map(|i| gen_func(g, nfuncs, Some(i))).collect(),
        memory: None,
        data: Vec::new(),
        imports: Vec::new(),
        exports: Vec::new(),
        debug_info: None,
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

/// **Differential interp↔JIT fuzzing of the fiber stack-switch.** The generated fiber programs run
/// on *both* backends and must agree — hardening the `svm-fiber` native stack-switch the JIT lowers
/// fibers to (the exact asm a future migratable-fiber resume reuses unchanged, DESIGN.md §23). The
/// interpreter is the spec.
///
/// **Termination safety (the load-bearing rule).** A generated program can recurse or fiber-bomb
/// without bound; the interpreter bounds that with **fuel** (and a depth cap), but `compile_and_run`
/// arms **no** kill-path, so running a non-terminating program on the JIT would hang or overflow its
/// OS stack. So we run the interpreter **first** and only hand the JIT programs the interp proved
/// terminate (returned `Ok` within fuel) — a completed interp run bounds the JIT to the *same* finite,
/// deterministic computation (fibers are cooperative/single-threaded, so there is no scheduling
/// nondeterminism to diverge on). Any interp trap (fuel/depth/fiber-bomb, or a semantic fault) is
/// skipped: the two backends bound resources differently, and fiber trap *kinds* are already pinned
/// by the hand-written `jit_fibers` cases. This makes the fuzzer hang-proof by construction while
/// still exercising thousands of real resume chains through the native switch.
#[test]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn generated_fiber_programs_agree_on_interp_and_jit() {
    use svm_interp::{run_with_host, Host, Value};
    use svm_jit::{CompiledModule, JitError, JitOutcome, INERT_CAP_THUNK};

    // A **low, symmetric fiber quota** on both backends. Interp fibers are cheap heap `Vec<Frame>`s,
    // but each JIT fiber `mmap`s a guard-paged native stack, so a program that creates thousands is
    // `Ok` on the interp yet exhausts the OS map limit on the JIT. `fiber_new` checks the quota
    // *before* allocating the stack, so a bomb is a clean `FiberFault` on both — and an interp run
    // that *completes* under it created ≤ `MAX_FIBERS_Q` fibers, bounding the JIT to that many stacks.
    const MAX_FIBERS_Q: usize = 64;
    let interp_quota = svm_interp::Quota {
        max_fibers: MAX_FIBERS_Q,
        ..Default::default()
    };
    let jit_quota = svm_jit::Quota {
        max_fibers: MAX_FIBERS_Q,
        ..Default::default()
    };

    // Windows commits every page against the system commit limit (no overcommit), and the JIT's
    // per-iteration window + code-arena commits don't all return immediately, so the long sweep runs
    // on Linux/macOS; Windows takes a smaller one over the same seeds (mirrors `jit_fuzz`).
    let iters: u64 = if cfg!(windows) { 400 } else { 1_500 };
    let mut rng = Rng(0x5F1B_E12D_1FF0_0D5A);
    let mut compared = 0u64;
    for _ in 0..iters {
        let m = gen_module_acyclic(&mut rng);
        verify_module(&m).expect("generated module must verify");

        // The fiber entry shape is `(i64 sp, i64 arg)`; a modest fuel keeps a runaway program's
        // interp run short (it bails `OutOfFuel`) while completing every bounded one.
        let args = [Value::I64(4096), Value::I64(1)];
        let mut fuel = 20_000u64;
        let mut host = Host::new();
        host.set_quota(interp_quota);
        let interp = run_with_host(&m, 0, &args, &mut fuel, &mut host);

        // Only the JIT-safe (interp-terminating, fiber-bounded) programs are run on the JIT — see the
        // doc comment.
        let Ok(vals) = interp else {
            continue;
        };

        // The lower-level compile entry (vs `compile_and_run`) lets us pass the matching fiber quota.
        let mut cm = match CompiledModule::compile(
            &m,
            0,
            INERT_CAP_THUNK,
            core::ptr::null_mut(),
            svm_ir::DEFAULT_RESERVED_LOG2,
            None,
            None,
            None,
            None,
            jit_quota,
            0,
        ) {
            Ok(cm) => cm,
            Err(JitError::Unsupported(_)) => continue, // off a fiber_rt target / an unlowered op
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => continue, // transient host OOM (Windows commit limit), not a divergence
            Err(e) => panic!("JIT failed to compile a verified fiber module: {e:?}\n{m:#?}"),
        };
        let (jit, _) = cm.run(&[4096, 1], None, None, None).expect("jit fiber run");

        match jit {
            JitOutcome::Returned(slots) => {
                let want: Vec<i64> = vals
                    .iter()
                    .map(|v| match v {
                        Value::I32(x) => *x as i64,
                        Value::I64(x) => *x,
                        other => panic!("unexpected fiber result type {other:?}"),
                    })
                    .collect();
                assert_eq!(
                    want, slots,
                    "interp completed but the JIT diverged on a fiber program\n{m:#?}"
                );
            }
            // The interp returned a value, so the program terminates; the JIT must too, identically.
            other => panic!("interp returned {vals:?} but the JIT gave {other:?}\n{m:#?}"),
        }
        compared += 1;
    }
    // Coverage guard: most generated programs terminate, so the differential must actually fire on a
    // healthy fraction (not silently skip everything via `Unsupported`/`OutOfFuel`/fiber-bomb).
    // ~⅓ of programs are compared; the rest are skipped (an entry-level `suspend` → `FiberFault`, a
    // fuel/fiber bound, or an unlowered op). A quarter is a comfortable floor that still guarantees
    // the differential actually fired on hundreds of real resume chains.
    assert!(
        compared > iters / 4,
        "the interp↔JIT fiber differential compared only {compared}/{iters} programs"
    );
}

/// **Randomized-migration differential (the D57 3c empirical net, layer 1 — DESIGN.md §23
/// "Verification story").** Generates programs in which fibers created (and part-run) on the root
/// vCPU are suspended and resumed across a random sequence of *spawned* vCPUs — every worker is a
/// fresh OS thread, so each generated "resume a fiber last suspended elsewhere" step drives the
/// real `svm-fiber` stack switch **cross-thread** through the shared-table claim. The interpreter
/// (safe Rust, the 3b-i migrating registry) is the oracle: results must match exactly.
///
/// Determinism by construction: workers run strictly sequentially (each `thread.spawn` is
/// immediately `thread.join`ed), every fiber is resumed at most `suspends + 1` times (no faults),
/// and all values are pure integer arithmetic — so one canonical execution order exists and any
/// interp↔JIT divergence is a migration-seam bug, not scheduling noise. Migration coverage is
/// **counted, not assumed**: the generator tallies resumes of a suspended fiber by a different
/// executor than the one that suspended it, and the test asserts the corpus actually performed
/// thousands of them.
#[test]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn generated_migration_schedules_agree_on_interp_and_jit() {
    use std::fmt::Write as _;
    use svm_interp::{run, Value};
    use svm_jit::{compile_and_run, JitOutcome};

    /// One generated step: executor `exec` (0 = root, 1.. = worker) resumes fiber `fiber` with a
    /// constant argument.
    struct Step {
        fiber: usize,
        arg: i64,
    }

    /// Emit one resume step into a function body. `addr_base + 4*fiber` holds the fiber's handle
    /// (the root stores each handle right after `cont.new`). Accumulates `1000*status + value`.
    fn emit_step(src: &mut String, v: &mut u32, step: &Step, acc: u32) -> u32 {
        let a = *v;
        writeln!(src, "  v{a} = i64.const {}", 16 + 8 * step.fiber).unwrap();
        writeln!(src, "  v{} = i64.load v{a}", a + 1).unwrap(); // i64 fiber handle
        writeln!(src, "  v{} = i64.const {}", a + 2, step.arg).unwrap();
        writeln!(
            src,
            "  v{}, v{} = cont.resume v{} v{}",
            a + 3,
            a + 4,
            a + 1,
            a + 2
        )
        .unwrap();
        writeln!(src, "  v{} = i64.extend_i32_u v{}", a + 5, a + 3).unwrap();
        writeln!(src, "  v{} = i64.const 1000", a + 6).unwrap();
        writeln!(src, "  v{} = i64.mul v{} v{}", a + 7, a + 5, a + 6).unwrap();
        writeln!(src, "  v{} = i64.add v{} v{}", a + 8, acc, a + 7).unwrap();
        writeln!(src, "  v{} = i64.add v{} v{}", a + 9, a + 8, a + 4).unwrap();
        *v += 10;
        a + 9
    }

    let mut rng = Rng(0x3C3C_F1BE_5EED_77AA);
    let iters: u64 = if cfg!(windows) { 80 } else { 250 };
    let mut migrations = 0u64;
    // Programs skipped for a transient Windows fiber-stack allocation failure (see the JIT arm).
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut alloc_skips = 0u64;
    for _ in 0..iters {
        let nf = 1 + rng.range(3); // fibers
        let nw = 1 + rng.range(3); // sequential workers (each a fresh OS thread)
        let suspends: Vec<usize> = (0..nf).map(|_| 1 + rng.range(3)).collect();
        let mut budget: Vec<usize> = suspends.iter().map(|s| s + 1).collect();

        // Phases alternate root, worker1, root, worker2, … root. Each phase takes 0..=2 random
        // steps on fibers with remaining resume budget.
        let nphases = 2 * nw + 1;
        let mut phases: Vec<Vec<Step>> = Vec::new();
        // Which executor last ran each fiber (suspended it), for the migration tally. `None` =
        // never started (a first resume enters a fresh stack — not a saved-context migration).
        let mut last_exec: Vec<Option<usize>> = vec![None; nf];
        for p in 0..nphases {
            let exec = if p % 2 == 0 { 0 } else { p.div_ceil(2) }; // 0 = root, else worker index
            let mut steps = Vec::new();
            for _ in 0..rng.range(3) {
                let candidates: Vec<usize> = (0..nf).filter(|&f| budget[f] > 0).collect();
                if candidates.is_empty() {
                    break;
                }
                let fiber = candidates[rng.range(candidates.len())];
                budget[fiber] -= 1;
                if let Some(prev) = last_exec[fiber] {
                    if prev != exec {
                        migrations += 1; // resuming a stack suspended on a different executor
                    }
                }
                last_exec[fiber] = Some(exec);
                steps.push(Step {
                    fiber,
                    arg: (rng.next_u64() % 1000) as i64,
                });
            }
            phases.push(steps);
        }

        // ---- Emit the module: func 0 = root, funcs 1..=nw = workers, then the fiber bodies.
        let mut src = String::from("memory 16\n");
        // Root: create the fibers (handle of fiber f stored at mem[16+8f]), then run its phases,
        // spawning + joining each worker in between (strictly sequential).
        src.push_str("func () -> (i64) {\nblock0():\n");
        let mut v: u32 = 0;
        for f in 0..nf {
            writeln!(src, "  v{v} = ref.func {}", 1 + nw + f).unwrap();
            writeln!(src, "  v{} = i64.const {}", v + 1, 4096 * (f + 1)).unwrap();
            writeln!(src, "  v{} = cont.new v{v} v{}", v + 2, v + 1).unwrap();
            writeln!(src, "  v{} = i64.const {}", v + 3, 16 + 8 * f).unwrap();
            writeln!(src, "  i64.store v{} v{}", v + 3, v + 2).unwrap();
            v += 4;
        }
        writeln!(src, "  v{v} = i64.const 0").unwrap(); // the root's accumulator
        let mut acc = v;
        v += 1;
        for (p, steps) in phases.iter().enumerate() {
            if p % 2 == 0 {
                for s in steps {
                    acc = emit_step(&mut src, &mut v, s, acc);
                }
            } else {
                // Spawn worker p/2+1 (func index (p+1)/2), join it immediately, accumulate.
                let w = p.div_ceil(2);
                writeln!(src, "  v{v} = i64.const 0").unwrap();
                writeln!(src, "  v{} = thread.spawn {} v{v} v{v}", v + 1, w).unwrap();
                writeln!(src, "  v{} = thread.join v{}", v + 2, v + 1).unwrap();
                writeln!(src, "  v{} = i64.add v{acc} v{}", v + 3, v + 2).unwrap();
                acc = v + 3;
                v += 4;
            }
        }
        writeln!(src, "  return v{acc}").unwrap();
        src.push_str("}\n");
        // Workers: each runs its phase's steps and returns its accumulator.
        for w in 1..=nw {
            src.push_str("func (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n");
            let mut v: u32 = 2;
            writeln!(src, "  v{v} = i64.const 0").unwrap();
            let mut acc = v;
            v += 1;
            for s in &phases[2 * w - 1] {
                acc = emit_step(&mut src, &mut v, s, acc);
            }
            writeln!(src, "  return v{acc}").unwrap();
            src.push_str("}\n");
        }
        // Fiber bodies: fiber f does `suspends[f]` suspends, mixing each delivered value in.
        for s in &suspends {
            src.push_str("func (i64, i64) -> (i64) {\nblock0(v0: i64, v1: i64):\n");
            let mut v: u32 = 2;
            let mut acc = 1; // start from the first-resume arg
            for _ in 0..*s {
                writeln!(src, "  v{v} = i64.const {}", (rng.next_u64() % 97) as i64).unwrap();
                writeln!(src, "  v{} = i64.add v{acc} v{v}", v + 1).unwrap();
                writeln!(src, "  v{} = suspend v{}", v + 2, v + 1).unwrap();
                writeln!(src, "  v{} = i64.const 3", v + 3).unwrap();
                writeln!(src, "  v{} = i64.mul v{} v{}", v + 4, v + 2, v + 3).unwrap();
                writeln!(src, "  v{} = i64.add v{} v{}", v + 5, v + 4, v + 1).unwrap();
                acc = v + 5;
                v += 6;
            }
            writeln!(src, "  return v{acc}").unwrap();
            src.push_str("}\n");
        }

        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));

        // Interp (the migrating oracle) — deterministic, must complete.
        let mut fuel = 10_000_000u64;
        let interp = run(&m, 0, &[], &mut fuel).unwrap_or_else(|t| {
            panic!("interp trapped on a by-construction-valid program: {t:?}\n{src}")
        });
        // JIT — the real cross-thread stack switches; must match exactly.
        match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
            JitOutcome::Returned(slots) => {
                let want: Vec<i64> = interp
                    .iter()
                    .map(|x| match x {
                        Value::I64(x) => *x,
                        other => panic!("unexpected result type {other:?}"),
                    })
                    .collect();
                assert_eq!(
                    want, slots,
                    "interp and JIT diverged on a migration schedule\n{src}"
                );
            }
            other => {
                // ISSUES.md I1/I3: on a memory-tight Windows runner the fiber control-stack
                // `VirtualAlloc` behind `cont.new` can fail transiently under the run's cumulative
                // commit pressure — a recoverable `FiberFault` trap (I1), not a backend divergence
                // (the explicit-stack interp allocates no native stack, so it proceeds). Skip the
                // program, like the other cross-backend harnesses skip transient host allocation
                // failures; a *deterministic* JIT `FiberFault` still fails the suite, as Linux and
                // macOS run a superset of these seeds (same RNG, longer loop) with no skip.
                #[cfg(windows)]
                if matches!(other, JitOutcome::Trapped(svm_jit::TrapKind::FiberFault)) {
                    alloc_skips += 1;
                    continue;
                }
                panic!("interp returned {interp:?} but the JIT gave {other:?}\n{src}")
            }
        }
    }
    // The skip above is for *transient* allocation pressure only: if a meaningful share of the
    // corpus FiberFaults, that is systematic (a real Windows fiber-stack bug), not pressure — fail.
    assert!(
        alloc_skips <= iters / 4,
        "too many JIT FiberFaults to be transient allocation pressure: {alloc_skips}/{iters}"
    );
    // Coverage guard: the corpus must have actually exercised saved-stack cross-executor resumes
    // (≈1.9 per program empirically; a quarter of that is a comfortable floor).
    assert!(
        migrations > iters / 2,
        "migration corpus degenerated: only {migrations} cross-executor resumes in {iters} programs"
    );
}
