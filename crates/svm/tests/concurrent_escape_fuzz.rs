//! **Generative** concurrent escape-oracle (`DESIGN.md` §4/§12/§18) — the broad-coverage version of
//! the hand-written cases in `concurrent_escape.rs`.
//!
//! `concurrent_fuzz.rs` generates interleaving-invariant concurrent programs (commutative
//! `atomic.rmw.add` to shared cells) and checks an exact checksum — a *correctness* oracle on the
//! interpreter. This file reuses that idea but turns it into an *escape* oracle on **both** backends:
//! the shared cells live at in-window addresses `cell*8`, and each generated program is run through
//! the **capture** path on the interpreter (the confinement reference) and the JIT, then their final
//! windows are **byte-compared**. Confinement of every concurrent atomic access — issued from spawned
//! threads — is the property under test; a thread-context bug diverges the windows (wrong cell) or
//! escapes (a stray write outside the cells). Out-of-window accesses fault under trap-confinement and
//! are covered by the hand-written `concurrent_escape.rs` cases.
//!
//! Determinism: atomic-add commutes, so each cell's final value (and the whole window — workers touch
//! only cells, `main` only loads) is a pure function of the program, independent of the schedule or
//! backend. Thread handles stay in SSA, never the window. Gated to the targets where the JIT runs
//! threads (`svm_fiber::supported()`).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use svm_interp::run_capture_reserved;
use svm_jit::{compile_and_run_capture_reserved, JitOutcome};

const WINDOW: u64 = 1 << 16; // `memory 16` ⇒ 64 KiB, fully mapped (reserved == mapped).

/// Reproducible xorshift PRNG (same family as `irgen`/`concurrent_fuzz`).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15 | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

struct Worker {
    cell: u64,
    amount: u64,
    iters: u64,
}

struct Program {
    text: String,
    /// `expected[c]` = final value of cell `c`, which must land confined at window offset `c*8`.
    expected: Vec<u64>,
}

/// Build a verifier-valid concurrent module whose shared cells live in-window at `cell*8`. The worker
/// (func 1) unpacks `(cell, amount, iters)` from its spawn arg (`(cell<<32)|(amount<<16)|iters`) and
/// adds `base + cell*8`. `main` (func 0) spawns and joins all workers, then loads each cell and returns
/// a checksum (so the concurrent atomic *load* path is exercised too).
fn gen_program(seed: u64) -> Program {
    let mut r = Rng::new(seed);
    let cells = r.range(1, 4);
    let n = r.range(2, 6);
    // In-window base (0): `base + cell*8 = cell*8` lands in `[0, WINDOW)`. (The generator kept a
    // window-aligned out-of-window base under the old wrap model; trap-confinement faults such
    // accesses, so the shared cells are now in-window and the atomic path is what's under test.)
    let base = 0u64;

    let workers: Vec<Worker> = (0..n)
        .map(|_| Worker {
            cell: r.range(0, cells - 1),
            amount: r.range(1, 8),
            iters: r.range(1, 150),
        })
        .collect();

    let mut expected = vec![0u64; cells as usize];
    for w in &workers {
        expected[w.cell as usize] += w.amount * w.iters;
    }

    let mut main = String::from("func () -> (i64) {\nblock0():\n  vsp = i64.const 0\n");
    for (t, w) in workers.iter().enumerate() {
        let arg = (w.cell << 32) | (w.amount << 16) | w.iters;
        main.push_str(&format!("  varg{t} = i64.const {arg}\n"));
        main.push_str(&format!("  vh{t} = thread.spawn 1 vsp varg{t}\n"));
    }
    for t in 0..workers.len() {
        main.push_str(&format!("  vj{t} = thread.join vh{t}\n"));
    }
    main.push_str("  vacc0 = i64.const 0\n");
    for c in 0..cells {
        let addr = base + c * 8; // in-window cell slot
        let wt = c + 1;
        main.push_str(&format!("  vaddr{c} = i64.const {addr}\n"));
        main.push_str(&format!("  vld{c} = i64.atomic.load vaddr{c}\n"));
        main.push_str(&format!("  vwt{c} = i64.const {wt}\n"));
        main.push_str(&format!("  vm{c} = i64.mul vld{c} vwt{c}\n"));
        main.push_str(&format!("  vacc{} = i64.add vacc{c} vm{c}\n", c + 1));
    }
    main.push_str(&format!("  return vacc{cells}\n}}\n"));

    let worker = format!(
        r#"func (i64, i64) -> (i64) {{
block0(vsp: i64, varg: i64):
  c32 = i64.const 32
  hi = i64.shr_u varg c32
  mask = i64.const 65535
  cell = i64.and hi mask
  c16 = i64.const 16
  mid = i64.shr_u varg c16
  amount = i64.and mid mask
  iters = i64.and varg mask
  eight = i64.const 8
  coff = i64.mul cell eight
  base = i64.const {base}
  off = i64.add base coff
  zero = i64.const 0
  br block1(zero, off, amount, iters)
block1(i: i64, off1: i64, amount1: i64, iters1: i64):
  cmp = i64.lt_u i iters1
  br_if cmp block2(i, off1, amount1, iters1) block3()
block2(i2: i64, off2: i64, amount2: i64, iters2: i64):
  rmw = i64.atomic.rmw.add off2 amount2
  one = i64.const 1
  inext = i64.add i2 one
  br block1(inext, off2, amount2, iters2)
block3():
  ret = i64.const 0
  return ret
}}
"#
    );

    Program {
        text: format!("memory 16\n{main}{worker}"),
        expected,
    }
}

/// Generate many concurrent programs with in-window shared cells; for each, run the interpreter
/// (confinement reference) and the JIT through the capture path and assert: both complete (no escape
/// trap), their final windows are byte-identical, and every cell at its `c*8` slot holds the
/// interleaving-invariant expected value — with nothing else in the window touched.
#[test]
fn generated_concurrent_programs_confine_in_window_atomics() {
    // Each program compiles a module + runs real threads, so keep the count modest — and smaller on
    // Windows, which charges committed window pages up front (see `jit_fuzz`).
    let programs: u64 = if cfg!(windows) { 40 } else { 150 };
    let init = vec![0u8; WINDOW as usize];

    for pseed in 0..programs {
        let prog = gen_program(pseed);
        let m = svm::text::parse_module(&prog.text)
            .unwrap_or_else(|e| panic!("parse (seed {pseed}): {e:?}\n{}", prog.text));
        svm::verify::verify_module(&m)
            .unwrap_or_else(|e| panic!("verify (seed {pseed}): {e:?}\n{}", prog.text));

        let mut fuel = 50_000_000u64;
        let (ir, imem) = run_capture_reserved(&m, 0, &[], &mut fuel, &init, 0);
        let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[], &init, 0)
            .unwrap_or_else(|e| panic!("jit compile (seed {pseed}): {e:?}\n{}", prog.text));
        assert!(
            ir.is_ok(),
            "interp trapped (seed {pseed}): {ir:?}\n{}",
            prog.text
        );
        assert!(
            matches!(jo, JitOutcome::Returned(_)),
            "jit did not return (concurrent escape?, seed {pseed}): {jo:?}\n{}",
            prog.text
        );
        assert_eq!(
            imem, jmem,
            "concurrent escape-oracle: interp/JIT windows diverge (seed {pseed})\n{}",
            prog.text
        );

        for (c, &want) in prog.expected.iter().enumerate() {
            let slot = c * 8;
            let got = u64::from_le_bytes(imem[slot..slot + 8].try_into().unwrap());
            assert_eq!(
                got, want,
                "cell {c} did not confine to slot {slot} with the expected sum \
                 (seed {pseed})\n{}",
                prog.text
            );
        }
        // Nothing outside the `cells` slots may be non-zero — a stray write would be an escape.
        let live = prog.expected.len() * 8;
        assert!(
            imem[live..].iter().all(|&b| b == 0),
            "a concurrent access landed outside the confined cells (seed {pseed})\n{}",
            prog.text
        );
    }
}
