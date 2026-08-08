//! W4 (NIM.md §3c) mechanism proof — a multi-*binary* compiler-driver shape on svm, decoupled
//! from the real nimony toolchain. nimony's `nifmake` spawns `nifler → nimony → hexer → lengc` as
//! subprocesses, each consuming the previous phase's output file. This test proves that
//! orchestration runs on svm today over an **existing seam** — the `exec` capability's
//! `domain_exec` backend (EXEC.md) — with **no new host op**: a driver module resolves `"exec"`,
//! runs phase `p1` with a seed input, drains its captured output, and feeds that output as the
//! **input** of phase `p2`, then emits `p2`'s result. Both phases and the driver are pure SVM
//! modules; each phase runs as an isolated child domain (own window, powerbox, fuel), exactly as a
//! spawned process would. This is the W4 analog of Path B's runtime shim: the mechanism, proven
//! with stand-in phases before the real phase binaries are compiled.
//!
//! The composition is content- and order-sensitive so it proves data *flowed*, not just that two
//! children ran: `p1` **doubles** its input, `p2` **echoes then appends `!`**. Seed `a` →
//! `p1` → `aa` → `p2` → `aa!`. If `p2` had been handed the original seed instead of `p1`'s output,
//! the result would be `a!` (2 bytes), not `aa!` (3) — so `aa!`/exit 3 witnesses the hand-off.

use std::sync::Arc;
use svm_run::exec::{domain_exec, domain_exec_with_fs, DomainProgram};
use svm_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Limits, Outcome, RunConfig,
};
use svm_text::parse_module;

/// Phase `p1` — a child domain that **doubles** its stdin: read up to 32 bytes into 64, then write
/// them back twice. `read`/`write` bind to the child's seeded stdin / captured stdout (the domain
/// runner wires them, exactly as `ECHO_DOMAIN`/`CAT_DOMAIN` in `exec_cap.rs`).
const P1_DOUBLE: &str = "\
memory 16
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 64
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vw2 = call.import 1 (vp, vn)
  return
  }
}
export 0 func \"_start\" 0
";

/// Phase `p2` — echo stdin, then append a fixed `!` from the phase's own data. `aa` → `aa!`.
const P2_ECHO_BANG: &str = "\
memory 16
data 0 \"!\"
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 64
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vbp = i64.const 0
  vbl = i64.const 1
  vw2 = call.import 1 (vbp, vbl)
  return
  }
}
export 0 func \"_start\" 0
";

/// The driver — the `nifmake` analog. Resolve `"exec"`; run `p1` with stdin `a`; read `p1`'s output
/// into 32..; run `p2` with **that** buffer as stdin; read `p2`'s output into 64..; emit it and exit
/// with its length. `exec` ops (iface 13): 0 = run(argv,stdin), 1 = read_out(job,buf,cap).
const DRIVER: &str = "\
memory 16
data 0 \"exec\"
data 8 \"p1\"
data 12 \"p2\"
data 16 \"a\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vp1 = i64.const 8
  vp1l = i64.const 2
  vsp = i64.const 16
  vsl = i64.const 1
  vjob1 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vp1, vp1l, vsp, vsl)
  vbuf1 = i64.const 32
  vcap = i64.const 16
  vn1 = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vjob1, vbuf1, vcap)
  vp2 = i64.const 12
  vp2l = i64.const 2
  vjob2 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vp2, vp2l, vbuf1, vn1)
  vbuf2 = i64.const 64
  vn2 = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vjob2, vbuf2, vcap)
  vw = call.import 0 (vbuf2, vn2)
  vcode = i32.wrap_i64 vn2
  call.import 1 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

fn registry() -> Imports {
    Imports::new()
        .provide("out", HostCap::stdout())
        .provide("exit", HostCap::exit())
}

fn phases() -> Vec<DomainProgram> {
    let p1 = parse_module(P1_DOUBLE).expect("parse p1");
    let p2 = parse_module(P2_ECHO_BANG).expect("parse p2");
    vec![
        DomainProgram {
            name: "p1".into(),
            instance: Arc::new(instantiate(p1).expect("instantiate p1")),
            limits: Limits::default(),
        },
        DomainProgram {
            name: "p2".into(),
            instance: Arc::new(instantiate(p2).expect("instantiate p2")),
            limits: Limits::default(),
        },
    ]
}

#[test]
fn driver_chains_two_phases_passing_output_to_input() {
    let m = parse_module(DRIVER).expect("parse driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases()))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"aa!",
            "{backend:?}: p2 consumed p1's doubled output (a -> aa -> aa!)"
        );
        assert_eq!(
            r.outcome,
            Outcome::Exited(3),
            "{backend:?}: 1 -> 2 -> 3 bytes witnesses the phase hand-off"
        );
    }
}

// ---------------------------------------------------------------------------
// The full nifmake shape: four phases, run-and-check-exit, abort on failure.
//
// The two-phase test above proves the data hand-off. A real `nifmake` driver
// does more than pipe: it runs each phase, **checks its exit status, and stops
// the pipeline if a phase fails** (a broken `nifler` must not feed garbage to
// `nimony`). This test proves that control flow at nimony's real phase depth
// (four: nifler -> nimony -> hexer -> lengc), still entirely within the tested
// `exec` seam (ops: 0 = run, 1 = read_out, 3 = status) — no new infra.
//
// Each phase appends its own letter to its stdin, so the happy path composes
// `a -> ab -> abc -> abcd -> abcde` and witnesses the 4-deep hand-off. Swapping
// one phase for a version that exits non-zero makes the driver short-circuit:
// it emits nothing and exits with the failed phase's code, and the later phases
// never run. The driver module is identical across both — only the registry
// differs — so the abort is the driver reacting to status, not a different path.

/// A stand-in phase: echo stdin, then append `letter` (its own 1-byte data). Exit 0.
fn appender(letter: &str) -> String {
    format!(
        "\
memory 16
data 0 \"{letter}\"
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {{
block 0 () {{
  vp = i64.const 64
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vbp = i64.const 0
  vbl = i64.const 1
  vw2 = call.import 1 (vbp, vbl)
  return
  }}
}}
export 0 func \"_start\" 0
"
    )
}

/// A phase that **fails**: its entry returns 7, so the child domain exits 7 (a
/// clean non-zero exit, not a trap) — the `domain_exec` outcome table maps the
/// first returned value to the exit code (`exec.rs:106`).
const P3_FAIL: &str = "\
memory 16
func 0 () -> (i64) {
block 0 () {
  vcode = i64.const 7
  return vcode
  }
}
export 0 func \"_start\" 0
";

/// The four-phase driver — the `nifmake` analog with abort-on-failure. Data: `exec`@0; phase names
/// `p1`@8 `p2`@12 `p3`@16 `p4`@20; seed `a`@24. Per phase: run it, read its output into a fresh
/// buffer (out1@64, out2@128, out3@192, out4@256), read its status, and `br_if status` to the abort
/// block (5) or the next phase. Block 4 emits the final output + exit 0; block 5 emits nothing and
/// exits with the failed phase's status. The `exec` handle is re-resolved per block (idempotent by
/// name), so only the inter-phase length is threaded.
const DRIVER4: &str = "\
memory 16
data 0 \"exec\"
data 8 \"p1\"
data 12 \"p2\"
data 16 \"p3\"
data 20 \"p4\"
data 24 \"a\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp0 = i64.const 0
  vl0 = i64.const 4
  vh0 = cap.self.resolve vp0 vl0
  va0 = i64.const 8
  vln0 = i64.const 2
  vsp0 = i64.const 24
  vsl0 = i64.const 1
  vj0 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh0 (va0, vln0, vsp0, vsl0)
  vb0 = i64.const 64
  vc0 = i64.const 32
  vn0 = cap.call 13 1 (i64, i64, i64) -> (i64) vh0 (vj0, vb0, vc0)
  vs0 = cap.call 13 3 (i64) -> (i64) vh0 (vj0)
  vs0w = i32.wrap_i64 vs0
  br_if vs0w 5(vs0) 1(vn0)
}
block 1 (q1: i64) {
  vp1 = i64.const 0
  vl1 = i64.const 4
  vh1 = cap.self.resolve vp1 vl1
  va1 = i64.const 12
  vln1 = i64.const 2
  vsp1 = i64.const 64
  vj1 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh1 (va1, vln1, vsp1, q1)
  vb1 = i64.const 128
  vc1 = i64.const 32
  vn1 = cap.call 13 1 (i64, i64, i64) -> (i64) vh1 (vj1, vb1, vc1)
  vs1 = cap.call 13 3 (i64) -> (i64) vh1 (vj1)
  vs1w = i32.wrap_i64 vs1
  br_if vs1w 5(vs1) 2(vn1)
}
block 2 (q2: i64) {
  vp2 = i64.const 0
  vl2 = i64.const 4
  vh2 = cap.self.resolve vp2 vl2
  va2 = i64.const 16
  vln2 = i64.const 2
  vsp2 = i64.const 128
  vj2 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh2 (va2, vln2, vsp2, q2)
  vb2 = i64.const 192
  vc2 = i64.const 32
  vn2 = cap.call 13 1 (i64, i64, i64) -> (i64) vh2 (vj2, vb2, vc2)
  vs2 = cap.call 13 3 (i64) -> (i64) vh2 (vj2)
  vs2w = i32.wrap_i64 vs2
  br_if vs2w 5(vs2) 3(vn2)
}
block 3 (q3: i64) {
  vp3 = i64.const 0
  vl3 = i64.const 4
  vh3 = cap.self.resolve vp3 vl3
  va3 = i64.const 20
  vln3 = i64.const 2
  vsp3 = i64.const 192
  vj3 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh3 (va3, vln3, vsp3, q3)
  vb3 = i64.const 256
  vc3 = i64.const 32
  vn3 = cap.call 13 1 (i64, i64, i64) -> (i64) vh3 (vj3, vb3, vc3)
  vs3 = cap.call 13 3 (i64) -> (i64) vh3 (vj3)
  vs3w = i32.wrap_i64 vs3
  br_if vs3w 5(vs3) 4(vn3)
}
block 4 (q4: i64) {
  vb4 = i64.const 256
  vw4 = call.import 0 (vb4, q4)
  vz4 = i32.const 0
  call.import 1 (vz4)
  unreachable
}
block 5 (code: i64) {
  vcw = i32.wrap_i64 code
  call.import 1 (vcw)
  unreachable
}
}
export 0 func \"_start\" 0
";

/// Registry of four appender phases `p1..p4` (letters b,c,d,e). If `fail_at_p3`, `p3` is swapped for
/// the failing phase — same names, so the driver is unchanged.
fn phases4(fail_at_p3: bool) -> Vec<DomainProgram> {
    let mk = |name: &str, src: String| DomainProgram {
        name: name.into(),
        instance: Arc::new(
            instantiate(parse_module(&src).expect("parse phase")).expect("instantiate phase"),
        ),
        limits: Limits::default(),
    };
    let p3 = if fail_at_p3 {
        P3_FAIL.to_string()
    } else {
        appender("d")
    };
    vec![
        mk("p1", appender("b")),
        mk("p2", appender("c")),
        mk("p3", p3),
        mk("p4", appender("e")),
    ]
}

#[test]
fn four_phase_pipeline_composes_when_every_phase_succeeds() {
    let m = parse_module(DRIVER4).expect("parse driver4");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver4");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases4(false)))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"abcde",
            "{backend:?}: 4-deep hand-off a -> ab -> abc -> abcd -> abcde"
        );
        assert_eq!(r.outcome, Outcome::Exited(0), "{backend:?}: all phases ok");
    }
}

#[test]
fn pipeline_aborts_with_the_failed_phase_status_and_skips_later_phases() {
    let m = parse_module(DRIVER4).expect("parse driver4");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver4");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases4(true)))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"",
            "{backend:?}: aborted before the success emit — p4 never ran"
        );
        assert_eq!(
            r.outcome,
            Outcome::Exited(7),
            "{backend:?}: driver exits with the failed phase's status (p3 -> 7)"
        );
    }
}

// ---------------------------------------------------------------------------
// W4 step (iii): a *real* compiled program as an isolated exec child.
//
// The phases above are hand-written stand-ins — they prove the driver's control flow (hand-off,
// status, abort), but not that a program produced by the actual `Leng → SVM-IR` backend boots and
// runs correctly as a `domain_exec` child. This closes that gap on a real compiled binary: a Leng
// module with real control flow (a counted `while` loop over scalar locals) is lowered by
// `svm-leng`, verified, registered as a phase, and `exec`d by the same nifmake-shaped driver. Its
// `main()` returns 55 (the sum 1..10), and `domain_exec` maps that return to the child's exit code,
// which the driver reads via the status op (13/3) and re-exits with — so exit 55 witnesses the whole
// path: frontend output → verify → isolated child domain → correct compute → status back to driver.
//
// Scope: this is a **self-contained compute** program (no heap, no host imports beyond the child's
// default stdin/stdout). A phase that uses the real allocator/`system` runtime additionally needs
// either a self-seeding in-window allocator (the Path-B shim, made startup-seeding) or — for the
// mmap-backed `system` — the bottom-edge host caps granted to the child, which `domain_exec` does
// not do today (it runs children with only stdin/stdout). That child-capability grant is the same
// owner-reviewed infra as the shared-memfs hand-off (NIM.md §3c), so it is scoped there, not here.

/// A real Leng compute program: `main()` sums 1..10 with a `while` loop and returns 55. Its sole
/// proc becomes func 0 — the entry `domain_exec` runs — and it takes no params and imports nothing,
/// so it runs standalone in a fresh child window. (Same shape svm-leng's own `whole_module` tests
/// compile and run; here it runs *as an exec child*.)
const SUM_1_TO_10_LENG: &str = "\
(stmts
 (proc :main.0. . (i +64) .
  (stmts .
   (var :s.0 . (i +64) 0)
   (var :i.0 . (i +64) 1)
   (while (lt i.0 11)
    (stmts .
     (asgn s.0 (add (i +64) s.0 i.0))
     (asgn i.0 (add (i +64) i.0 1))))
   (ret s.0))))";

/// The nifmake-shaped driver for one real phase: resolve `exec`, run `compute` with no stdin, read
/// its exit status (op 13/3), and re-exit with it.
const DRIVER_REAL: &str = "\
memory 16
data 0 \"exec\"
data 8 \"compute\"
import 0 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vep = i64.const 0
  vel = i64.const 4
  vh = cap.self.resolve vep vel
  vname = i64.const 8
  vnamelen = i64.const 7
  vzero = i64.const 0
  vjob = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vname, vnamelen, vzero, vzero)
  vstatus = cap.call 13 3 (i64) -> (i64) vh (vjob)
  vcode = i32.wrap_i64 vstatus
  call.import 0 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

/// Build the `compute` phase by compiling `SUM_1_TO_10_LENG` through the real svm-leng backend and
/// verifying it (the frontend is untrusted; the verifier re-checks — DESIGN.md §2a).
fn compiled_compute_phase() -> DomainProgram {
    let module = svm_leng::translate(SUM_1_TO_10_LENG).expect("svm-leng compiles the program");
    svm_verify::verify_module(&module).expect("the compiled program verifies");
    DomainProgram {
        name: "compute".into(),
        instance: Arc::new(instantiate(module).expect("instantiate compute")),
        limits: Limits::default(),
    }
}

#[test]
fn driver_runs_a_real_svm_leng_compiled_program_as_an_exec_child() {
    let m = parse_module(DRIVER_REAL).expect("parse real driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(vec![compiled_compute_phase()]))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(55),
            "{backend:?}: real compiled main() summed 1..10 in an isolated child; status reached the driver"
        );
    }
}

// ---------------------------------------------------------------------------
// W4, the faithful `nifmake` hand-off: phases pass an intermediate **file** through a shared memfs.
//
// The pipelines above hand off via stdout→stdin piping. Real `nifmake` phases pass *files*
// (`.p.nif`/`.s.nif`/…) down the chain, and this is that shape: `domain_exec_with_fs` grants every
// child one shared in-memory filesystem, so phase `gen` writes `mid` and phase `use` reads it — the
// data crosses the child boundary through the file, not a pipe. The child-fs grant is a deliberate,
// explicit widening of child authority (NIM.md §3c): plain `domain_exec` gives children only
// stdin/stdout; here the embedder hands the phases one seeded, in-memory store and nothing else.
//
// `gen` writes the bytes "OK" to `mid` and exits 0; `use` opens `mid`, reads it, echoes it to stdout,
// and exits with the byte count; the driver runs both, then emits what `use` produced. Driver stdout
// == "OK" witnesses that the file `gen` wrote reached `use` through the shared store — a fresh store
// per run, so `use` reads *this* run's write, not a leftover. (fs cap = iface 13, ops FS_OPEN=0,
// FS_READ=1, FS_WRITE=2, FS_CLOSE=4; open flags O_READ=1, O_CREATE|O_WRITE=18.)

/// Phase `gen`: create `mid` (O_CREATE|O_WRITE) and write "OK" into it.
const GEN_PHASE: &str = "\
memory 16
data 0 \"fs\"
data 4 \"mid\"
data 8 \"OK\"
func 0 () -> (i64) {
block 0 () {
  vfp = i64.const 0
  vfl = i64.const 2
  vfs = cap.self.resolve vfp vfl
  vpath = i64.const 4
  vplen = i64.const 3
  vflags = i64.const 18
  vfd = cap.call 13 0 (i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags)
  vbuf = i64.const 8
  vblen = i64.const 2
  vn = cap.call 13 2 (i64, i64, i64) -> (i64) vfs (vfd, vbuf, vblen)
  vcl = cap.call 13 4 (i64) -> (i64) vfs (vfd)
  vz = i64.const 0
  return vz
  }
}
export 0 func \"_start\" 0
";

/// Phase `use`: open `mid` (O_READ), read it into a buffer, echo it to stdout, exit with the count.
const USE_PHASE: &str = "\
memory 16
data 0 \"fs\"
data 4 \"mid\"
import 0 \"write\" (i64, i64) -> (i64)
func 0 () -> (i64) {
block 0 () {
  vfp = i64.const 0
  vfl = i64.const 2
  vfs = cap.self.resolve vfp vfl
  vpath = i64.const 4
  vplen = i64.const 3
  vflags = i64.const 1
  vfd = cap.call 13 0 (i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags)
  vbuf = i64.const 64
  vcap = i64.const 32
  vn = cap.call 13 1 (i64, i64, i64) -> (i64) vfs (vfd, vbuf, vcap)
  vw = call.import 0 (vbuf, vn)
  vcl = cap.call 13 4 (i64) -> (i64) vfs (vfd)
  return vn
  }
}
export 0 func \"_start\" 0
";

/// The driver: run `gen` (writes `mid`), then `use` (reads `mid`), drain `use`'s output and emit it.
const DRIVER_FS: &str = "\
memory 16
data 0 \"exec\"
data 8 \"gen\"
data 12 \"use\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vep = i64.const 0
  vel = i64.const 4
  vh = cap.self.resolve vep vel
  vgen = i64.const 8
  vgenl = i64.const 3
  vzero = i64.const 0
  vjg = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vgen, vgenl, vzero, vzero)
  vsg = cap.call 13 3 (i64) -> (i64) vh (vjg)
  vuse = i64.const 12
  vusel = i64.const 3
  vju = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vuse, vusel, vzero, vzero)
  vbuf = i64.const 64
  vcap = i64.const 32
  vnu = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vju, vbuf, vcap)
  vw = call.import 0 (vbuf, vnu)
  vsu = cap.call 13 3 (i64) -> (i64) vh (vju)
  vcode = i32.wrap_i64 vsu
  call.import 1 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

fn fs_phase(name: &str, src: &str) -> DomainProgram {
    DomainProgram {
        name: name.into(),
        instance: Arc::new(
            instantiate(parse_module(src).expect("parse fs phase")).expect("instantiate"),
        ),
        limits: Limits::default(),
    }
}

#[test]
fn driver_hands_off_a_file_between_phases_through_a_shared_memfs() {
    let m = parse_module(DRIVER_FS).expect("parse fs driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        // A fresh shared store per run: both phase children are granted one grant each from the same
        // factory, so `gen`'s write and `use`'s read hit the same filesystem.
        let (factory, _handle) = svm_run::fs::mem_fs_shared_factory(vec![], vec![]);
        let child_fs = HostCap::host_proc(0, factory);
        let phases = vec![fs_phase("gen", GEN_PHASE), fs_phase("use", USE_PHASE)];
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec_with_fs(phases, child_fs))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"OK",
            "{backend:?}: 'use' read the file 'gen' wrote to the shared memfs (file hand-off)"
        );
    }
}

/// Confinement probe: resolve `"fs"` and return whether it was granted (1) or not (0). `cap.self.resolve`
/// yields an i32 handle ≥ 0 on success, negative `-errno` when the name isn't granted.
const PROBE_PHASE: &str = "\
memory 16
data 0 \"fs\"
func 0 () -> (i32) {
block 0 () {
  vfp = i64.const 0
  vfl = i64.const 2
  vfs = cap.self.resolve vfp vfl
  vzero = i32.const 0
  vok = i32.ge_s vfs vzero
  return vok
  }
}
export 0 func \"_start\" 0
";

/// A driver that runs `probe` and exits with its status (1 = it got `fs`, 0 = it didn't).
const DRIVER_PROBE: &str = "\
memory 16
data 0 \"exec\"
data 8 \"probe\"
import 0 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vep = i64.const 0
  vel = i64.const 4
  vh = cap.self.resolve vep vel
  vname = i64.const 8
  vnamel = i64.const 5
  vzero = i64.const 0
  vj = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vname, vnamel, vzero, vzero)
  vs = cap.call 13 3 (i64) -> (i64) vh (vj)
  vcode = i32.wrap_i64 vs
  call.import 0 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_child_gets_fs_only_when_the_parent_grants_it() {
    // The confinement invariant behind the file hand-off: `fs` is inherited by a child ONLY through an
    // explicit `domain_exec_with_fs` grant. A child under plain `domain_exec` cannot resolve `fs` — no
    // ambient authority, and a child cannot widen its own grant.
    let m = parse_module(DRIVER_PROBE).expect("parse probe driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let (factory, _h) = svm_run::fs::mem_fs_shared_factory(vec![], vec![]);
        let granted = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[(
                    "exec",
                    domain_exec_with_fs(
                        vec![fs_phase("probe", PROBE_PHASE)],
                        HostCap::host_proc(0, factory),
                    ),
                )],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            granted.outcome,
            Outcome::Exited(1),
            "{backend:?}: an explicitly-granted child resolves fs"
        );

        let default = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(vec![fs_phase("probe", PROBE_PHASE)]))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            default.outcome,
            Outcome::Exited(0),
            "{backend:?}: a child under plain domain_exec has no fs (default confinement)"
        );
    }
}
