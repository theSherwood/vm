//! EXEC.md acceptance — the `exec` capability end to end: a guest resolves `"exec"` by name,
//! runs `echo hi`, drains the captured stdout, and reports `{bytes, status}` — identically on
//! all three backends, and **byte-identical** between the real-subprocess `host_exec(["echo"])`
//! backend and a `scripted_exec` table seeded with the same entry (the guest cannot tell which
//! served it). Un-granted, resolve stays negative and the guest's fallback runs; an allowlist
//! miss is a refused op (negative return), never a trap.

use svm_run::exec::{host_exec, scripted_exec, ScriptedEntry};
use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// The consumer: resolves `"exec"`, runs `echo hi` (argv `"echo\0hi"` — the separator byte at
/// 12 is fresh-memory zero), reads the output into 32.., echoes it to the granted `out` stream
/// (the cross-backend byte observable), and exits with `status * 100 + nread` — 3 iff the run
/// produced `hi\n` and exit 0.
const EXEC_CONSUMER: &str = "\
memory 16
data 0 \"exec\"
data 8 \"echo\"
data 13 \"hi\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vap = i64.const 8
  val = i64.const 7
  vz = i64.const 0
  vjob = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vap, val, vz, vz)
  vbuf = i64.const 32
  vcap = i64.const 16
  vn = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vjob, vbuf, vcap)
  vs = cap.call 13 3 (i64) -> (i64) vh (vjob)
  vw = call.import 0 (vbuf, vn)
  vhund = i64.const 100
  vmul = i64.mul vs vhund
  vsum = i64.add vmul vn
  vcode = i32.wrap_i64 vsum
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

fn echo_table() -> Vec<ScriptedEntry> {
    vec![ScriptedEntry {
        argv_prefix: vec!["echo".into(), "hi".into()],
        stdout: b"hi\n".to_vec(),
        stderr: Vec::new(),
        exit: 0,
    }]
}

/// The scripted backend: deterministic, no processes — what differential tests and browser
/// embedders grant. `echo hi` → `hi\n` + exit 0, on all three backends.
#[test]
fn scripted_exec_runs_echo_identically_on_all_three_backends() {
    let m = parse_module(EXEC_CONSUMER).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", scripted_exec(echo_table()))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(r.stdout, b"hi\n", "{backend:?}: the captured output");
        assert_eq!(r.outcome, Outcome::Exited(3), "{backend:?}: status*100 + n");
    }
}

/// The real backend, attenuated to `["echo"]`: **byte-identical** to the scripted run — the
/// guest cannot tell a host process from a table entry. Unix-only (Windows has no `echo`
/// executable; the protocol itself is covered everywhere by the scripted test).
#[cfg(unix)]
#[test]
fn host_exec_matches_scripted_byte_for_byte() {
    let m = parse_module(EXEC_CONSUMER).expect("parse");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", host_exec(&["echo"]))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(r.stdout, b"hi\n", "{backend:?}: a real echo's output");
        assert_eq!(
            r.outcome,
            Outcome::Exited(3),
            "{backend:?}: real exit 0, 3 bytes"
        );
    }
}

/// Un-granted, `cap.self.resolve("exec")` stays negative and the guest's fallback runs (EXEC.md:
/// a subprocess is pure authority; without a grant the honest behavior is an error, not
/// emulation). The guest exits 42 iff the resolve came back negative.
#[test]
fn ungranted_exec_resolves_negative_and_the_fallback_runs() {
    let m = parse_module(
        "memory 16\n\
         data 0 \"exec\"\n\
         import 0 \"exit\" (i32) -> ()\n\
         func 0 () -> () {\n\
         block 0 () {\n\
           vp = i64.const 0\n\
           vl = i64.const 4\n\
           vh = cap.self.resolve vp vl\n\
           vz = i32.const 0\n\
           vneg = i32.lt_s vh vz\n\
           vft = i32.const 42\n\
           vcode = i32.mul vneg vft\n\
           call.import 0 (vcode)\n\
           unreachable\n\
           }\n\
         }\n\
         export 0 func \"_start\" 0\n",
    )
    .expect("parse");
    let inst =
        instantiate_with_imports(m, Imports::new().provide("exit", HostCap::exit())).expect("inst");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(backend, &RunConfig::default(), &[]) // no "exec" grant
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(42),
            "{backend:?}: negative resolve → the fallback path"
        );
    }
}

/// A program outside the allowlist is a **refused op** (negative return the guest can probe),
/// never a trap: `cat` against `host_exec(["echo"])` — and identically against a scripted
/// table with no `cat` entry, so the refusal shape does not reveal the backend either.
#[test]
fn a_program_outside_the_allowlist_is_refused_not_trapped() {
    // argv "cat\0hi": "cat" at 8..10, fresh-zero separator at 11, "hi" at 12..13, len 6.
    let m = parse_module(
        "memory 16\n\
         data 0 \"exec\"\n\
         data 8 \"cat\"\n\
         data 12 \"hi\"\n\
         import 0 \"exit\" (i32) -> ()\n\
         func 0 () -> () {\n\
         block 0 () {\n\
           vp = i64.const 0\n\
           vl = i64.const 4\n\
           vh = cap.self.resolve vp vl\n\
           vap = i64.const 8\n\
           val = i64.const 6\n\
           vz = i64.const 0\n\
           vjob = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vap, val, vz, vz)\n\
           vzero = i64.const 0\n\
           vrefused = i64.lt_s vjob vzero\n\
           call.import 0 (vrefused)\n\
           unreachable\n\
           }\n\
         }\n\
         export 0 func \"_start\" 0\n",
    )
    .expect("parse");
    let inst =
        instantiate_with_imports(m, Imports::new().provide("exit", HostCap::exit())).expect("inst");
    for (name, cap) in [
        ("host_exec", host_exec(&["echo"])),
        ("scripted_exec", scripted_exec(echo_table())),
    ] {
        for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
            let r = inst
                .run_with_caps(backend, &RunConfig::default(), &[("exec", cap.clone())])
                .unwrap_or_else(|e| panic!("{name}/{backend:?}: {e}"));
            assert_eq!(
                r.outcome,
                Outcome::Exited(1),
                "{name}/{backend:?}: the miss is a probeable negative, not a trap"
            );
        }
    }
}
