//! Stage 1 (STAGE1.md) — the **`argv[]` vector ABI** a spawned command consumes. The earlier slices
//! seeded a single token; a real command receives `main(argc, argv)`: a count, a pointer array, and a
//! string blob. This pins that marshalling layout in the child's carve and proves an applet *indexes*
//! it — reads `argc`, follows `argv[1]`'s pointer to its bytes, and echoes them to the granted stdout.
//!
//! Layout (child-window-relative), the format the personality's `spawn` will lay down:
//! ```text
//!   off 0 : argc            (i32)
//!   off 4 : argv[0] ptr     (i32, child-relative)
//!   off 8 : argv[1] ptr     (i32)
//!   off 32: argv[0] bytes   (NUL not required — lengths are fixed here)
//!   off 40: argv[1] bytes
//! ```
//! The applet resolves `stdout` by name (`instantiate_named`, op 11), loads `argv[1]`'s pointer, and
//! writes 2 bytes from it — so the output is `argv[1]`, proving pointer-array indirection, not just a
//! flat read. Differential interp==JIT; the output tracks `argv[1]`, so it's really indexed.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10;
const CARVE: u64 = 64 << 10;
const ARGV0_AT: u64 = 32; // child-relative offset of argv[0]'s bytes
const ARGV1_AT: u64 = 40; // child-relative offset of argv[1]'s bytes

/// Parent seeds an `argc=2` argv block (each arg 2 bytes) plus a `stdout` grant record, then
/// `instantiate_named`s the applet (func 1), joins, and returns its status (`argc`).
fn src(argv0: &[u8; 2], argv1: &[u8; 2]) -> String {
    // argc + the two argv pointers, as i32 stores at CARVE+{0,4,8}.
    let hdr = format!(
        "  h0 = i64.const {a0}\n  two = i32.const 2\n  i32.store h0 two\n\
         \x20 h4 = i64.const {a4}\n  pv0 = i32.const {ARGV0_AT}\n  i32.store h4 pv0\n\
         \x20 h8 = i64.const {a8}\n  pv1 = i32.const {ARGV1_AT}\n  i32.store h8 pv1\n",
        a0 = CARVE,
        a4 = CARVE + 4,
        a8 = CARVE + 8,
    );
    // The two arg strings' bytes.
    let mut strs = String::new();
    for (base, s) in [(ARGV0_AT, argv0), (ARGV1_AT, argv1)] {
        for (i, &b) in s.iter().enumerate() {
            let addr = CARVE + base + i as u64;
            strs.push_str(&format!(
                "  s{base}_{i} = i64.const {addr}\n  v{base}_{i} = i32.const {b}\n  i32.store8 s{base}_{i} v{base}_{i}\n"
            ));
        }
    }
    format!(
        r#"memory 17
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32) {{
  a0 = i64.const 0
  n100 = i32.const 100
  i32.store a0 n100
  a4 = i64.const 4
  n6 = i32.const 6
  i32.store a4 n6
  a8 = i64.const 8
  i32.store a8 vout
  a12 = i64.const 12
  z0 = i32.const 0
  i32.store a12 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  p100 = i64.const 100
  i32.store8 p100 cs
  p101 = i64.const 101
  i32.store8 p101 ct
  p102 = i64.const 102
  i32.store8 p102 cd
  p103 = i64.const 103
  i32.store8 p103 co
  p104 = i64.const 104
  i32.store8 p104 cu
  p105 = i64.const 105
  i32.store8 p105 ct
{hdr}{strs}  gp = i64.const 0
  gn = i64.const 1
  ent = i64.const 1
  off = i64.const {CARVE}
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  r = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return r
  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  a200 = i64.const 200
  i32.store8 a200 cs
  a201 = i64.const 201
  i32.store8 a201 ct
  a202 = i64.const 202
  i32.store8 a202 cd
  a203 = i64.const 203
  i32.store8 a203 co
  a204 = i64.const 204
  i32.store8 a204 cu
  a205 = i64.const 205
  i32.store8 a205 ct
  len6 = i64.const 6
  hout = cap.self.resolve a200 len6
  a8 = i64.const 8
  p1 = i32.load a8
  p1x = i64.extend_i32_u p1
  len2 = i64.const 2
  w = cap.call 0 1 (i64, i64) -> (i64) hout (p1x, len2)
  a0 = i64.const 0
  argc = i32.load a0
  argcx = i64.extend_i32_u argc
  return argcx
  }}
}}
"#
    )
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

fn run_interp(argv0: &[u8; 2], argv1: &[u8; 2]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(&src(argv0, argv1)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh)],
        &mut fuel,
        &[0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(argv0: &[u8; 2], argv1: &[u8; 2]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(&src(argv0, argv1)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, oh as i64],
        &[0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

/// The applet indexes `argv[1]` through the pointer array and echoes its bytes; the status is `argc`.
/// Output tracks `argv[1]` (not `argv[0]`), proving real pointer-array indirection — identically on
/// both backends. This is `main(argc, argv)` for a spawned command.
#[test]
fn applet_indexes_argv1_through_the_pointer_array() {
    // `argv[0]` is a decoy; only `argv[1]` should reach stdout.
    for (argv0, argv1) in [(b"ls", b"-l"), (b"wc", b"-c")] {
        let (ir, iout) = run_interp(argv0, argv1);
        let (jo, jout) = run_jit(argv0, argv1);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(2)],
            "interp: status = argc = 2 for argv {argv0:?} {argv1:?}"
        );
        assert_eq!(iout, argv1, "interp: applet echoed argv[1], not argv[0]");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2]),
            "jit: status = argc = 2, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: argv[1] echo must match interp");
    }
}
