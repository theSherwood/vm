//! End-to-end tests for the embedding runtime + CLI: hand-written sandboxed programs (text IR,
//! no frontend) exercised through `run_powerbox`/`run_kernel`, the shipped demo, and the `svm-run`
//! binary itself driving a `.svm` file with real stdout + exit code.

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use svm_ir::Module;
use svm_run::{
    is_powerbox_entry, run_kernel, run_powerbox, run_powerbox_with_deadline,
    run_powerbox_with_deadline_and_quota, Outcome, Quota, Value,
};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Parse + verify a text-IR program (verification is what makes it safe to run).
fn load(src: &str) -> Module {
    let m = parse_module(src).expect("parse text IR");
    verify_module(&m).expect("verify");
    m
}

#[test]
fn writes_to_stdout_and_returns() {
    // A powerbox program: write "hi\n" to stdout (Stream cap, type 0 op 1) on the granted
    // stdout handle (v0), then return 7.
    let m = load(
        "memory 16\n\
         data 16 \"hi\\n\"\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 16\n\
         \x20 v4 = i64.const 3\n\
         \x20 v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)\n\
         \x20 v6 = i32.const 7\n\
         \x20 return v6\n\
         }\n",
    );
    assert!(is_powerbox_entry(&m));
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.stdout, b"hi\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(7)]));
    assert!(run.stderr.is_empty());
}

#[test]
fn exit_capability_sets_code() {
    // The guest invokes Exit(5) (type 1 op 0) on the granted exit handle (v2) — terminal.
    let m = load(
        "func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i32.const 5\n\
         \x20 cap.call 1 0 (i32) -> () v2(v3)\n\
         \x20 unreachable\n\
         }\n",
    );
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.outcome, Outcome::Exited(5));
}

#[test]
fn echoes_stdin_to_stdout() {
    // read(stdin) into the window, then write that many bytes back out — a stdin round-trip.
    let m = load(
        "memory 16\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 0\n\
         \x20 v4 = i64.const 64\n\
         \x20 v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)\n\
         \x20 v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)\n\
         \x20 v7 = i32.const 0\n\
         \x20 return v7\n\
         }\n",
    );
    let run = run_powerbox(&m, b"ping").expect("run");
    assert_eq!(run.stdout, b"ping");
}

#[test]
fn bare_kernel_returns_value() {
    // A non-powerbox entry — a pure function (i64 x) -> (i64) returning x + 1.
    let m = load(
        "func (i64) -> (i64) {\n\
         block0(v0: i64):\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.add v0 v1\n\
         \x20 return v2\n\
         }\n",
    );
    assert!(!is_powerbox_entry(&m));
    let out = run_kernel(&m, &[41]).expect("run kernel");
    assert_eq!(out, vec![Value::I64(42)]);
}

fn demo(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("demos")
        .join(name)
}

/// Run a demo through the `svm-run` binary with a hard wall-clock timeout, so the rare
/// work-stealing **wedge** (ISSUES.md I7 — a liveness flake in the guest scheduler / fiber-steal
/// path, seen once on Linux CI where it hung the job for >1 h) fails *fast* with a captured thread
/// dump instead of blocking CI for hours. Two layers:
///   1. `SVM_DEADLINE_MS` arms the §5 detect-and-kill, so a *guest-side* wedge — a spinning **or**
///      futex-parked worker (`KILL_RECHECK` wakes a parked vCPU) — is unwound and the process exits
///      non-zero with the runner's own kill diagnostic.
///   2. A host-side process timeout is the backstop for any stall the guest kill can't reach: on
///      expiry it best-effort dumps every thread's backtrace (`gdb -p` — the exact root-cause data
///      I7 asks for) and then SIGKILLs the child.
///
/// A healthy demo finishes in milliseconds, far under either bound, so this never trips normally.
#[cfg(all(unix, target_arch = "x86_64"))]
fn run_demo_failfast(rel: &str) -> std::process::Output {
    use std::io::Read;
    use std::process::Stdio;
    use std::sync::mpsc;

    // A wedge is pathological; a healthy run is ~milliseconds. The guest deadline sits well inside
    // the process timeout so a guest-side wedge fails with the graceful kill message before the
    // hard backstop fires.
    const GUEST_DEADLINE_MS: u64 = 30_000;
    const HARD_TIMEOUT: Duration = Duration::from_secs(90);

    let mut child = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo(rel))
        .env("SVM_DEADLINE_MS", GUEST_DEADLINE_MS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn svm-run");
    let pid = child.id();

    // Drain the pipes on their own threads so a chatty child can't deadlock on a full pipe buffer.
    let mut so = child.stdout.take().expect("stdout pipe");
    let mut se = child.stderr.take().expect("stderr pipe");
    let h_out = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = so.read_to_end(&mut v);
        v
    });
    let h_err = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = se.read_to_end(&mut v);
        v
    });

    // Wait for exit on a helper thread; the main thread enforces the timeout, keeping the ability to
    // SIGKILL the child by pid if it wedges (a moved-in `Child::wait` would surrender that).
    let (tx, rx) = mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let _ = tx.send(child.wait());
    });

    match rx.recv_timeout(HARD_TIMEOUT) {
        Ok(status) => {
            let status = status.expect("wait svm-run");
            let stdout = h_out.join().expect("join stdout reader");
            let stderr = h_err.join().expect("join stderr reader");
            let _ = waiter.join();
            std::process::Output {
                status,
                stdout,
                stderr,
            }
        }
        Err(_) => {
            // Wedged. Capture every thread's backtrace before killing — the root-cause data I7 asks
            // for ("attach gdb and dump all thread backtraces"). Best-effort: gdb may be absent or
            // ptrace-restricted on a given runner, in which case we still SIGKILL and fail.
            match Command::new("gdb")
                .args([
                    "-p",
                    &pid.to_string(),
                    "-batch",
                    "-nx",
                    "-ex",
                    "set pagination off",
                    "-ex",
                    "info threads",
                    "-ex",
                    "thread apply all bt",
                ])
                .output()
            {
                Ok(o) => eprintln!(
                    "I7 WEDGE thread dump for {rel} (pid {pid}):\n{}\n{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr),
                ),
                Err(e) => eprintln!("I7 WEDGE: gdb dump unavailable ({e}); killing pid {pid}"),
            }
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
            let _ = waiter.join();
            panic!(
                "demo {rel} WEDGED (> {HARD_TIMEOUT:?}) — the ISSUES.md I7 work-stealing liveness \
                 flake. Fail-fast tripped (this used to hang CI for hours); see the thread dump above."
            );
        }
    }
}

#[test]
fn runs_shipped_demo() {
    let src = std::fs::read_to_string(demo("hello.svm")).expect("read hello.svm");
    let m = load(&src);
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(run.stdout, b"hello, sandbox!\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

/// Drive the actual `svm-run` binary on the demo `.svm`: it must print the greeting to stdout
/// and exit 0 — the "a program runs in the sandbox from the command line" milestone.
#[test]
fn cli_runs_svm_file() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("hello.svm"))
        .output()
        .expect("spawn svm-run");
    assert!(
        out.status.success(),
        "exit: {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"hello, sandbox!\n");
}

/// Compile a demo `.c` with native `cc`, run both it and `svm-run`, and assert identical
/// stdout — a real-program oracle (the whole stack vs. a real compiler). Skipped (not failed)
/// when `cc` or the frontend is unavailable.
fn assert_demo_matches_cc(name: &str) {
    let c = demo(name);
    let exe = std::env::temp_dir().join(format!(
        "svm_demo_{}_{}",
        std::process::id(),
        name.replace(['.', '/'], "_") // flatten subdirs (e.g. `jsmn/jsmn_demo.c`) into one name
    ));
    match Command::new("cc").arg(&c).arg("-o").arg(&exe).status() {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let native = Command::new(&exe).output().expect("run native build");
    let svm = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(&c)
        .output()
        .expect("spawn svm-run");
    if !svm.status.success() {
        let err = String::from_utf8_lossy(&svm.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping {name} (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on {name} failed: {err}");
    }
    assert_eq!(
        String::from_utf8_lossy(&svm.stdout),
        String::from_utf8_lossy(&native.stdout),
        "{name}: svm-run vs native cc stdout differ"
    );
}

/// A recursive-descent calculator (recursion, a global string table + a global struct-array of
/// function pointers, indirect dispatch) — sandboxed output must match native `cc`.
#[test]
fn demo_calc_matches_native() {
    assert_demo_matches_cc("calc.c");
}

/// **The capstone: a real third-party C library runs in the sandbox.** The Clay UI layout
/// library (`demos/clay/clay.h`, ~5k lines, zlib-licensed, vendored) compiles through the
/// frontend to ~93k lines of IR, verifies, and runs on the JIT — building a small layout and
/// printing its render commands, deterministically and identically to a native build. Skipped
/// (not failed) when the chibicc frontend is unavailable.
#[test]
fn demo_clay_layout_runs() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("clay/clay_demo.c"))
        .output()
        .expect("spawn svm-run");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping Clay demo (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on clay_demo.c failed: {err}");
    }
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "3 render commands:\n\
         \x20 cmd 1 bbox=(16,16 768x40)\n\
         \x20 cmd 3 bbox=(16,16 152x18)\n\
         \x20 cmd 1 bbox=(16,64 768x520)\n",
    );
}

/// Exact-rational arithmetic (by-value struct args/returns through direct *and* indirect calls,
/// recursion) — sandboxed output must match native `cc`. The program that surfaced the
/// sret-from-a-non-entry-block bug.
#[test]
fn demo_rational_matches_native() {
    assert_demo_matches_cc("rational.c");
}

/// The jsmn JSON tokenizer (`demos/jsmn/jsmn.h`, MIT, vendored) — a different shape from Clay:
/// pure char/state-machine string scanning, zero allocations. Tokenizes a JSON string sandboxed
/// and prints the token types/spans; output must match a native `cc` build. (It needed no new
/// fixes — a clean validation that string parsing, escapes, nesting, and error paths work.)
#[test]
fn demo_jsmn_matches_native() {
    assert_demo_matches_cc("jsmn/jsmn_demo.c");
}

/// SHA-256 (B-Con's `crypto-algorithms`, public domain, vendored) — a pure integer/bit shape
/// (32-bit wrapping arithmetic, rotates-as-shifts, a round-key table). Hashes a few strings
/// sandboxed and prints the hex digests; must match a native `cc` build (and the standard test
/// vectors). The shakedown turned a `func_index` null-token crash into a clean error.
#[test]
fn demo_sha256_matches_native() {
    assert_demo_matches_cc("sha256/sha_demo.c");
}

/// xxHash (Cyan4973/xxHash, BSD-2-Clause, vendored) — XXH32/XXH64 in a self-contained scalar
/// build. Another integer/bit shape (multiply/rotate hashing); output matches a native `cc`
/// build and the standard test vectors. The shakedown added `_Static_assert` support.
#[test]
fn demo_xxhash_matches_native() {
    assert_demo_matches_cc("xxhash/xxh_demo.c");
}

/// tinfl (miniz, MIT, vendored) — miniz's standalone DEFLATE/zlib *inflate* engine. A new shape:
/// a coroutine-style inflate state machine (a deeply nested `switch` driven by `TINFL_CR_*`
/// macros), bit-buffer shifts, Huffman lookup tables, and a 32 KiB LZ77 dictionary inside the
/// `tinfl_decompressor` struct. Inflates an embedded zlib stream and writes the result; output
/// matches a native `cc` build byte-for-byte. It ran identically with no new fixes.
#[test]
fn demo_tinfl_matches_native() {
    assert_demo_matches_cc("tinfl/tinfl_demo.c");
}

/// stb_perlin (Sean Barrett, public domain, vendored) — the series' first floating-point-heavy
/// shakedown. Dense f32 arithmetic (gradient dot products, the quintic ease polynomial, trilinear
/// lerps), int<->float conversion, and multiply/accumulate chains over octaves. The driver prints
/// each noise value as a fixed-point integer, so any f32 divergence from native `cc` would show in
/// the digits; output matches byte-for-byte with no new fixes.
#[test]
fn demo_perlin_matches_native() {
    assert_demo_matches_cc("perlin/perlin_demo.c");
}

/// tiny-regex-c (kokke, Unlicense/public domain, vendored) — a Rob-Pike-style backtracking
/// matcher: `re_match` recurses through `matchpattern`/`matchstar`/`matchplus`, retrying on
/// failure. Recursion-with-backtracking is a new control-flow shape for the series (a workout for
/// data-stack threading + general goto/branch lowering). Runs a table of (pattern, text) cases and
/// prints match index/length; output matches a native `cc` build with no new fixes.
#[test]
fn demo_regex_matches_native() {
    assert_demo_matches_cc("regex/regex_demo.c");
}

/// heapgrow — the first demo to **consume the Memory capability**: a guest `malloc` (`vm_malloc.h`)
/// that grows the window into the reserved tail via `__vm_map` on demand. Allocates 1 MiB (~16× the
/// initial window) in eight blocks, sums them, and prints the totals — byte-identical to a native
/// `cc` build (which uses the real `malloc`). Exercises the powerbox Memory grant → `__vm_map`
/// builtin → `cap.call` → `mprotect` growth → masked tail access, the §1a sparse-address-space win.
#[test]
fn demo_heapgrow_matches_native() {
    assert_demo_matches_cc("heapgrow/heapgrow.c");
}

/// If the chibicc frontend is buildable, the CLI compiles and runs the C demo too — the same
/// greeting from C source. Skipped (not failed) when the toolchain is unavailable.
#[test]
fn cli_compiles_and_runs_c() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("hello.c"))
        .output()
        .expect("spawn svm-run");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("chibicc") {
            eprintln!(
                "note: skipping C demo (frontend unavailable): {}",
                err.trim()
            );
            return;
        }
        panic!("svm-run on hello.c failed: {err}");
    }
    assert_eq!(out.stdout, b"hello, sandbox!\n");
}

// ── §5 kill-path through the embedding entry (`run_powerbox_with_deadline`) ────────────────────

/// A runaway powerbox guest (ignores its handles, loops forever) is **detect-and-killed** at the
/// deadline rather than hanging the process — the CLI's `SVM_DEADLINE_MS` is this, end to end.
#[test]
fn deadline_kills_runaway_powerbox_guest() {
    let m = load(
        "func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i32.const 0\n\
         \x20 br block1(v3)\n\
         block1(v4: i32):\n\
         \x20 v5 = i32.const 1\n\
         \x20 v6 = i32.add v4 v5\n\
         \x20 br block1(v6)\n\
         }\n",
    );
    let err = run_powerbox_with_deadline(&m, b"", Some(Duration::from_millis(100)))
        .expect_err("a runaway guest must be killed, not returned");
    assert!(
        err.contains("OutOfFuel"),
        "expected an OutOfFuel detect-and-kill, got: {err}"
    );
}

/// §5 W3 — a trap's kill message carries the **source backtrace** when the module was built with
/// `-g`: a powerbox guest that divides by zero is detect-and-killed, and the error names the div's
/// `file:line:col in <fn>`. Cross-platform: the explicit-trap capture (`trap_capture.c`'s
/// `svm_capture_explicit_trap`, threaded the trapping frame pointer via `get_frame_pointer`) runs on
/// unix **and** windows, so the source frame is present on every target.
#[test]
fn trap_kill_message_carries_a_source_backtrace() {
    let m = load(
        "func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i32.const 0\n\
         \x20 v4 = i32.div_s v0 v3\n\
         \x20 return v4\n\
         }\n\
         debug.file 0 \"guest.c\"\n\
         debug.fname 0 \"divide\"\n\
         debug.loc 0 0 1 0 7 5\n",
    );
    let err = run_powerbox_with_deadline(&m, b"", None).expect_err("div-by-zero must be killed");
    assert!(err.contains("DivByZero"), "names the trap kind: {err}");
    assert!(
        err.contains("guest.c:7:5 in divide"),
        "the kill message carries the trap-time source backtrace + function name: {err}"
    );
}

/// §5 W3 — a **memory-fault** kill message carries the source backtrace on **every** platform: the
/// capture is the SIGSEGV/SIGBUS handler on unix and the Vectored Exception Handler on Windows, so an
/// out-of-bounds store that the §5 guard catches names the store's `file:line` cross-platform. (The
/// explicit-check-trap path — `trap_kill_message_carries_a_source_backtrace`, div-by-zero — is
/// unix-only; see ISSUES I3.) An 8-byte store at 65532 in a 64 KiB window overruns into the guard.
#[test]
fn memfault_kill_message_carries_a_source_backtrace() {
    let m = load(
        "memory 16\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 65532\n\
         \x20 v4 = i64.const 0\n\
         \x20 i64.store v3 v4\n\
         \x20 v5 = i32.const 0\n\
         \x20 return v5\n\
         }\n\
         debug.file 0 \"mem.c\"\n\
         debug.fname 0 \"store_oob\"\n\
         debug.loc 0 0 2 0 9 5\n",
    );
    let err = run_powerbox_with_deadline(&m, b"", None)
        .expect_err("the overrun must be detect-and-killed");
    assert!(err.contains("MemoryFault"), "names the trap kind: {err}");
    assert!(
        err.contains("mem.c:9:5 in store_oob"),
        "the kill message carries the trap-time source backtrace + function name: {err}"
    );
}

/// Arming the kill-path must not penalize a well-behaved guest: a fast program finishes normally —
/// and *quickly* — because the watchdog wakes the instant the run completes (it never blocks the
/// full deadline).
#[test]
fn deadline_does_not_delay_fast_guest() {
    let m = load(
        "memory 16\n\
         data 16 \"hi\\n\"\n\
         func (i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32):\n\
         \x20 v3 = i64.const 16\n\
         \x20 v4 = i64.const 3\n\
         \x20 v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)\n\
         \x20 v6 = i32.const 7\n\
         \x20 return v6\n\
         }\n",
    );
    let t0 = Instant::now();
    let run = run_powerbox_with_deadline(&m, b"", Some(Duration::from_secs(30))).expect("run");
    let elapsed = t0.elapsed();
    assert_eq!(run.stdout, b"hi\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(7)]));
    assert!(
        elapsed < Duration::from_secs(5),
        "the watchdog must not delay a fast run (took {elapsed:?} of a 30s deadline)"
    );
}

// ── §13/§14 region minting through the powerbox (the 5-handle grant + region factory) ──────────

/// A powerbox guest that takes the **5th** handle (an `AddressSpace`, §14) mints a `SharedRegion`,
/// maps it at two window offsets, and aliases through it — pinning that `run_powerbox` grants the
/// AddressSpace *and* installs the OS-shared-memory factory so a stock embedded guest can build the
/// zero-copy data plane (the same capability `<svm.h>` exposes to C). Host-granularity-agnostic: it
/// queries `region_page_size` (op 3) and works in whole granules.
#[test]
fn powerbox_region_minting_round_trips() {
    let m = load(
        "memory 17\n\
         func (i32, i32, i32, i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32):\n\
         \x20 v5 = i64.const 65536\n\
         \x20 v6 = cap.call 5 5 (i64) -> (i64) v4(v5)\n\
         \x20 v7 = i32.wrap_i64 v6\n\
         \x20 v8 = cap.call 4 3 () -> (i64) v7()\n\
         \x20 v9 = i64.const 0\n\
         \x20 v10 = i32.const 3\n\
         \x20 v11 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v7(v9, v9, v8, v10)\n\
         \x20 v12 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v7(v8, v9, v8, v10)\n\
         \x20 v13 = i32.const 123\n\
         \x20 i32.store8 v9 v13\n\
         \x20 v14 = i32.load8_u v8\n\
         \x20 return v14\n\
         }\n",
    );
    // The 5-param entry is still a recognized powerbox shape.
    assert!(is_powerbox_entry(&m));
    let run = run_powerbox(&m, b"").expect("run");
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I32(123)]),
        "the value stored through one mapping must read back through the alias"
    );
}

/// The capstone for the §5 kill-path: drive the **`svm-run` binary** on a C `for(;;){}` compiled by
/// the frontend, with `SVM_DEADLINE_MS` set — it must be detect-and-killed (non-zero exit, an
/// `OutOfFuel` message) instead of hanging the process. The real end-to-end product path: C source →
/// frontend → JIT → watchdog → CLI exit. Skipped (not failed) when the frontend is unavailable.
#[test]
fn cli_deadline_kills_runaway_c_program() {
    let cfile = std::env::temp_dir().join(format!("svm_runaway_{}.c", std::process::id()));
    std::fs::write(&cfile, "int main(void){ for(;;){} return 0; }\n").expect("write temp C");
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(&cfile)
        .env("SVM_DEADLINE_MS", "200")
        .output()
        .expect("spawn svm-run");
    let _ = std::fs::remove_file(&cfile);
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("chibicc") {
        eprintln!("note: skipping (frontend unavailable): {}", err.trim());
        return;
    }
    assert!(
        !out.status.success(),
        "a detect-and-killed guest must exit non-zero; stderr: {err}"
    );
    assert!(
        err.contains("OutOfFuel"),
        "expected an OutOfFuel detect-and-kill on the CLI; stderr: {err}"
    );
}

/// The guest-built **M:N green-thread scheduler** demo (`demos/mn_sched`), end to end through the
/// `svm-run` binary: 4 worker threads each cooperatively scheduling 8 fibers over the VM's
/// primitives — the scheduler is entirely guest code (D56/D57). Must print the interleaving-
/// invariant total `1024`. The interp↔JIT differential lives in `c_frontend::c_guest_mn_scheduler_demo`;
/// this is the product-path smoke test. Skipped (not failed) when the frontend is unavailable.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_mn_scheduler_runs() {
    // Fail-fast like the work-stealing siblings (ISSUES.md I7): a threaded/fiber scheduler is the
    // same wedge class, so don't let a hang block on a bare unbounded `.output()`.
    let out = run_demo_failfast("mn_sched/mn_sched.c");
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("chibicc") {
        eprintln!(
            "note: skipping mn_sched demo (frontend unavailable): {}",
            err.trim()
        );
        return;
    }
    assert!(out.status.success(), "svm-run on mn_sched failed: {err}");
    assert_eq!(out.stdout, b"1024\n", "guest M:N scheduler total");
}

/// The guest-built **work-stealing** M:N scheduler demo (`demos/work_stealing`, stackless tasks),
/// end to end through the `svm-run` binary — must print the interleaving-invariant total `256`. The
/// interp↔JIT differential lives in `c_frontend::c_guest_work_stealing_demo`; this is the
/// product-path smoke test. Skipped (not failed) when the frontend is unavailable.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_work_stealing_runs() {
    let out = run_demo_failfast("work_stealing/work_stealing.c");
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("chibicc") {
        eprintln!(
            "note: skipping work_stealing demo (frontend unavailable): {}",
            err.trim()
        );
        return;
    }
    assert!(
        out.status.success(),
        "svm-run on work_stealing failed: {err}"
    );
    assert_eq!(out.stdout, b"256\n", "guest work-stealing scheduler total");
}

/// **Demo 3** — the guest-built work-stealing scheduler over **stackful, migratable fibers**
/// (`demos/steal_fibers`, D57 complete), end to end through the `svm-run` binary: suspended
/// fibers are stolen across real OS threads and must print both interleaving-invariant totals
/// (`256` work units; `121920` = the sum of returns whose values depend on locals carried across
/// every migration — the stack-integrity check). The interp↔JIT differential lives in
/// `c_frontend::c_guest_steal_fibers_demo`; this is the product-path smoke test. Skipped (not
/// failed) when the frontend is unavailable.
#[test]
#[cfg(all(unix, target_arch = "x86_64"))]
fn demo_steal_fibers_runs() {
    let out = run_demo_failfast("steal_fibers/steal_fibers.c");
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("chibicc") {
        eprintln!(
            "note: skipping steal_fibers demo (frontend unavailable): {}",
            err.trim()
        );
        return;
    }
    assert!(
        out.status.success(),
        "svm-run on steal_fibers failed: {err}"
    );
    assert_eq!(
        out.stdout, b"256\n121920\n",
        "stackful work-stealing scheduler totals"
    );
}

/// The **threaded guest-driven JIT** demo (`demos/jit/jit_threads.c`, DESIGN.md §22), end to end
/// through the `svm-run` binary: 4 worker threads each Cranelift-compile a distinct unit
/// **concurrently** (several `Jit.compile`s in flight, serialized through the per-domain
/// `Mutex<Host>` the powerbox engages for a `thread.spawn`ing guest) and invoke the native code,
/// checking each against a C reference. Must print `0` (no input mismatches across any worker) — the
/// product-path proof that concurrent guest JIT compilation is sound. Gated to the `fiber_rt` targets
/// (elsewhere `thread.spawn` is `Unsupported`). Skipped (not failed) when the frontend is unavailable.
#[test]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
fn demo_jit_threads_runs() {
    let out = Command::new(env!("CARGO_BIN_EXE_svm-run"))
        .arg(demo("jit/jit_threads.c"))
        .output()
        .expect("spawn svm-run");
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("chibicc") {
        eprintln!(
            "note: skipping jit_threads demo (frontend unavailable): {}",
            err.trim()
        );
        return;
    }
    assert!(out.status.success(), "svm-run on jit_threads failed: {err}");
    assert_eq!(
        out.stdout, b"0\n",
        "every worker's concurrently-JITed unit must agree with the reference"
    );
}

/// §15 the embedder-facing spawn quota (`run_powerbox_with_deadline_and_quota`) is enforced
/// end-to-end on the JIT: a powerbox guest that spawns a vCPU is **detect-and-killed** under a
/// `max_vcpus = 1` quota (the root fills it), and runs under the default. Gated to the targets where
/// the JIT thread runtime exists (elsewhere `thread.spawn` is `Unsupported`, a different `Err`).
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
#[test]
fn quota_contains_a_powerbox_thread_bomb() {
    // A 3-handle powerbox entry (stdout, stdin, exit) that just spawns a vCPU and returns.
    let src = "memory 16\n\
        func (i32, i32, i32) -> () {\n\
        block0(vout: i32, vin: i32, vexit: i32):\n\
        \x20 v0 = i64.const 5\n\
        \x20 v1 = thread.spawn 1 v0 v0\n\
        \x20 v2 = thread.join v1\n\
        \x20 return\n\
        }\n\
        func (i64, i64) -> (i64) {\n\
        block0(vsp: i64, varg: i64):\n\
        \x20 return varg\n\
        }\n";
    let m = load(src);
    assert!(is_powerbox_entry(&m));

    // max_vcpus = 1 ⇒ the root alone fills the quota; the spawn detect-and-kills (Err).
    let tight = Quota {
        max_fibers: 1 << 16,
        max_vcpus: 1,
    };
    let r = run_powerbox_with_deadline_and_quota(&m, b"", None, tight);
    assert!(
        r.is_err(),
        "a spawn over the quota must detect-and-kill, got {r:?}"
    );

    // The default quota admits the spawn+join.
    let r = run_powerbox_with_deadline_and_quota(&m, b"", None, Quota::default());
    assert!(
        r.is_ok(),
        "the default quota must run the program, got {r:?}"
    );
}

// ----------------------------------------------------------------------------
// §7 named capability imports (late binding): a frontend declares `extern`-style
// capability imports by name; the host resolves each to a concrete `cap.call` at load.
// ----------------------------------------------------------------------------

#[test]
fn named_imports_resolve_and_run_like_inline_capcalls() {
    // The same program as `writes_to_stdout_and_returns`/`exit_capability_sets_code`, but the
    // capabilities are reached by NAME (`write`, `exit`) instead of inline `cap.call 0 1`/`1 0`.
    // The handle is still supplied by the call site (v0 = stdout, v2 = exit); the host policy
    // binds only the (type_id, op). resolve_capability_imports lowers them to the same cap.calls.
    let src = "memory 16\n\
        import 0 \"write\" (i64, i64) -> (i64)\n\
        import 1 \"exit\" (i32) -> ()\n\
        data 16 \"hi\\n\"\n\
        func (i32, i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32, v2: i32):\n\
        \x20 v3 = i64.const 16\n\
        \x20 v4 = i64.const 3\n\
        \x20 v5 = call.import 0 v0 (v3, v4)\n\
        \x20 v6 = i32.const 0\n\
        \x20 call.import 1 v2 (v6)\n\
        \x20 unreachable\n\
        }\n";
    let m = parse_module(src).expect("parse text IR with imports");
    assert_eq!(m.imports.len(), 2, "two named imports declared");
    // Resolve under the reference host policy, then verify + run.
    let resolved = svm_run::resolve_capability_imports(m).expect("resolve imports");
    assert!(resolved.imports.is_empty(), "imports must be lowered away");
    verify_module(&resolved).expect("verify resolved module");
    assert!(is_powerbox_entry(&resolved));
    let run = run_powerbox(&resolved, b"").expect("run");
    assert_eq!(run.stdout, b"hi\n", "write import produced stdout");
    assert_eq!(run.outcome, Outcome::Exited(0), "exit import set the code");
}

#[test]
fn unknown_named_import_fails_closed() {
    // A capability name the host policy doesn't know is a clean load error — never a silent
    // no-op or a wrong call.
    let src = "func (i32, i32, i32) -> (i32) {\n\
        block0(v0: i32, v1: i32, v2: i32):\n\
        \x20 v3 = i64.const 0\n\
        \x20 v4 = call.import 0 v0 (v3)\n\
        \x20 v5 = i32.const 0\n\
        \x20 return v5\n\
        }\n";
    // Declare the unknown import at the top.
    let src = format!("import 0 \"frobnicate\" (i64) -> (i64)\n{src}");
    let m = parse_module(&src).expect("parse");
    let err = svm_run::resolve_capability_imports(m).expect_err("unknown import must fail closed");
    assert!(
        err.contains("frobnicate"),
        "error names the bad import: {err}"
    );
}
