//! Host-side **differential corpus** generator. For each guest module it (1) encodes the module to
//! its `svm-encode` binary form under `corpus/`, and (2) computes the **native** bytecode-engine
//! result for a set of args — the ground truth `corpus.mjs` checks the wasm `svm_run` against.
//!
//! The native run here uses the *exact same* `bytecode::compile_and_run` the wasm entry calls, so a
//! mismatch isolates a wasm-compilation / sandbox effect (not an engine difference). The repo
//! already gates the bytecode engine against the tree-walker oracle (`bytecode_diff.rs`); this gates
//! the *wasm build of it* against the native build.
//!
//! Status codes mirror `lib.rs`: 0 OK · 2 UNSUPPORTED (`None`) · 3 TRAP (`Err`) · 4 BAD_RESULT.

use std::io::Write;

use svm_browser::{capture_exec, powerbox_exec};
use svm_interp::{bytecode, Value};

// Three op-family kernels lifted verbatim from `crates/svm/tests/bytecode_diff.rs` (known parseable
// and engine-supported), plus a divide-by-zero trap kernel.
const ALU: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br block1(v7, v14, v16)
block3(v17: i64):
  return v17
}
"#;

const CALL: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = call 1 (v8, v9)
  v11 = i64.const 1
  v12 = i64.add v9 v11
  br block1(v7, v10, v12)
block3(v13: i64):
  return v13
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.add v0 v1
  return v2
}
"#;

const MEM: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.const 0
  br block1(v0, v1, v2)
block1(v3: i64, v4: i64, v5: i64):
  v6 = i64.lt_s v5 v3
  br_if v6 block2(v3, v4, v5) block3(v4)
block2(v7: i64, v8: i64, v9: i64):
  v10 = i64.const 8
  i64.store v10 v8
  v11 = i64.load v10
  v12 = i64.add v11 v9
  v13 = i64.const 1
  v14 = i64.add v9 v13
  br block1(v7, v12, v14)
block3(v15: i64):
  return v15
}
"#;

const DIVTRAP: &str = r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i64.div_s v0 v1
  return v2
}
"#;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× — total exactly 4000 on every interleaving.
// Lifted from `crates/svm/tests/bytecode_threads.rs`; exercises `thread.spawn`/`join` + atomics on
// the bytecode engine's cooperative `drive` (the browser concurrency model). Takes no args.
const THREADS: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

// ---- powerbox guests: exercise the real capability set (streams / clock / exit) ----------------
// Granted by entry arity (see `powerbox_exec`): 1 Stream(Out) · 2 Stream(In) · 3 Exit ·
// 4 Stream(Err) · 5 Clock. I/O is deterministic (stdout/stderr buffers, monotonic clock), so the
// native result here is an exact ground truth for the wasm `svm_run_pb`.

// `(out, in, exit)`: write a fixed 17-byte greeting to stdout via Stream.write (type 0, op 1).
const PB_HELLO: &str = r#"
memory 16
data 0 "hello, powerbox!\n"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 17
  v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)
  v6 = i32.const 0
  return v6
}
"#;

// `(out, in, exit)`: read up to 256 bytes of stdin (type 0, op 0) into the window, echo them back to
// stdout (type 0, op 1) — a full host→guest→host roundtrip through the buffers.
const PB_ECHO: &str = r#"
memory 16
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 256
  v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)
  v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)
  v7 = i32.const 0
  return v7
}
"#;

// `(out, in, exit, err, clock)`: read the monotonic clock twice (type 2, op 0) and return the delta
// — exactly 1, proving the deterministic strictly-increasing counter works under wasm.
const PB_CLOCK: &str = r#"
func (i32, i32, i32, i32, i32) -> (i64) {
block0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32):
  v5 = i32.const 0
  v6 = cap.call 2 0 (i32) -> (i64) v4(v5)
  v7 = cap.call 2 0 (i32) -> (i64) v4(v5)
  v8 = i64.sub v7 v6
  return v8
}
"#;

// `(out, in, exit)`: call Exit.exit(42) (type 1, op 0) — a non-error trap surfaced as STATUS_EXIT
// with exit code 42; the trailing return is unreachable.
const PB_EXIT: &str = r#"
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i32.const 42
  cap.call 1 0 (i32) -> () v2(v3)
  v4 = i32.const 0
  return v4
}
"#;

// `(out, in, exit, err)`: write a 9-byte message to **stderr** (type 0, op 1, on the Err handle) —
// proving role routing (Out → stdout, Err → stderr).
const PB_STDERR: &str = r#"
memory 16
data 0 "warning!\n"
func (i32, i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32, v3: i32):
  v4 = i64.const 0
  v5 = i64.const 9
  v6 = cap.call 0 1 (i64, i64) -> (i64) v3(v4, v5)
  v7 = i32.const 0
  return v7
}
"#;

// Live-import guest (encoded for `live.mjs`, not part of the deterministic corpus): `(console,
// clock)` are host-fn caps (iface 13) the live cdylib bridges to real wasm imports. Writes a 16-byte
// line to stdout via `console.write(stream=0, ptr, len)`, then returns `clock.now()` — so `live.mjs`
// asserts the bytes reached the host import and the host clock value flowed back to the guest.
const LIVE_GUEST: &str = r#"
memory 16
data 0 "live from wasm!\n"
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = i64.const 16
  v5 = cap.call 13 1 (i64, i64, i64) -> (i64) v0(v2, v3, v4)
  v6 = cap.call 13 0 () -> (i64) v1()
  return v6
}
"#;

// Large-I/O echo guest (encoded for `corpus.mjs`'s alloc-ABI roundtrip, not the corpus): a 4 MiB
// window, reads up to 4 MiB of stdin and echoes it to stdout — used to push **megabytes** through
// `svm_alloc`ed buffers, well past the old fixed 1 MiB scratch cap.
const BIG_ECHO: &str = r#"
memory 22
func (i32, i32, i32) -> (i32) {
block0(v0: i32, v1: i32, v2: i32):
  v3 = i64.const 0
  v4 = i64.const 4194304
  v5 = cap.call 0 0 (i64, i64) -> (i64) v1(v3, v4)
  v6 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v5)
  v7 = i32.const 0
  return v7
}
"#;

// Memory-snapshot guest: the window is seeded with 16 little-endian i64 words; the guest adds `arg`
// to each in place and returns word 0's new value. The captured final image (128 bytes) is the
// interesting output — the "host hands in a buffer, guest transforms it in place" embedder shape.
const CAP_ADDK: &str = r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  br block1(v0, v1)
block1(v2: i64, v3: i64):
  v4 = i64.const 128
  v5 = i64.lt_u v3 v4
  br_if v5 block2(v2, v3) block3()
block2(v6: i64, v7: i64):
  v8 = i64.load v7
  v9 = i64.add v8 v6
  i64.store v7 v9
  v10 = i64.const 8
  v11 = i64.add v7 v10
  br block1(v6, v11)
block3():
  v12 = i64.const 0
  v13 = i64.load v12
  return v13
}
"#;

/// Lowercase-hex encode (corpus.json carries stdin/stdout/stderr as hex to stay escaping-free).
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Args fed to each kernel (all `(i64) -> (i64)`), incl. negatives and a large value.
const ARGS: &[i64] = &[0, 1, 2, 5, 64, 1000, -1, -1000, 100_000];

fn native(m: &svm_ir::Module, args: &[Value]) -> (i32, i64) {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, 0, args, &mut fuel) {
        None => (2, 0),
        Some(Err(_)) => (3, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (0, *x),
            _ => (4, 0),
        },
    }
}

/// Encode a text module to `corpus/<name>.svmbc` and return the parsed module + file path.
fn emit(name: &str, src: &str) -> (svm_ir::Module, String) {
    let m = svm_text::parse_module(src).unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
    let bytes = svm_encode::encode_module(&m);
    let file = format!("corpus/{name}.svmbc");
    std::fs::File::create(&file)
        .and_then(|mut f| f.write_all(&bytes))
        .expect("write module");
    eprintln!("{name}: {} bytes", bytes.len());
    (m, file)
}

fn main() {
    // Compute corpus — (name, source, nargs): nargs==1 sweeps `ARGS`; nargs==0 runs once, no arg.
    let compute = [
        ("alu", ALU, 1u32),
        ("call", CALL, 1),
        ("mem", MEM, 1),
        ("divtrap", DIVTRAP, 1),
        ("threads", THREADS, 0),
    ];
    // Powerbox corpus — (name, source, stdin): each runs once under the real capability set.
    let powerbox = [
        ("pb_hello", PB_HELLO, &b""[..]),
        ("pb_echo", PB_ECHO, &b"ping\n"[..]),
        ("pb_clock", PB_CLOCK, &b""[..]),
        ("pb_exit", PB_EXIT, &b""[..]),
        ("pb_stderr", PB_STDERR, &b""[..]),
    ];
    std::fs::create_dir_all("corpus").expect("mkdir corpus");

    let mut json = String::from("{\n\"compute\":[\n");
    for (i, (name, src, nargs)) in compute.iter().enumerate() {
        let (m, file) = emit(name, src);
        // args sweep for 1-arg kernels; a single no-arg case otherwise.
        let args: &[i64] = if *nargs == 1 { ARGS } else { &[0] };
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"nargs\":{nargs},\"cases\":["
        ));
        for (j, &arg) in args.iter().enumerate() {
            let call_args: &[Value] = if *nargs == 1 { &[Value::I64(arg)] } else { &[] };
            let (status, value) = native(&m, call_args);
            // i64s as JSON strings so JS keeps full precision (BigInt).
            json.push_str(&format!(
                "{}{{\"arg\":\"{arg}\",\"status\":{status},\"value\":\"{value}\"}}",
                if j == 0 { "" } else { "," }
            ));
        }
        json.push_str(if i + 1 == compute.len() { "]}\n" } else { "]},\n" });
    }
    json.push_str("],\n\"powerbox\":[\n");
    for (i, (name, src, stdin)) in powerbox.iter().enumerate() {
        let (m, file) = emit(name, src);
        // Native ground truth via the *same* `powerbox_exec` the wasm `svm_run_pb` calls.
        let out = powerbox_exec(&m, stdin);
        json.push_str(&format!(
            "  {{\"name\":\"{name}\",\"file\":\"{file}\",\"stdin\":\"{}\",\"status\":{},\
             \"value\":\"{}\",\"exit\":{},\"stdout\":\"{}\",\"stderr\":\"{}\"}}{}",
            hex(stdin),
            out.status,
            out.value,
            out.exit_code,
            hex(&out.stdout),
            hex(&out.stderr),
            if i + 1 == powerbox.len() { "\n" } else { ",\n" },
        ));
    }
    // Capture corpus — a window seeded with 16 i64 words (word i = i*1000), the addk guest run for
    // each arg; the captured final image is the ground truth.
    json.push_str("],\n\"capture\":[\n");
    let (cap_m, cap_file) = emit("cap_addk", CAP_ADDK);
    let mut init = Vec::new();
    for i in 0..16i64 {
        init.extend_from_slice(&(i * 1000).to_le_bytes());
    }
    let cap_args: &[i64] = &[0, 42, -1];
    for (k, &arg) in cap_args.iter().enumerate() {
        let out = capture_exec(&cap_m, &init, arg);
        json.push_str(&format!(
            "  {{\"name\":\"cap_addk\",\"file\":\"{cap_file}\",\"init\":\"{}\",\"arg\":\"{arg}\",\
             \"status\":{},\"value\":\"{}\",\"snapshot\":\"{}\"}}{}",
            hex(&init),
            out.status,
            out.value,
            hex(&out.snapshot),
            if k + 1 == cap_args.len() { "\n" } else { ",\n" },
        ));
    }
    json.push_str("]\n}\n");
    std::fs::write("corpus.json", json).expect("write corpus.json");
    eprintln!(
        "wrote corpus.json ({} compute, {} powerbox, {} capture)",
        compute.len(),
        powerbox.len(),
        cap_args.len()
    );

    // Encode the guests validated by harnesses (not the deterministic corpus): the live-import guest
    // (`live.mjs`, host-backed) and the large-I/O echo guest (`corpus.mjs`'s alloc-ABI roundtrip).
    emit("live", LIVE_GUEST);
    emit("bigecho", BIG_ECHO);
}
