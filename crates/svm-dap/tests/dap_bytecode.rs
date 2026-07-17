//! DAP-over-bytecode parity (DEBUGGING.md G3). The same scripted DAP conversation is replayed against
//! both backends — the default tree-walker `Inspector` and the `"engine":"bytecode"` `BytecodeBackend`
//! (the engine the browser playground runs) — and the debugger observations must be **identical**:
//! breakpoint binding, the stop reason, the source frame, and the inspected locals at **every** loop
//! iteration (resume included), through to termination. This is the server-side proof that the backend
//! seam dispatches the forward-debug subset to either engine with the same result — the runtime
//! counterpart of `crates/svm/tests/debug_parity.rs`'s engine-level checks.

use svm_dap::{DapServer, Json};

// LOOP_SUM with a §6/W4 debug section: a source location at the loop body (sum.c:7) and the two loop
// variables mapped to their block-relative SSA value indices. Same fixture as the tree-walker suite.
const LOOP_SUM_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}

debug.file 0 "sum.c"
debug.fname 0 "sum"
debug.loc 0 1 0 0 7 5
debug.var 0 "i" ssa 0 "int"
debug.var 0 "acc" ssa 1 "int"
"#;

fn req(seq: i64, command: &str, args: Json) -> Json {
    Json::obj(vec![
        ("seq", Json::i(seq)),
        ("type", Json::s("request")),
        ("command", Json::s(command)),
        ("arguments", args),
    ])
}

fn response(msgs: &[Json]) -> &Json {
    msgs.iter()
        .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .expect("a response")
}

fn event<'a>(msgs: &'a [Json], name: &str) -> Option<&'a Json> {
    msgs.iter().find(|m| {
        m.get("type").and_then(|t| t.as_str()) == Some("event")
            && m.get("event").and_then(|e| e.as_str()) == Some(name)
    })
}

/// What the debugger reported over one full session — the fields that must match across engines.
#[derive(Debug, PartialEq)]
struct Observations {
    bp_verified: bool,
    bp_line: i64,
    first_reason: String,
    top_frame_name: String,
    top_frame_line: i64,
    /// `(i, acc)` read at each successive breakpoint hit, innermost frame — proves resume/loop parity.
    iter_locals: Vec<(String, String)>,
    terminated: bool,
}

/// Read the innermost frame's `(i, acc)` at the current stop via stackTrace → scopes → variables.
fn read_locals(s: &mut DapServer, seq: i64) -> (String, String) {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    let top = &frames.as_array().unwrap()[0];
    let frame_id = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
    ));
    let scope = &response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let vref = scope.get("variablesReference").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let map: std::collections::HashMap<&str, &str> = vars
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap(),
                v.get("value").unwrap().as_str().unwrap(),
            )
        })
        .collect();
    (
        map.get("i").copied().unwrap_or("?").to_string(),
        map.get("acc").copied().unwrap_or("?").to_string(),
    )
}

/// Run the whole scripted session on `engine` (`None` ⇒ the default tree-walker) and record what the
/// debugger reported.
fn run_session(engine: Option<&str>) -> Observations {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));

    let mut launch_args = vec![
        ("programText", Json::s(LOOP_SUM_DBG)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![Json::i(3)])),
    ];
    if let Some(e) = engine {
        launch_args.push(("engine", Json::s(e)));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch_args)));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok ({engine:?})"
    );

    // Breakpoint on the loop body (sum.c:7).
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/work/sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    let bp0 = &response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let bp_verified = bp0.get("verified") == Some(&Json::Bool(true));
    let bp_line = bp0.get("line").and_then(|l| l.as_i64()).unwrap_or(-1);

    // Run to the first hit.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stopped = event(&out, "stopped").expect("stops at the breakpoint");
    let first_reason = stopped
        .get("body")
        .unwrap()
        .get("reason")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // First frame + locals.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let top = &response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let top_frame_name = top.get("name").unwrap().as_str().unwrap().to_string();
    let top_frame_line = top.get("line").unwrap().as_i64().unwrap();

    let mut iter_locals = vec![read_locals(&mut s, 6)];

    // Continue through every remaining hit, recording locals each time, until the guest terminates.
    let mut terminated = false;
    for seq in 10..200 {
        let out = s.handle(&req(seq, "continue", Json::obj(vec![])));
        if event(&out, "terminated").is_some() {
            terminated = true;
            break;
        }
        if event(&out, "stopped").is_some() {
            iter_locals.push(read_locals(&mut s, seq));
        }
    }

    Observations {
        bp_verified,
        bp_line,
        first_reason,
        top_frame_name,
        top_frame_line,
        iter_locals,
        terminated,
    }
}

#[test]
fn dap_over_bytecode_matches_the_tree_walker() {
    let treewalk = run_session(None);
    let bytecode = run_session(Some("bytecode"));

    // Absolute sanity: the bytecode backend binds the line, stops for a breakpoint, shows `#0 sum` at
    // line 7, reads i=3/acc=0 at the first hit, and terminates.
    assert!(bytecode.bp_verified, "bytecode: breakpoint binds");
    assert_eq!(bytecode.bp_line, 7);
    assert_eq!(bytecode.first_reason, "breakpoint");
    assert_eq!(bytecode.top_frame_name, "#0 sum");
    assert_eq!(bytecode.top_frame_line, 7);
    assert_eq!(
        bytecode.iter_locals.first(),
        Some(&("3".to_string(), "0".to_string()))
    );
    assert!(bytecode.terminated, "bytecode: the guest terminates");
    assert!(
        bytecode.iter_locals.len() >= 3,
        "loops at least 3 iterations"
    );

    // The load-bearing claim: the two engines report *identical* debugger observations — breakpoint,
    // stop reason, frame, and the (i, acc) at every iteration.
    assert_eq!(
        bytecode, treewalk,
        "DAP-over-bytecode ≡ DAP-over-tree-walker"
    );
}

// A program with NO explicit debug section — the server synthesizes one (line table + SSA names) so
// hand-written SVM text is debuggable. Same sum loop, sans the `debug.*` directives.
const PLAIN_SUM: &str = r#"
func () -> (i64) {
block0():
  vn = i64.const 5
  vacc0 = i64.const 0
  br block1(vn, vacc0)
block1(vi: i64, vacc: i64):
  vsum = i64.add vacc vi
  vone = i64.const 1
  vnext = i64.sub vi vone
  br_if vnext block1(vnext, vsum) block2(vsum)
block2(vr: i64):
  return vr
}
"#;

#[test]
fn dap_over_bytecode_debugs_plain_svm_via_synthesized_debug_info() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(PLAIN_SUM)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // A breakpoint on the loop body (line 8, `vsum = i64.add`) binds against the synthesized table.
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.svm"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(8))])]),
            ),
        ]),
    ));
    let bp0 = &response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap()
        .as_array()
        .unwrap()[0];
    assert_eq!(
        bp0.get("verified"),
        Some(&Json::Bool(true)),
        "breakpoint binds on plain SVM"
    );
    assert_eq!(bp0.get("line"), Some(&Json::i(8)));

    // Run to it; the loop variables read back by their **text names** (i / acc), not v-indices.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(
        event(&out, "stopped").is_some(),
        "stops at the synthesized breakpoint"
    );
    let (i, acc) = read_named(&mut s, 5, "vi", "vacc");
    assert_eq!(
        (i.as_str(), acc.as_str()),
        ("5", "0"),
        "vi=5, vacc=0 at the first hit"
    );
}

/// Read two named locals at the current stop (innermost frame).
fn read_named(s: &mut DapServer, seq: i64, a: &str, b: &str) -> (String, String) {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let top = &response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let fid = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    let vref = response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        seq,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let map: std::collections::HashMap<&str, &str> = vars
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap(),
                v.get("value").unwrap().as_str().unwrap(),
            )
        })
        .collect();
    (
        map.get(a).copied().unwrap_or("?").to_string(),
        map.get(b).copied().unwrap_or("?").to_string(),
    )
}

#[test]
fn dap_over_bytecode_refuses_reverse_debugging() {
    // The bytecode backend is forward-only: stepBack / reverseContinue fail cleanly (the browser's
    // client hides those controls, but the server must not pretend to honor them).
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    let back = s.handle(&req(4, "stepBack", Json::obj(vec![])));
    assert_eq!(
        response(&back).get("success"),
        Some(&Json::Bool(false)),
        "stepBack refused"
    );
    let rev = s.handle(&req(5, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        response(&rev).get("success"),
        Some(&Json::Bool(false)),
        "reverseContinue refused"
    );
}
