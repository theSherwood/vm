//! End-to-end Debug Adapter Protocol conversation against the interpreter-backed server
//! (DEBUGGING.md W5): the acceptance loop — set a source-line breakpoint, hit it, see the source
//! frame and inspectable locals — scripted as DAP requests, no editor needed.

use svm_dap::{DapServer, Json};

// LOOP_SUM with a hand-written §6/W4 debug section: a source location at the loop body (sum.c:7)
// and the two loop variables mapped to their block-relative SSA value indices.
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

/// The single response message in a handle() result (type == "response").
fn response(msgs: &[Json]) -> &Json {
    msgs.iter()
        .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .expect("a response")
}

/// The first event with the given name, if any.
fn event<'a>(msgs: &'a [Json], name: &str) -> Option<&'a Json> {
    msgs.iter().find(|m| {
        m.get("type").and_then(|t| t.as_str()) == Some("event")
            && m.get("event").and_then(|e| e.as_str()) == Some(name)
    })
}

#[test]
fn dap_breakpoint_hit_shows_source_frame_and_locals() {
    let mut s = DapServer::new();

    // initialize → success + an `initialized` event telling the client to send breakpoints.
    let out = s.handle(&req(1, "initialize", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    assert!(
        event(&out, "initialized").is_some(),
        "initialize emits `initialized`"
    );

    // launch the guest (inline IR), entry func 0, arg 3.
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // setBreakpoints on sum.c:7 → binds (verified) to line 7.
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
    let bps = response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap();
    let bp0 = &bps.as_array().unwrap()[0];
    assert_eq!(
        bp0.get("verified"),
        Some(&Json::Bool(true)),
        "breakpoint binds"
    );
    assert_eq!(bp0.get("line"), Some(&Json::i(7)));

    // configurationDone runs to the breakpoint → a `stopped` event (reason breakpoint).
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stopped = event(&out, "stopped").expect("stops at the breakpoint");
    assert_eq!(
        stopped.get("body").unwrap().get("reason"),
        Some(&Json::s("breakpoint"))
    );

    // stackTrace shows the source frame at sum.c:7.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    let top = &frames.as_array().unwrap()[0];
    assert_eq!(top.get("line"), Some(&Json::i(7)));
    assert_eq!(
        top.get("source").unwrap().get("name"),
        Some(&Json::s("sum.c"))
    );
    // The frame carries the `-g` function name, so the VS Code call stack reads `#0 sum`, not `func0`.
    assert_eq!(top.get("name"), Some(&Json::s("#0 sum")));

    // scopes → a Locals scope; variables on it shows i = 3 and acc = 0 (first iteration).
    let frame_id = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        6,
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
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let named: std::collections::HashMap<&str, &str> = vars
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
    assert_eq!(named.get("i"), Some(&"3"), "local i = 3");
    assert_eq!(named.get("acc"), Some(&"0"), "local acc = 0");

    // continue eventually runs to completion → a `terminated` event.
    let mut terminated = false;
    for seq in 8..40 {
        let out = s.handle(&req(seq, "continue", Json::obj(vec![])));
        if event(&out, "terminated").is_some() {
            terminated = true;
            break;
        }
    }
    assert!(terminated, "the guest eventually terminates");
    assert!(s.is_terminated());
}

#[test]
fn dap_json_round_trips() {
    // The wire codec handles nested objects, arrays, strings with escapes, and integers.
    let v = Json::obj(vec![
        ("a", Json::i(42)),
        ("s", Json::s("a\"b\\c\n")),
        ("arr", Json::Arr(vec![Json::Bool(true), Json::Null])),
    ]);
    let text = v.to_string();
    assert_eq!(svm_dap::parse(&text), Some(v));
}

/// Read the top frame's locals as name→value (stackTrace → scopes → variables).
fn locals(s: &mut DapServer) -> std::collections::HashMap<String, String> {
    let out = s.handle(&req(
        900,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    let fid = frames.as_array().unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        901,
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
        902,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap().to_string(),
                v.get("value").unwrap().as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// Reverse debugging over DAP (DEBUGGING.md W1 time-travel): the loop-body breakpoint hits three
/// times (i = 3, 2, 1); `reverseContinue` walks *backward* through those hits, and `stepBack`
/// reverse-single-steps — all via `Inspector::seek`/`step_back`.
#[test]
fn dap_reverse_debugging_walks_back_through_breakpoint_hits() {
    let mut s = DapServer::new();
    let out = s.handle(&req(1, "initialize", Json::obj(vec![])));
    assert_eq!(
        response(&out).get("body").unwrap().get("supportsStepBack"),
        Some(&Json::Bool(true)),
        "advertises reverse debugging so the client enables the controls"
    );
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));

    // First hit (i=3, acc=0), then forward to the third hit (i=1, acc=5).
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(locals(&mut s).get("i").map(String::as_str), Some("3"));
    s.handle(&req(5, "continue", Json::obj(vec![])));
    s.handle(&req(6, "continue", Json::obj(vec![])));
    let l = locals(&mut s);
    assert_eq!(
        (l["i"].as_str(), l["acc"].as_str()),
        ("1", "5"),
        "third hit"
    );

    // reverseContinue → the *previous* breakpoint hit (i=2, acc=3).
    let out = s.handle(&req(7, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("breakpoint"))
    );
    let l = locals(&mut s);
    assert_eq!(
        (l["i"].as_str(), l["acc"].as_str()),
        ("2", "3"),
        "reversed to the 2nd hit"
    );

    // reverseContinue again → the first hit (i=3, acc=0).
    s.handle(&req(8, "reverseContinue", Json::obj(vec![])));
    let l = locals(&mut s);
    assert_eq!(
        (l["i"].as_str(), l["acc"].as_str()),
        ("3", "0"),
        "reversed to the 1st hit"
    );

    // No earlier breakpoint → reverseContinue rewinds to the start (entry).
    let out = s.handle(&req(9, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("entry")),
        "rewinds to the start when there is no earlier breakpoint"
    );

    // stepBack at the start is a no-op move that still reports a `step` stop; session stays live.
    let out = s.handle(&req(10, "stepBack", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("step"))
    );
    assert!(!s.is_terminated());
}

// Two worker threads (func 1) over a shared counter, with a §6/W4 debug section: the worker's load
// op maps to worker.c:4 and its argument is the source variable `delta`.
const WORKERS_DBG: &str = r#"
memory 16
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.load vaddr
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
}

debug.file 0 "worker.c"
debug.loc 1 0 1 0 4 3
debug.var 1 "delta" ssa 1 "long"
"#;

/// Multithreaded debugging over DAP (DEBUGGING.md Milestone B): a scheduled launch surfaces every
/// `thread.spawn` vCPU as a DAP thread; a source breakpoint in the worker fires per-thread, the
/// `stopped` event names which thread, and `stackTrace`/`variables` inspect a chosen thread's frame.
#[test]
fn dap_multithreaded_threads_per_thread_stacks_and_which_stopped() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    // `schedule: []` ⇒ a deterministic multithreaded run (every thread.spawn vCPU is a DAP thread).
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(WORKERS_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("schedule", Json::Arr(vec![])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("worker.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(4))])]),
            ),
        ]),
    ));

    // configurationDone runs to the worker breakpoint — a *worker* thread stops, not the root (1).
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let first = event(&out, "stopped")
        .unwrap()
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert!(
        first >= 2,
        "a spawned worker stopped, not the root thread 1 (got {first})"
    );

    // `threads` lists every live vCPU (root + two workers).
    let out = s.handle(&req(5, "threads", Json::obj(vec![])));
    let threads = response(&out).get("body").unwrap().get("threads").unwrap();
    assert!(
        threads.as_array().unwrap().len() >= 2,
        "multiple threads are live: {}",
        threads.as_array().unwrap().len()
    );

    // stackTrace of the stopped worker → its frame at worker.c:4, with delta = 1.
    let out = s.handle(&req(
        6,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(first))]),
    ));
    let top = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .clone();
    assert_eq!(top.get("line"), Some(&Json::i(4)));
    assert_eq!(
        top.get("source").unwrap().get("name"),
        Some(&Json::s("worker.c"))
    );

    let fid = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        7,
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
        8,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let delta = vars
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("delta"))
        .and_then(|v| v.get("value"))
        .and_then(|v| v.as_str());
    assert_eq!(delta, Some("1"), "the worker's argument delta = 1");

    // continue → the *other* worker hits the same breakpoint (a different DAP thread).
    let out = s.handle(&req(9, "continue", Json::obj(vec![])));
    let second = event(&out, "stopped")
        .unwrap()
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_ne!(first, second, "the second hit is a different worker thread");
}

/// `evaluate` resolves a source variable by name in a frame (watch / hover), via `read_var`.
#[test]
fn dap_evaluate_resolves_a_variable_in_a_frame() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));

    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();

    let eval = |s: &mut DapServer, seq: i64, expr: &str| -> Vec<Json> {
        s.handle(&req(
            seq,
            "evaluate",
            Json::obj(vec![
                ("expression", Json::s(expr)),
                ("frameId", Json::i(fid)),
                ("context", Json::s("watch")),
            ]),
        ))
    };

    // Known locals resolve to their live values (first iteration: i = 3, acc = 0).
    let out = eval(&mut s, 6, "i");
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        response(&out).get("body").unwrap().get("result"),
        Some(&Json::s("3"))
    );
    let out = eval(&mut s, 7, "acc");
    assert_eq!(
        response(&out).get("body").unwrap().get("result"),
        Some(&Json::s("0"))
    );

    // An unknown name fails (the client shows "not available").
    let out = eval(&mut s, 8, "nope");
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}

/// Multithreaded reverse debugging: `reverseContinue` over the global scheduler `turn` walks back
/// through the per-thread breakpoint hits, then to the start.
#[test]
fn dap_multithreaded_reverse_continue_walks_back_through_hits() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(WORKERS_DBG)),
            ("function", Json::i(0)),
            ("schedule", Json::Arr(vec![])), // deterministic multithreaded
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("worker.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(4))])]),
            ),
        ]),
    ));

    // First worker hit, then continue to the second worker hit.
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(5, "continue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("breakpoint")),
        "second worker hits the breakpoint"
    );

    // reverseContinue → back to the earlier (first) hit; again → no earlier hit, rewind to start.
    let out = s.handle(&req(6, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("breakpoint")),
        "reverses to the earlier worker hit (over the global turn)"
    );
    let out = s.handle(&req(7, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("entry")),
        "no earlier hit → rewinds to the start"
    );
}

// Caller (func 0) calls helper (func 1) then continues; the call is m.c:5 and the next line m.c:6.
const CALL_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}

debug.file 0 "m.c"
debug.loc 0 0 0 0 5 1
debug.loc 0 0 1 0 6 1
"#;

/// DAP `next` steps over a call: from the call line it lands on the *next* line in the same frame,
/// without descending into the callee.
#[test]
fn dap_next_steps_over_a_call() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(CALL_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(5)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("m.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(5))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));

    // Stopped at the call line (m.c:5), one frame.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    assert_eq!(frames.as_array().unwrap().len(), 1);
    assert_eq!(frames.as_array().unwrap()[0].get("line"), Some(&Json::i(5)));

    // `next` runs the call to completion and lands on m.c:6 — still one frame (did not step in).
    let out = s.handle(&req(6, "next", Json::obj(vec![("threadId", Json::i(1))])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("step"))
    );
    let out = s.handle(&req(
        7,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    assert_eq!(
        frames.as_array().unwrap().len(),
        1,
        "stepped over the call, did not descend"
    );
    assert_eq!(frames.as_array().unwrap()[0].get("line"), Some(&Json::i(6)));
}

// Two source lines, the first spanning two ops (so op-stepping would stutter on it).
const TWO_LINES_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  v3 = i32.const 1
  v4 = i32.add v2 v3
  return v4
}

debug.file 0 "f.c"
debug.loc 0 0 0 0 1 1
debug.loc 0 0 2 0 2 1
"#;

/// DAP `next` advances a whole *source line*, not one IR op: from f.c:1 (two ops) a single `next`
/// lands on f.c:2.
#[test]
fn dap_next_steps_a_whole_source_line() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(TWO_LINES_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(5)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("f.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(1))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let line = |s: &mut DapServer| -> i64 {
        let out = s.handle(&req(
            99,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        response(&out)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("line")
            .unwrap()
            .as_i64()
            .unwrap()
    };
    assert_eq!(line(&mut s), 1, "stopped on f.c:1");
    s.handle(&req(5, "next", Json::obj(vec![("threadId", Json::i(1))])));
    assert_eq!(
        line(&mut s),
        2,
        "one `next` advanced the whole line to f.c:2"
    );
}

/// A conditional breakpoint stops only when its expression is nonzero: `i == 1` skips the i=3 and
/// i=2 iterations of the loop and stops at i=1.
#[test]
fn dap_conditional_breakpoint_stops_only_when_true() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![
                    ("line", Json::i(7)),
                    ("condition", Json::s("i == 1")),
                ])]),
            ),
        ]),
    ));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("breakpoint"))
    );
    // It skipped i=3 and i=2 and stopped where the condition held: i = 1, acc = 5.
    let l = locals(&mut s);
    assert_eq!((l["i"].as_str(), l["acc"].as_str()), ("1", "5"));
}

/// `reverseContinue` honors conditional breakpoints, like forward `continue` (DEBUGGING.md W5): it
/// skips a breakpoint hit whose condition is false when walking *backward*, so it lands on the
/// previous hit that actually fires. With `i != 2` over a loop hitting i=3,2,1, the hit immediately
/// before the i=1 stop (i=2) has a false condition — reverseContinue must skip it back to i=3.
#[test]
fn dap_reverse_continue_honors_conditional_breakpoints() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![
                    ("line", Json::i(7)),
                    ("condition", Json::s("i != 2")),
                ])]),
            ),
        ]),
    ));

    // Forward: i=3 fires; the next firing hit skips i=2 (condition false) → i=1.
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(locals(&mut s).get("i").map(String::as_str), Some("3"));
    s.handle(&req(5, "continue", Json::obj(vec![])));
    assert_eq!(
        locals(&mut s).get("i").map(String::as_str),
        Some("1"),
        "skipped i=2 forward"
    );

    // reverseContinue from i=1 → skips the false-condition i=2 hit, landing on i=3 (not i=2).
    let out = s.handle(&req(6, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        event(&out, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("breakpoint"))
    );
    assert_eq!(
        locals(&mut s).get("i").map(String::as_str),
        Some("3"),
        "reverse skipped the false-condition i=2 hit back to i=3"
    );
}

/// `evaluate` computes an integer expression over the frame's variables (watch / REPL / condition).
#[test]
fn dap_evaluate_computes_an_expression() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let eval = |s: &mut DapServer, e: &str| -> Vec<Json> {
        s.handle(&req(
            6,
            "evaluate",
            Json::obj(vec![("expression", Json::s(e)), ("frameId", Json::i(fid))]),
        ))
    };
    // First iteration: i = 3, acc = 0.
    let out = eval(&mut s, "i * 2 + acc");
    assert_eq!(
        response(&out).get("body").unwrap().get("result"),
        Some(&Json::s("6"))
    );
    let out = eval(&mut s, "i > acc");
    assert_eq!(
        response(&out).get("body").unwrap().get("result"),
        Some(&Json::s("1"))
    );
    // A divide-by-zero (acc = 0) fails cleanly.
    let out = eval(&mut s, "i / acc");
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}

// A function that fills a `struct Point { int x, y; }` at data-SP+0 and an `int[3]` at +8 (with
// known values), with a hand-written §6 structured-type debug section. Param v0 is the data-SP.
// Breakpoint line 6 maps to the last op (after all stores), so the window holds the values.
const AGGREGATES_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 11
  i32.store v0 v1
  v2 = i64.const 4
  v3 = i64.add v0 v2
  v4 = i32.const 22
  i32.store v3 v4
  v5 = i64.const 8
  v6 = i64.add v0 v5
  v7 = i32.const 100
  i32.store v6 v7
  v8 = i64.const 12
  v9 = i64.add v0 v8
  v10 = i32.const 200
  i32.store v9 v10
  v11 = i64.const 16
  v12 = i64.add v0 v11
  v13 = i32.const 300
  i32.store v12 v13
  v14 = i32.const 0
  return v14
}

debug.file 0 "s.c"
debug.loc 0 0 18 0 6 1
debug.type 0 base "int" signed 4
debug.type 1 agg "struct" 8
debug.field 1 "x" 0 0
debug.field 1 "y" 4 0
debug.type 2 array "array" 0 3
debug.var 0 "p" win 0 "struct" 1
debug.var 0 "row" win 8 "array" 2
"#;

/// `variables` as a map name → (value, variablesReference).
fn vars_map(out: &[Json]) -> std::collections::HashMap<String, (String, i64)> {
    response(out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap().to_string(),
                (
                    v.get("value").unwrap().as_str().unwrap().to_string(),
                    v.get("variablesReference").unwrap().as_i64().unwrap(),
                ),
            )
        })
        .collect()
}

#[test]
fn dap_expands_struct_and_array_in_the_variables_pane() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    // Drive func 0 with the data-SP as its argument (a window address with headroom).
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(AGGREGATES_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(1024)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/work/s.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(6))])]),
            ),
        ]),
    ));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(event(&out, "stopped").is_some(), "stops at the breakpoint");

    // stackTrace → frame → scopes → the Locals reference.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    let scope_ref = response(&out)
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

    // Locals: `p` and `row` are *expandable* (nonzero variablesReference), not leaf scalars.
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(scope_ref))]),
    ));
    let locals = vars_map(&out);
    let (p_summary, p_ref) = locals.get("p").expect("local p");
    let (row_summary, row_ref) = locals.get("row").expect("local row");
    assert_eq!(p_summary, "{...}", "struct summary");
    assert_eq!(row_summary, "[3]", "array summary (count)");
    assert!(*p_ref >= (1 << 20), "p is expandable");
    assert!(*row_ref >= (1 << 20), "row is expandable");

    // Expand the struct: fields x = 11, y = 22, both scalar leaves (reference 0).
    let out = s.handle(&req(
        8,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(*p_ref))]),
    ));
    let fields = vars_map(&out);
    assert_eq!(
        fields.get("x").map(|(v, r)| (v.as_str(), *r)),
        Some(("11", 0))
    );
    assert_eq!(
        fields.get("y").map(|(v, r)| (v.as_str(), *r)),
        Some(("22", 0))
    );

    // Expand the array: elements [0]=100, [1]=200, [2]=300.
    let out = s.handle(&req(
        9,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(*row_ref))]),
    ));
    let elems = vars_map(&out);
    assert_eq!(elems.get("[0]").map(|(v, _)| v.as_str()), Some("100"));
    assert_eq!(elems.get("[1]").map(|(v, _)| v.as_str()), Some("200"));
    assert_eq!(elems.get("[2]").map(|(v, _)| v.as_str()), Some("300"));
}

// A *non-C* debug section — Rust-flavored type names (`i32`, `Pair`) and a `.rs` file — over the
// same IR. The consumer (interpreter + DAP) must inspect it correctly using only the structured
// layout, never the type-name spelling: nothing here matches the `ty_width` C-name heuristic, so
// if the width came from the name (`i32` → fallback 8) the scalar would read 8 bytes and report a
// huge number. `n` and `pair` overlay the same 8 window bytes: i32 1000 at +0, i32 2000 at +4.
const NON_C_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 1000
  i32.store v0 v1
  v2 = i64.const 4
  v3 = i64.add v0 v2
  v4 = i32.const 2000
  i32.store v3 v4
  v5 = i32.const 0
  return v5
}

debug.file 0 "x.rs"
debug.loc 0 0 6 0 3 1
debug.type 0 base "i32" signed 4
debug.type 1 agg "Pair" 8
debug.field 1 "a" 0 0
debug.field 1 "b" 4 0
debug.var 0 "n" win 0 "i32" 0
debug.var 0 "pair" win 0 "Pair" 1
"#;

#[test]
fn dap_inspects_a_non_c_frontend_by_structured_layout_only() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(NON_C_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(1024)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/work/x.rs"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(3))])]),
            ),
        ]),
    ));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(event(&out, "stopped").is_some(), "stops at the breakpoint");

    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    let scope_ref = response(&out)
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

    // The scalar `n` reads exactly 4 bytes (from `TypeDef.size`), not 8 (the `i32` name heuristic).
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(scope_ref))]),
    ));
    let locals = vars_map(&out);
    assert_eq!(
        locals.get("n").map(|(v, _)| v.as_str()),
        Some("1000"),
        "scalar width came from the structured type, not the C-name fallback"
    );
    let (_, pair_ref) = locals.get("pair").expect("local pair");

    // The aggregate expands purely from its field offsets — type names are opaque to the consumer.
    let out = s.handle(&req(
        8,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(*pair_ref))]),
    ));
    let fields = vars_map(&out);
    assert_eq!(fields.get("a").map(|(v, _)| v.as_str()), Some("1000"));
    assert_eq!(fields.get("b").map(|(v, _)| v.as_str()), Some("2000"));

    // `evaluate` of the bare scalar likewise uses the structured width.
    let out = s.handle(&req(
        9,
        "evaluate",
        Json::obj(vec![
            ("expression", Json::s("n")),
            ("frameId", Json::i(fid)),
        ]),
    ));
    assert_eq!(
        response(&out).get("body").unwrap().get("result"),
        Some(&Json::s("1000"))
    );
}

/// `(success, result string)` of an `evaluate` response.
fn eval_result(out: &[Json]) -> (bool, Option<String>) {
    let r = response(out);
    let ok = r.get("success") == Some(&Json::Bool(true));
    let val = r
        .get("body")
        .and_then(|b| b.get("result"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    (ok, val)
}

/// Launch `program` (data-SP arg 1024), break at source `line`, and return the stopped frame id.
fn launch_and_break(s: &mut DapServer, program: &str, path: &str, line: i64) -> i64 {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(program)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(1024)])),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s(path))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(line))])]),
            ),
        ]),
    ));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(event(&out, "stopped").is_some(), "stops at the breakpoint");
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap()
}

#[test]
fn dap_evaluate_member_and_index_access() {
    // AGGREGATES_DBG: struct p{x=11,y=22} at +0, int row[3]={100,200,300} at +8.
    let mut s = DapServer::new();
    let fid = launch_and_break(&mut s, AGGREGATES_DBG, "/work/s.c", 6);
    let mut seq = 10;
    let mut eval = |s: &mut DapServer, e: &str| -> (bool, Option<String>) {
        seq += 1;
        let out = s.handle(&req(
            seq,
            "evaluate",
            Json::obj(vec![("expression", Json::s(e)), ("frameId", Json::i(fid))]),
        ));
        eval_result(&out)
    };

    // Member access over a struct, index over an array, and mixed arithmetic.
    assert_eq!(eval(&mut s, "p.x"), (true, Some("11".into())));
    assert_eq!(eval(&mut s, "p.y"), (true, Some("22".into())));
    assert_eq!(eval(&mut s, "row[0]"), (true, Some("100".into())));
    assert_eq!(eval(&mut s, "row[2]"), (true, Some("300".into())));
    assert_eq!(eval(&mut s, "p.x + row[1]"), (true, Some("211".into())));
    // The index is itself an expression (resolved to 0 here → row[0]).
    assert_eq!(eval(&mut s, "row[p.y - 22]"), (true, Some("100".into())));
    // Errors fail cleanly: unknown member, and member access on a scalar.
    assert!(!eval(&mut s, "p.nope").0, "unknown member fails");
    assert!(!eval(&mut s, "p.x.y").0, "member access on a scalar fails");
}

// A pointer `pp` (at +0) to a `struct Point {x=7,y=9}` placed at +16. Tests `->` and pointer
// indexing (`pp[0].y`). data-SP arg is 1024, so the stored pointer value is 1040.
const POINTER_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 16
  v2 = i64.add v0 v1
  v3 = i32.const 7
  i32.store v2 v3
  v4 = i64.const 20
  v5 = i64.add v0 v4
  v6 = i32.const 9
  i32.store v5 v6
  i64.store v0 v2
  v7 = i32.const 0
  return v7
}

debug.file 0 "p.c"
debug.loc 0 0 9 0 3 1
debug.type 0 base "int" signed 4
debug.type 1 agg "Point" 8
debug.field 1 "x" 0 0
debug.field 1 "y" 4 0
debug.type 2 ptr "ptr" 1 8
debug.var 0 "pp" win 0 "ptr" 2
"#;

#[test]
fn dap_evaluate_pointer_arrow_and_index() {
    let mut s = DapServer::new();
    let fid = launch_and_break(&mut s, POINTER_DBG, "/work/p.c", 3);
    let mut seq = 10;
    let mut eval = |s: &mut DapServer, e: &str| -> (bool, Option<String>) {
        seq += 1;
        let out = s.handle(&req(
            seq,
            "evaluate",
            Json::obj(vec![("expression", Json::s(e)), ("frameId", Json::i(fid))]),
        ));
        eval_result(&out)
    };

    // `pp->x` = (*pp).x = 7; `pp->y` = 9; `pp[0].y` = *(pp+0) then .y = 9.
    assert_eq!(eval(&mut s, "pp->x"), (true, Some("7".into())));
    assert_eq!(eval(&mut s, "pp->y"), (true, Some("9".into())));
    assert_eq!(eval(&mut s, "pp[0].y"), (true, Some("9".into())));
    assert_eq!(eval(&mut s, "pp->x + pp->y"), (true, Some("16".into())));
    // Arrow through a non-pointer fails cleanly.
    assert!(
        !eval(&mut s, "pp->x->y").0,
        "arrow through a non-pointer fails"
    );
}

/// The Locals `variablesReference` for the given frame (`scopes` → first scope).
fn scope_ref(s: &mut DapServer, fid: i64) -> i64 {
    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap()
}

/// `variables` on a reference, as a name → (value, variablesReference) map.
fn variables(s: &mut DapServer, vref: i64) -> std::collections::HashMap<String, (String, i64)> {
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    vars_map(&out)
}

#[test]
fn dap_expands_a_pointer_to_its_pointee() {
    // POINTER_DBG: `pp` (at +0) points at a `struct Point {x=7,y=9}` at +16; data-SP arg 1024,
    // so the stored pointer value is 1040 (= 0x410).
    let mut s = DapServer::new();
    let fid = launch_and_break(&mut s, POINTER_DBG, "/work/p.c", 3);
    let sref = scope_ref(&mut s, fid);

    // The pointer local shows its address and is expandable (not a bare scalar).
    let locals = variables(&mut s, sref);
    let (summary, pp_ref) = locals.get("pp").expect("local pp");
    assert_eq!(summary, "0x410", "pointer shows its hex value");
    assert!(*pp_ref >= (1 << 20), "pointer is expandable");

    // Expanding the pointer yields a single `*` child — the pointee struct, itself expandable.
    let deref = variables(&mut s, *pp_ref);
    let (star_summary, star_ref) = deref.get("*").expect("deref child");
    assert_eq!(star_summary, "{...}", "pointee is the struct");
    assert!(*star_ref >= (1 << 20), "pointee struct is expandable");

    // And through it, the struct's fields read correctly.
    let fields = variables(&mut s, *star_ref);
    assert_eq!(fields.get("x").map(|(v, _)| v.as_str()), Some("7"));
    assert_eq!(fields.get("y").map(|(v, _)| v.as_str()), Some("9"));
}

// A `double d = 2.5` stored in the window (typed via the structured table). Param v0 is data-SP;
// the break (line 3) is after the store.
const FLOAT_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = f64.const 2.5
  f64.store v0 v1
  v2 = i32.const 0
  return v2
}

debug.file 0 "f.c"
debug.loc 0 0 2 0 3 1
debug.type 0 base "double" float 8
debug.var 0 "d" win 0 "double" 0
"#;

#[test]
fn dap_evaluate_floats_and_short_circuit() {
    let mut s = DapServer::new();
    let fid = launch_and_break(&mut s, FLOAT_DBG, "/work/f.c", 3);
    let mut seq = 10;
    let mut eval = |s: &mut DapServer, e: &str| -> (bool, Option<String>) {
        seq += 1;
        let out = s.handle(&req(
            seq,
            "evaluate",
            Json::obj(vec![("expression", Json::s(e)), ("frameId", Json::i(fid))]),
        ));
        eval_result(&out)
    };

    // A bare `double` reads as its value (not the raw 64-bit pattern), via the structured type.
    assert_eq!(eval(&mut s, "d"), (true, Some("2.5".into())));
    // Float arithmetic with int promotion; a fractional result stays a float.
    assert_eq!(eval(&mut s, "d + 0.25"), (true, Some("2.75".into())));
    assert_eq!(eval(&mut s, "d * 2"), (true, Some("5".into())));
    // Comparisons over floats yield 0/1.
    assert_eq!(eval(&mut s, "d > 2"), (true, Some("1".into())));
    assert_eq!(eval(&mut s, "d < 2"), (true, Some("0".into())));
    // A bitwise op on a float is rejected.
    assert!(!eval(&mut s, "d & 1").0, "bitwise on a float fails");
    // Short-circuit: the dead branch isn't evaluated, so `1/0` never traps.
    assert_eq!(eval(&mut s, "d < 2 && 1 / 0"), (true, Some("0".into())));
    assert_eq!(eval(&mut s, "d > 2 || 1 / 0"), (true, Some("1".into())));
}

// A `WindowVia` struct local (the wasm/DWARF case): `p` lives at a runtime base (frame value v2 =
// the arg + 16) + 0, with a `struct {int x, y}`. v0 (= the launch arg 1024) is the frame pointer.
const WINVIA_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i64.const 16
  v2 = i64.add v0 v1
  v3 = i32.const 10
  i32.store v2 v3
  v4 = i64.const 4
  v5 = i64.add v2 v4
  v6 = i32.const 20
  i32.store v5 v6
  v7 = i32.const 0
  return v7
}

debug.file 0 "s.c"
debug.loc 0 0 8 0 3 1
debug.type 0 base "int" signed 4
debug.type 1 agg "struct" 8
debug.field 1 "x" 0 0
debug.field 1 "y" 4 0
debug.var 0 "p" winvia 1 0 2 2 0 "struct" 1
"#;

#[test]
fn dap_expands_and_evaluates_a_windowvia_struct() {
    // A WindowVia aggregate is a first-class window location: it expands in the Variables pane and
    // `evaluate` resolves its members — the wasm-variable consumer path (base resolved per pc).
    let mut s = DapServer::new();
    let fid = launch_and_break(&mut s, WINVIA_DBG, "/work/s.c", 3);
    let sref = scope_ref(&mut s, fid);

    // `p` shows as an expandable struct (a nonzero variablesReference).
    let locals = variables(&mut s, sref);
    let (summary, p_ref) = locals.get("p").expect("local p");
    assert_eq!(summary, "{...}", "struct summary");
    assert!(*p_ref >= (1 << 20), "p is expandable");

    // Expand it: fields x = 10, y = 20, read through the runtime base + field offsets.
    let fields = variables(&mut s, *p_ref);
    assert_eq!(fields.get("x").map(|(v, _)| v.as_str()), Some("10"));
    assert_eq!(fields.get("y").map(|(v, _)| v.as_str()), Some("20"));

    // `evaluate` resolves member access over the WindowVia place.
    let out = s.handle(&req(
        50,
        "evaluate",
        Json::obj(vec![
            ("expression", Json::s("p.x + p.y")),
            ("frameId", Json::i(fid)),
        ]),
    ));
    assert_eq!(eval_result(&out), (true, Some("30".into())));
}
