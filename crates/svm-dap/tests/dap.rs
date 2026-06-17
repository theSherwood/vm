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
