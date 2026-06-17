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
