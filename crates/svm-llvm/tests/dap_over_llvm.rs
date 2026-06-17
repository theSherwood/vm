//! End-to-end W5 over the **LLVM frontend**: a real LLVM-bitcode → SVM-IR translation (with its §6
//! debug info, slice 25) is serialized to text and driven through the Debug Adapter Protocol — a
//! source breakpoint binds by line, the Variables pane expands an LLVM-ingested `struct`, and
//! `evaluate` reads a member. This proves the LLVM producer feeds the *actual DAP consumer* through
//! the text round-trip, not just the interpreter Inspector (DEBUGGING.md slice 27 / §6).

use std::path::PathBuf;
use std::process::Command;

use svm_dap::{DapServer, Json};

/// Compile a C snippet at `-O0 -g` (every local is an `alloca` + `dbg.declare`, the §6 `Window`
/// variable case). `None` (skip) if clang is unavailable.
fn compile_o0g(name: &str, src: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_llvm_dap_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_llvm_dap_{}_{}.bc", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let ok = Command::new("clang")
        .args([
            "-O0",
            "-g",
            "-emit-llvm",
            "-c",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        Some(bc)
    } else {
        eprintln!("note: skipping {name} (clang unavailable)");
        None
    }
}

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

#[test]
fn dap_inspects_llvm_translated_struct_end_to_end() {
    let src = "\
struct Point { int x; int y; };
int dist(int n) {
  struct Point p;
  struct Point *pp = &p;
  p.x = n; p.y = n + 1;
  return p.x + p.y + pp->x;
}
";
    let Some(bc) = compile_o0g("struct", src) else {
        return; // toolchain unavailable — skip
    };
    let t = svm_llvm::translate_bc_path(&bc).expect("translate bitcode");
    svm_verify::verify_module(&t.module).expect("verify");
    let di = t.module.debug_info.as_ref().expect("debug info");
    let path = di.files[0].clone();

    // The §6 debug info survives the text round-trip the DAP server consumes (`programText`).
    let text = svm_text::print_module(&t.module);
    assert!(
        svm_text::parse_module(&text).is_ok(),
        "the serialized module re-parses"
    );

    let n = 5i64;
    let mut s = DapServer::new();
    assert_eq!(
        response(&s.handle(&req(1, "initialize", Json::obj(vec![])))).get("success"),
        Some(&Json::Bool(true))
    );

    // launch: pass the data-SP (§3d) then the C arg, matching the entry signature `(sp, n)`.
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&text)),
            ("function", Json::i(0)),
            (
                "args",
                Json::Arr(vec![Json::i(t.entry_sp as i64), Json::i(n)]),
            ),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // Break on the `return` line (6) — by then p.x/p.y are stored. Binds by the recorded source path.
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s(&path))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(6))])]),
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
        "breakpoint binds to a source line"
    );

    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(event(&out, "stopped").is_some(), "stops at the breakpoint");

    // stackTrace → the top frame's id.
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
    let frame_id = top.get("id").unwrap().as_i64().unwrap();

    // scopes → the Locals variablesReference.
    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
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

    // variables on Locals → `p` is present and **expandable** (a nonzero, Place-tagged ref).
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(scope_ref))]),
    ));
    let resp = response(&out);
    let locals = resp
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap();
    let p = locals
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("p"))
        .expect("the struct local `p` appears in the Variables pane");
    let p_ref = p.get("variablesReference").unwrap().as_i64().unwrap();
    assert!(p_ref >= (1 << 20), "p is expandable (a struct)");

    // Expand `p` → its members read x = n, y = n + 1.
    let out = s.handle(&req(
        8,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(p_ref))]),
    ));
    let members: std::collections::HashMap<String, String> = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                v.get("value")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
            )
        })
        .collect();
    assert_eq!(
        members.get("x").map(String::as_str),
        Some(n.to_string()).as_deref(),
        "p.x"
    );
    assert_eq!(
        members.get("y").map(String::as_str),
        Some((n + 1).to_string()).as_deref(),
        "p.y"
    );

    // evaluate a member expression over the LLVM-ingested struct.
    let out = s.handle(&req(
        9,
        "evaluate",
        Json::obj(vec![
            ("expression", Json::s("p.x + p.y")),
            ("frameId", Json::i(frame_id)),
        ]),
    ));
    let r = response(&out);
    assert_eq!(
        r.get("success"),
        Some(&Json::Bool(true)),
        "evaluate succeeds"
    );
    assert_eq!(
        r.get("body")
            .unwrap()
            .get("result")
            .and_then(|v| v.as_str()),
        Some((n + n + 1).to_string()).as_deref(),
        "p.x + p.y = 2n+1"
    );
}
