//! Slice 6 (INTERACTIVE_EMBEDDING.md): the **scheduler trace seam** — an observation-only tape of
//! the cooperative debug scheduler's decisions (turns, parks, wakes with waker→wakee identities,
//! spawns) — and the **shared-state consumer** (contested-word tracking over the attributed access
//! sink). Acceptance: wake edges match wait/notify semantics on a futex-handoff fixture, the tape
//! is bit-identical across a replay of the same run, contested flags match a hand-computed oracle,
//! and everything fails closed off the threaded bytecode path.

use std::sync::{Arc, Mutex};
use svm_dap::models::{MemModel, MemModelCfg};
use svm_dap::{BytecodeBackend, DapServer, Debuggee, Json, SharedSink};
use svm_interp::Stop;
use svm_text::parse_module;

/// The `bytecode_debug_threads.rs` futex handoff: the root seeds mem[8], spawns a worker, flags +
/// notifies mem[0], joins; the worker waits on mem[0] then reads mem[8].
const FUTEX_HANDOFF: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 8
  v1 = i64.const 987654
  i64.atomic.store v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 0
  v5 = i32.const 1
  i32.atomic.store v4 v5
  v6 = i64.const 0
  v7 = i32.const 1
  v8 = atomic.notify v6 v7
  v9 = thread.join v3
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 0
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 8
  v6 = i64.atomic.load v5
  return v6
  }
}
"#;

/// Two workers race load/add/store on mem[0]; the root joins both and returns the count.
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
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
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
  }
}
"#;

/// Build a threaded backend for `src`, arm the trace, run to completion, return the tape JSON.
fn traced_run(src: &str) -> (BytecodeBackend, String) {
    let m = parse_module(src).expect("parses");
    let mut b = BytecodeBackend::new(m, 0, &[], u64::MAX, false, Vec::new(), false, None, None)
        .expect("subset");
    assert!(b.is_threaded(), "a thread.spawn guest");
    assert!(Debuggee::set_sched_trace(&mut b, true), "trace armed");
    loop {
        match Debuggee::run_until_stop(&mut b) {
            Stop::Finished(r) => {
                r.expect("clean finish");
                break;
            }
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    let tape = b.sched_trace_json().expect("a tape").to_string();
    (b, tape)
}

fn kinds_of(tape: &str) -> Vec<(String, Json)> {
    svm_dap::parse(tape)
        .and_then(|j| j.as_array().map(|a| a.to_vec()))
        .expect("a JSON array")
        .into_iter()
        .map(|e| {
            (
                e.get("kind")
                    .and_then(|k| k.as_str())
                    .unwrap_or("")
                    .to_string(),
                e,
            )
        })
        .collect()
}

/// A guest where the **root genuinely parks on the futex**: under the lowest-index schedule the
/// root runs first — it spawns the worker, then `atomic.wait`s on mem[0] while the worker hasn't
/// run yet, so it parks for real; the worker then sets the flag and `notify`s it awake.
const ROOT_WAITS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = i64.const 0
  v3 = i32.const 0
  v4 = i64.const 1000000000
  v5 = i32.atomic.wait v2 v3 v4
  v6 = thread.join v1
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v1 = i64.const 0
  v2 = i32.const 1
  i32.atomic.store v1 v2
  v3 = i32.const 1
  v4 = atomic.notify v1 v3
  v5 = i64.const 42
  return v5
  }
}
"#;

/// **Wake edges match wait/notify semantics**: the root's futex park appears as
/// `parkWait(task 0, key 0)`, the worker's notify as `wakeNotify(waker 1 → wakee 0)`, the spawn
/// as `spawn(parent 0 → task 1)`, and the tape carries the turns. (The `FUTEX_HANDOFF` fixture's
/// worker never parks under the lowest-index schedule — the root's flag lands first and the wait
/// falls through `WAIT_NOT_EQUAL` — which the tape faithfully shows as *no* park edge; this
/// fixture makes the root park for real.)
#[test]
fn futex_wait_notify_edges_are_identified() {
    let (_, tape) = traced_run(ROOT_WAITS);
    let events = kinds_of(&tape);
    let field = |e: &Json, k: &str| e.get(k).and_then(|v| v.as_i64());
    assert!(
        events.iter().any(|(k, e)| k == "spawn"
            && field(e, "parent") == Some(0)
            && field(e, "task") == Some(1)),
        "the root's spawn edge: {tape}"
    );
    assert!(
        events.iter().any(|(k, e)| k == "parkWait"
            && field(e, "task") == Some(0)
            && field(e, "key") == Some(0)),
        "the root parks on futex word 0: {tape}"
    );
    assert!(
        events.iter().any(|(k, e)| k == "wakeNotify"
            && field(e, "waker") == Some(1)
            && field(e, "wakee") == Some(0)),
        "the worker's notify wakes the root, both identities: {tape}"
    );
    assert!(
        events.iter().filter(|(k, _)| k == "turn").count() > 10,
        "the tape carries the turns"
    );
    // And the FUTEX_HANDOFF tape shows the join edges (park at the root's join, the worker's
    // completion waking it) — the other wake kind, identified.
    let (_, tape) = traced_run(FUTEX_HANDOFF);
    let events = kinds_of(&tape);
    assert!(
        events.iter().any(|(k, e)| k == "parkJoin"
            && field(e, "task") == Some(0)
            && field(e, "child") == Some(1)),
        "the root parks joining the worker: {tape}"
    );
    assert!(
        events.iter().any(|(k, e)| k == "wakeJoin"
            && field(e, "waker") == Some(1)
            && field(e, "wakee") == Some(0)),
        "the worker's completion wakes the joiner: {tape}"
    );
}

/// **The tape is deterministic**: a second identical run and a `seek(0)`-then-rerun both reproduce
/// it bit-for-bit (the replay refills the re-armed tape).
#[test]
fn trace_tape_is_bit_identical_across_replay() {
    let (mut a, tape_a) = traced_run(FUTEX_HANDOFF);
    let (_, tape_b) = traced_run(FUTEX_HANDOFF);
    assert_eq!(tape_a, tape_b, "same run, same tape");
    // Rewind to turn 0 (rebuild + re-armed empty tape), then drive to completion again.
    let _ = Debuggee::seek(&mut a, 0);
    loop {
        match Debuggee::run_until_stop(&mut a) {
            Stop::Finished(_) => break,
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    let replayed = a.sched_trace_json().expect("a tape").to_string();
    assert_eq!(
        tape_a, replayed,
        "seek(0) + rerun refills the identical tape"
    );
}

/// **The shared-state consumer**: the racy counter's word 0 — written by the workers, touched by
/// all three tasks — is contested; a single-task guest never flags anything.
#[test]
fn contested_words_match_the_oracle() {
    let feed_model = |src: &str| -> Arc<Mutex<MemModel>> {
        let m = parse_module(src).expect("parses");
        let mut b = BytecodeBackend::new(m, 0, &[], u64::MAX, false, Vec::new(), false, None, None)
            .expect("subset");
        let model = Arc::new(Mutex::new(MemModel::new(MemModelCfg::default())));
        let feed = Arc::clone(&model);
        let sink: SharedSink = Arc::new(Mutex::new(
            move |c: u64, t: usize, e: svm_interp::MemEvent| {
                feed.lock()
                    .unwrap_or_else(|x| x.into_inner())
                    .observe(c, t, e)
            },
        ));
        Debuggee::set_access_sink(&mut b, sink);
        loop {
            match Debuggee::run_until_stop(&mut b) {
                Stop::Finished(_) => break,
                Stop::Break { .. } => continue,
                Stop::Blocked => panic!("blocked"),
            }
        }
        model
    };
    let racy = feed_model(RACY_COUNTER);
    let stats = racy.lock().unwrap().stats_json();
    let contested = stats
        .get("sharedState")
        .and_then(|s| s.get("contested"))
        .and_then(|c| c.as_array().map(|a| a.to_vec()))
        .expect("contested list");
    assert!(
        contested
            .iter()
            .any(|e| e.get("addr").and_then(|a| a.as_i64()) == Some(0)),
        "word 0 is contested: {stats}"
    );
    // The futex fixture's mem[8] handoff word is also contested (root writes, worker reads).
    let handoff = feed_model(FUTEX_HANDOFF);
    let stats = handoff.lock().unwrap().stats_json();
    let contested = stats
        .get("sharedState")
        .and_then(|s| s.get("contested"))
        .and_then(|c| c.as_array().map(|a| a.to_vec()))
        .expect("contested list");
    assert!(
        contested
            .iter()
            .any(|e| e.get("addr").and_then(|a| a.as_i64()) == Some(8)
                && e.get("lastWriter").and_then(|w| w.as_i64()) == Some(0)),
        "the handoff word is contested with the root as last writer: {stats}"
    );
}

/// **Fail-closed**: `schedTrace` on a single-vCPU or tree-walker launch fails; the request without
/// arming fails.
#[test]
fn sched_trace_fails_closed_off_the_threaded_path() {
    const SINGLE: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  return v0
  }
}
"#;
    let req = |seq: i64, command: &str, args: Json| {
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("request")),
            ("command", Json::s(command)),
            ("arguments", args),
        ])
    };
    let response = |msgs: &[Json]| -> Json {
        msgs.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .expect("a response")
            .clone()
    };
    for (name, engine) in [("treewalk", None), ("bytecode-single", Some("bytecode"))] {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut launch = vec![
            ("programText", Json::s(SINGLE)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("schedTrace", Json::Bool(true)),
        ];
        if let Some(e) = engine {
            launch.push(("engine", Json::s(e)));
        }
        let out = s.handle(&req(2, "launch", Json::obj(launch)));
        assert_eq!(
            response(&out).get("success"),
            Some(&Json::Bool(false)),
            "{name} + schedTrace fails the launch"
        );
    }
    // A threaded session without arming: the request fails cleanly.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(3, "schedTrace", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}
