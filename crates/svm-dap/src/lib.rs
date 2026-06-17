//! An **interpreter-backed Debug Adapter Protocol server** (DEBUGGING.md W5, slice 1).
//!
//! DAP is the protocol VS Code (and other editors) speak to a debugger. This crate translates DAP
//! requests onto the `svm-interp` [`Inspector`] — which already does the real work (breakpoints,
//! stepping, backtrace, source-location and named-variable resolution from the §6/W4 debug info).
//! So the **interpreter is the stepping engine** and source mapping comes straight from the IR
//! debug info: no DWARF, no JIT, sidestepping optimized-code inspection entirely (the doc's
//! recommended first tier). The result is the W5 acceptance: set a breakpoint on a source line, it
//! binds, hitting it shows the source frame and inspectable locals.
//!
//! [`DapServer::handle`] is pure request→messages logic (response + any events), unit-testable by
//! scripting a DAP conversation; [`run_stdio`] is the thin `Content-Length`-framed wire loop a real
//! client connects to.

use std::collections::BTreeMap;

use svm_interp::{Inspector, IrPc, Stop, StopReason, Value, VarValue};
use svm_ir::{DebugInfo, ValType};

mod expr;
mod json;
pub use json::{parse, Json};

/// The DAP server: protocol state + (after `launch`) a debug [`Session`]. Drive it by feeding parsed
/// request objects to [`handle`](DapServer::handle), or run the wire loop with [`run_stdio`].
#[derive(Default)]
pub struct DapServer {
    /// Monotonic sequence number for messages this server emits.
    seq: i64,
    session: Option<Session>,
    /// `launch` arg: pause at the entry op instead of running to the first breakpoint.
    stop_on_entry: bool,
    /// Set once the guest has finished (or the client disconnected).
    terminated: bool,
}

/// The live debug target, created by `launch`.
struct Session {
    inspector: Inspector,
    debug: Option<DebugInfo>,
    /// `(file, line) → first IR pc on that line` — the reverse of `Inspector::source_loc`, for
    /// binding source-line breakpoints.
    line_index: BTreeMap<(u32, u32), IrPc>,
    /// IR pcs currently set as breakpoints (so `setBreakpoints` can replace them per the protocol).
    breakpoints: Vec<IrPc>,
    /// Conditional breakpoints (DAP `condition`): an IR pc → an integer expression that must be
    /// nonzero in the stopped frame for the breakpoint to actually stop (DEBUGGING.md W5). Absent ⇒
    /// unconditional. Evaluated by [`expr::eval_int`] over the frame's source variables.
    conditions: BTreeMap<IrPc, String>,
    /// DAP `frameId` → `(thread id, frame index)`, assigned during `stackTrace` and consumed by
    /// `scopes`/`variables`/`evaluate`, so a reference names a specific thread's frame (multithreaded
    /// DAP). Cleared whenever execution resumes (old frames become stale).
    frame_refs: Vec<(u64, usize)>,
    /// Multithreaded run (`attach_scheduled[_seeded]`)? Selects the time-travel coordinate: the
    /// global scheduler `turn` when scheduled, the op `clock` single-threaded.
    scheduled: bool,
}

impl Session {
    /// Resolve a source variable `name` (declared in function `func`) to an `i64`, in the frame
    /// `frame_idx` levels up the focused thread — the binding the expression evaluator and
    /// conditional breakpoints use. `None` if there is no such variable or its value isn't integral.
    fn resolve_var(&self, frame_idx: usize, func: u32, name: &str) -> Option<i64> {
        let ty = self
            .debug
            .as_ref()?
            .vars
            .iter()
            .find(|v| v.func == func && v.name == name)?
            .ty
            .clone();
        var_to_i64(&self.inspector.read_var(frame_idx, name, ty_width(&ty))?)
    }
}

/// An event to emit after a response: `(event-name, body)`.
type Event = (&'static str, Json);

/// Which DAP step request is being served: `stepIn` (into calls), `next` (over calls), `stepOut`.
enum StepKind {
    In,
    Over,
    Out,
}

impl DapServer {
    pub fn new() -> DapServer {
        DapServer::default()
    }

    /// Whether the session has ended (guest finished or client disconnected).
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Process one DAP request, returning the response followed by any events to emit (each a
    /// complete DAP message object).
    pub fn handle(&mut self, request: &Json) -> Vec<Json> {
        let command = request
            .get("command")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        let req_seq = request.get("seq").and_then(|s| s.as_i64()).unwrap_or(0);
        let args = request.get("arguments");

        let (success, body, events) = match command.as_str() {
            "initialize" => self.on_initialize(),
            "launch" => self.on_launch(args),
            "setBreakpoints" => self.on_set_breakpoints(args),
            "configurationDone" => self.on_configuration_done(),
            "threads" => self.on_threads(),
            "stackTrace" => self.on_stack_trace(args),
            "scopes" => self.on_scopes(args),
            "variables" => self.on_variables(args),
            "continue" => self.on_continue(),
            "next" => self.on_step(StepKind::Over),
            "stepIn" => self.on_step(StepKind::In),
            "stepOut" => self.on_step(StepKind::Out),
            "stepBack" => self.on_step_back(),
            "reverseContinue" => self.on_reverse_continue(),
            "evaluate" => self.on_evaluate(args),
            "disconnect" => self.on_disconnect(),
            // An unrecognized request fails cleanly rather than crashing the session.
            _ => (false, Json::Null, vec![]),
        };

        let mut out = vec![self.response(req_seq, &command, success, body)];
        for (name, ebody) in events {
            out.push(self.event(name, ebody));
        }
        out
    }

    // --- message builders ---------------------------------------------------------------------

    fn next_seq(&mut self) -> i64 {
        self.seq += 1;
        self.seq
    }

    fn response(&mut self, req_seq: i64, command: &str, success: bool, body: Json) -> Json {
        let seq = self.next_seq();
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("response")),
            ("request_seq", Json::i(req_seq)),
            ("success", Json::Bool(success)),
            ("command", Json::s(command)),
            ("body", body),
        ])
    }

    fn event(&mut self, name: &str, body: Json) -> Json {
        let seq = self.next_seq();
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("event")),
            ("event", Json::s(name)),
            ("body", body),
        ])
    }

    // --- request handlers ---------------------------------------------------------------------

    fn on_initialize(&mut self) -> (bool, Json, Vec<Event>) {
        let caps = Json::obj(vec![
            ("supportsConfigurationDoneRequest", Json::Bool(true)),
            ("supportsSingleThreadExecutionRequests", Json::Bool(true)),
            // Reverse debugging (DEBUGGING.md W1 time-travel): tells the client to enable its
            // step-back / reverse-continue controls, backed by `Inspector::step_back`/`seek`.
            ("supportsStepBack", Json::Bool(true)),
            // Resolve a variable on hover, not just in the Variables pane.
            ("supportsEvaluateForHovers", Json::Bool(true)),
            // Conditional breakpoints (DEBUGGING.md W5): stop only when the `condition` is nonzero.
            ("supportsConditionalBreakpoints", Json::Bool(true)),
        ]);
        // The client now sends breakpoints, then `configurationDone`.
        (true, caps, vec![("initialized", Json::obj(vec![]))])
    }

    fn on_launch(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        let Some(args) = args else {
            return (false, Json::Null, vec![]);
        };
        self.stop_on_entry = args
            .get("stopOnEntry")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // `programText` (inline IR) wins for tests; otherwise read `program` (a path).
        let text = match args.get("programText").and_then(|v| v.as_str()) {
            Some(t) => t.to_string(),
            None => match args.get("program").and_then(|v| v.as_str()) {
                Some(path) => match std::fs::read_to_string(path) {
                    Ok(t) => t,
                    Err(_) => return (false, Json::Null, vec![]),
                },
                None => return (false, Json::Null, vec![]),
            },
        };
        let Ok(module) = svm_text::parse_module(&text) else {
            return (false, Json::Null, vec![]);
        };
        let func = args.get("function").and_then(|v| v.as_i64()).unwrap_or(0) as u32;
        let Some(fdef) = module.funcs.get(func as usize) else {
            return (false, Json::Null, vec![]);
        };
        let call_args = coerce_args(args.get("args"), &fdef.params);
        let fuel = args
            .get("fuel")
            .and_then(|v| v.as_i64())
            .unwrap_or(1_000_000_000) as u64;

        let debug = module.debug_info.clone();
        let line_index = debug.as_ref().map(build_line_index).unwrap_or_default();
        // Execution mode (DEBUGGING.md Milestone B): `seed` ⇒ a fuzzed interleaving; `schedule`
        // (possibly empty) ⇒ a fixed multithreaded interleaving (a witness, or the deterministic
        // default); neither ⇒ single-threaded. Multithreaded debugging surfaces every `thread.spawn`
        // vCPU as a DAP thread.
        let scheduled = args.get("seed").is_some() || args.get("schedule").is_some();
        let inspector = if let Some(seed) = args.get("seed").and_then(|v| v.as_i64()) {
            Inspector::attach_scheduled_seeded(&module, func, &call_args, fuel, seed as u64)
        } else if let Some(plan) = args.get("schedule").and_then(|v| v.as_array()) {
            let plan: Vec<u64> = plan
                .iter()
                .filter_map(|t| t.as_i64())
                .map(|t| t as u64)
                .collect();
            Inspector::attach_scheduled(&module, func, &call_args, fuel, plan)
        } else {
            Inspector::attach(&module, func, &call_args, fuel)
        };
        self.session = Some(Session {
            inspector,
            debug,
            line_index,
            breakpoints: Vec::new(),
            conditions: BTreeMap::new(),
            frame_refs: Vec::new(),
            scheduled,
        });
        (true, Json::Null, vec![])
    }

    fn on_set_breakpoints(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        let Some(session) = self.session.as_mut() else {
            return (false, Json::Null, vec![]);
        };
        // Resolve the requested source to a debug-info file index (exact / suffix / basename).
        let path = args
            .and_then(|a| a.get("source"))
            .and_then(|s| s.get("path").or_else(|| s.get("name")))
            .and_then(|p| p.as_str())
            .unwrap_or("");
        let files = session.debug.as_ref().map(|d| &d.files);
        let file_idx = files.and_then(|fs| match_file(fs, path));

        // Per DAP, setBreakpoints *replaces* this source's breakpoints — clear the old set first.
        // (Conditions are keyed by pc; clearing the whole map is fine since slice 1 is single-source.)
        for pc in session.breakpoints.drain(..) {
            session.inspector.clear_breakpoint(pc);
        }
        session.conditions.clear();

        let requested = args
            .and_then(|a| a.get("breakpoints"))
            .and_then(|b| b.as_array())
            .map(|a| a.to_vec())
            .unwrap_or_default();
        let mut out = Vec::new();
        for bp in &requested {
            let line = bp.get("line").and_then(|l| l.as_i64()).unwrap_or(0) as u32;
            match file_idx.and_then(|fi| resolve_line(&session.line_index, fi, line)) {
                Some((actual_line, pc)) => {
                    session.inspector.set_breakpoint(pc);
                    session.breakpoints.push(pc);
                    // A `condition` (an integer expression) only stops when it evaluates nonzero.
                    if let Some(cond) = bp.get("condition").and_then(|c| c.as_str()) {
                        if !cond.trim().is_empty() {
                            session.conditions.insert(pc, cond.to_string());
                        }
                    }
                    out.push(Json::obj(vec![
                        ("verified", Json::Bool(true)),
                        ("line", Json::i(actual_line as i64)),
                    ]));
                }
                None => out.push(Json::obj(vec![("verified", Json::Bool(false))])),
            }
        }
        (
            true,
            Json::obj(vec![("breakpoints", Json::Arr(out))]),
            vec![],
        )
    }

    fn on_configuration_done(&mut self) -> (bool, Json, Vec<Event>) {
        // The interpreter starts paused before the first op. If `stopOnEntry`, surface that as a
        // stop; otherwise run to the first breakpoint (or completion).
        if self.stop_on_entry {
            let ev = stopped_event("entry", 1);
            (true, Json::Null, vec![ev])
        } else if self.session.is_none() {
            (false, Json::Null, vec![])
        } else {
            let stop = self.run_with_conditions();
            (true, Json::Null, self.stop_events(stop))
        }
    }

    fn on_threads(&mut self) -> (bool, Json, Vec<Event>) {
        // Every live `thread.spawn` vCPU is a DAP thread (Milestone B). DAP thread id = vCPU id + 1
        // (1-based; the root vCPU 0 ⇒ thread 1). Single-threaded runs report the sole vCPU.
        let ids = self
            .session
            .as_ref()
            .map(|s| s.inspector.threads())
            .unwrap_or_default();
        let threads = ids
            .iter()
            .map(|&tid| {
                Json::obj(vec![
                    ("id", Json::i(tid as i64 + 1)),
                    ("name", Json::s(format!("thread-{tid}"))),
                ])
            })
            .collect();
        (
            true,
            Json::obj(vec![("threads", Json::Arr(threads))]),
            vec![],
        )
    }

    fn on_stack_trace(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        // Focus the requested thread, then report *its* stack (multithreaded DAP). In a
        // single-threaded run `select_task` is a no-op and the sole vCPU is used.
        let thread_id = args
            .and_then(|a| a.get("threadId"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1);
        let tid = thread_id.max(1) as u64 - 1;
        let Some(session) = self.session.as_mut() else {
            return (false, Json::Null, vec![]);
        };
        session.inspector.select_task(tid);
        let frames = session.inspector.backtrace();
        let mut out = Vec::new();
        for f in frames.iter() {
            // Assign a `frameId` that records (thread, frame index) so `variables` reads the right
            // thread's frame even after the client switches threads.
            let frame_id = session.frame_refs.len();
            session.frame_refs.push((tid, out.len()));
            let mut fields = vec![
                ("id", Json::i(frame_id as i64)),
                ("name", Json::s(format!("#{} func{}", out.len(), f.pc.func))),
                (
                    "line",
                    Json::i(f.source.as_ref().map(|s| s.line as i64).unwrap_or(0)),
                ),
                (
                    "column",
                    Json::i(f.source.as_ref().map(|s| s.col as i64).unwrap_or(0)),
                ),
            ];
            if let Some(src) = &f.source {
                fields.push((
                    "source",
                    Json::obj(vec![
                        ("name", Json::s(basename(&src.file))),
                        ("path", Json::s(src.file.clone())),
                    ]),
                ));
            }
            out.push(Json::Obj(
                fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ));
        }
        let total = out.len() as i64;
        (
            true,
            Json::obj(vec![
                ("stackFrames", Json::Arr(out)),
                ("totalFrames", Json::i(total)),
            ]),
            vec![],
        )
    }

    fn on_scopes(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        let frame_id = args
            .and_then(|a| a.get("frameId"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        // A single "Locals" scope per frame; variablesReference = frameId+1 (nonzero ⇒ has children,
        // and `variables` recovers the frame from `frame_refs[ref-1]`).
        let scope = Json::obj(vec![
            ("name", Json::s("Locals")),
            ("variablesReference", Json::i(frame_id + 1)),
            ("expensive", Json::Bool(false)),
        ]);
        (
            true,
            Json::obj(vec![("scopes", Json::Arr(vec![scope]))]),
            vec![],
        )
    }

    fn on_variables(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        let vref = args
            .and_then(|a| a.get("variablesReference"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let empty = (
            true,
            Json::obj(vec![("variables", Json::Arr(vec![]))]),
            vec![],
        );
        let Some(session) = self.session.as_mut() else {
            return (false, Json::Null, vec![]);
        };
        // Recover (thread, frame index) from the reference, focus that thread, then read its locals.
        let Some(&(tid, frame_idx)) = vref
            .checked_sub(1)
            .and_then(|r| session.frame_refs.get(r as usize))
        else {
            return empty;
        };
        let Some(debug) = session.debug.as_ref() else {
            return empty;
        };
        session.inspector.select_task(tid);
        let frames = session.inspector.backtrace();
        let mut out = Vec::new();
        if let Some(frame) = frames.get(frame_idx) {
            // Collect the (name, type) pairs first so the `debug` borrow ends before `read_var`.
            let vars: Vec<(String, String)> = debug
                .vars
                .iter()
                .filter(|v| v.func == frame.pc.func)
                .map(|v| (v.name.clone(), v.ty.clone()))
                .collect();
            for (name, ty) in vars {
                if let Some(val) = session.inspector.read_var(frame_idx, &name, ty_width(&ty)) {
                    out.push(Json::obj(vec![
                        ("name", Json::s(name)),
                        ("value", Json::s(fmt_var(&val))),
                        ("type", Json::s(ty)),
                        ("variablesReference", Json::i(0)),
                    ]));
                }
            }
        }
        (true, Json::obj(vec![("variables", Json::Arr(out))]), vec![])
    }

    fn on_continue(&mut self) -> (bool, Json, Vec<Event>) {
        if self.session.is_none() {
            return (false, Json::Null, vec![]);
        }
        let stop = self.run_with_conditions();
        (
            true,
            Json::obj(vec![("allThreadsContinued", Json::Bool(true))]),
            self.stop_events(stop),
        )
    }

    /// Resume until a stop, transparently skipping conditional breakpoints whose condition is false
    /// (DEBUGGING.md W5) — so `continue` lands only on breakpoints that actually fire.
    fn run_with_conditions(&mut self) -> Stop {
        loop {
            let stop = match self.session.as_mut() {
                Some(s) => s.inspector.run_until_stop(),
                None => return Stop::Blocked,
            };
            match stop {
                Stop::Break {
                    reason: StopReason::Breakpoint,
                    pc,
                } if !self.condition_holds(pc) => continue, // condition false: keep going
                other => return other,
            }
        }
    }

    /// Whether a stop at `pc` should surface: `true` for an unconditional breakpoint or a condition
    /// that evaluates nonzero in the stopped (innermost) frame. A malformed/unresolvable condition
    /// also stops (so the user notices rather than silently skipping).
    fn condition_holds(&self, pc: IrPc) -> bool {
        let Some(session) = self.session.as_ref() else {
            return true;
        };
        let Some(cond) = session.conditions.get(&pc) else {
            return true; // unconditional
        };
        let func = match session.inspector.backtrace().first() {
            Some(f) => f.pc.func,
            None => return true,
        };
        let resolve = |name: &str| session.resolve_var(0, func, name);
        match expr::eval_int(cond, &resolve) {
            Some(v) => v != 0,
            None => true,
        }
    }

    fn on_step(&mut self, kind: StepKind) -> (bool, Json, Vec<Event>) {
        // `stepIn` descends into calls (single op); `next` runs over them; `stepOut` runs to return.
        // With debug info, `next`/`stepIn` step a whole *source line* (op-stepping until the frame's
        // line changes) so the editor advances a line at a time, not an op at a time; `stepOut`
        // already lands in the caller. Without debug info, all three stay op-level (IR debugging).
        let has_debug = match self.session.as_ref() {
            Some(s) => s.debug.is_some(),
            None => return (false, Json::Null, vec![]),
        };
        let stop = match kind {
            StepKind::Out => self.session.as_mut().unwrap().inspector.step_out(),
            StepKind::In | StepKind::Over if has_debug => self.step_source_line(kind),
            StepKind::In => self.session.as_mut().unwrap().inspector.step(),
            StepKind::Over => self.session.as_mut().unwrap().inspector.step_over(),
        };
        (true, Json::Null, self.stop_events(stop))
    }

    /// Op-step (in / over) until the focused frame's *source line* changes — so DAP `next`/`stepIn`
    /// advance one source line, not one IR op (DEBUGGING.md W5). Stops early on a breakpoint, a
    /// finish, or a generous op cap (defensive against unmapped code). Skips ops with no source line
    /// and ops still on the starting line.
    fn step_source_line(&mut self, kind: StepKind) -> Stop {
        const CAP: usize = 1_000_000;
        let start = self.current_source_line();
        let mut last = Stop::Blocked;
        for _ in 0..CAP {
            last = match self.session.as_mut() {
                Some(s) => match kind {
                    StepKind::In => s.inspector.step(),
                    _ => s.inspector.step_over(),
                },
                None => return Stop::Blocked,
            };
            match last {
                Stop::Break {
                    reason: StopReason::Step,
                    ..
                } => {
                    let now = self.current_source_line();
                    if now.is_some() && now != start {
                        return last; // reached a different source line
                    }
                    // else: same line or an unmapped op — keep stepping
                }
                _ => return last, // a real breakpoint, finish, or block preempts the line-step
            }
        }
        last
    }

    /// The `(file, line)` of the focused thread's innermost frame, if it maps to source.
    fn current_source_line(&self) -> Option<(String, u32)> {
        let s = self.session.as_ref()?;
        s.inspector
            .backtrace()
            .first()
            .and_then(|f| f.source.as_ref().map(|src| (src.file.clone(), src.line)))
    }

    fn on_step_back(&mut self) -> (bool, Json, Vec<Event>) {
        // Reverse single-step (DEBUGGING.md W1): `Inspector::step_back` re-executes to one unit of
        // logical time earlier. At the start it stays put. Lands as a `step` stop.
        let stop = match self.session.as_mut() {
            Some(s) => s.inspector.step_back(),
            None => return (false, Json::Null, vec![]),
        };
        (true, Json::Null, self.stop_events(stop))
    }

    fn on_reverse_continue(&mut self) -> (bool, Json, Vec<Event>) {
        // Run *backward* to the previous breakpoint, else to the start (DEBUGGING.md W1). Found by
        // re-executing from time 0 and remembering the last stop strictly before the current time,
        // then seeking there. (Single-threaded DAP uses the op `clock`; multithreaded would use the
        // global `turn`.)
        let Some(session) = self.session.as_mut() else {
            return (false, Json::Null, vec![]);
        };
        session.frame_refs.clear();
        // The time-travel coordinate: the global scheduler `turn` when multithreaded, the op `clock`
        // single-threaded.
        let scheduled = session.scheduled;
        let pos = |i: &Inspector| if scheduled { i.turn() } else { i.clock() };
        let target = pos(&session.inspector);
        session.inspector.seek(0);
        let mut prev: Option<u64> = None;
        while let Stop::Break { .. } = session.inspector.run_until_stop() {
            let t = pos(&session.inspector);
            if t < target {
                prev = Some(t);
            } else {
                break; // reached (or passed) where we started
            }
        }
        let (landed, reason) = match prev {
            Some(t) => (t, "breakpoint"),
            None => (0, "entry"), // no earlier breakpoint: rewound to the start
        };
        session.inspector.seek(landed);
        let tid = session
            .inspector
            .stopped_task()
            .map(|t| t as i64 + 1)
            .unwrap_or(1);
        (true, Json::Null, vec![stopped_event(reason, tid)])
    }

    fn on_evaluate(&mut self, args: Option<&Json>) -> (bool, Json, Vec<Event>) {
        // Watch / hover / REPL / breakpoint-condition expression. A bare known variable keeps its
        // rich form (declared type + formatted value); anything else is an integer expression over
        // the frame's source variables (`expr::eval_int`). Member/index access (`a.b`, `arr[i]`)
        // needs structured type layout — the W4 `TypeRef` follow-up. Failure ⇒ `success:false`.
        let fail = (false, Json::Null, vec![]);
        let expr = args
            .and_then(|a| a.get("expression"))
            .and_then(|e| e.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let Some(frame_id) = args.and_then(|a| a.get("frameId")).and_then(|v| v.as_i64()) else {
            return fail; // no frame context to resolve names in
        };
        let Some(session) = self.session.as_mut() else {
            return fail;
        };
        let Some(&(tid, frame_idx)) = usize::try_from(frame_id)
            .ok()
            .and_then(|i| session.frame_refs.get(i))
        else {
            return fail;
        };
        session.inspector.select_task(tid); // focus the requested thread, then read it immutably
        let session = self.session.as_ref().unwrap();
        let frames = session.inspector.backtrace();
        let Some(func) = frames.get(frame_idx).map(|f| f.pc.func) else {
            return fail;
        };

        // A bare known variable: return its declared type + formatted value (richer than a raw i64).
        let bare = session
            .debug
            .as_ref()
            .and_then(|d| d.vars.iter().find(|v| v.func == func && v.name == expr));
        if let Some(var) = bare {
            if let Some(val) = session
                .inspector
                .read_var(frame_idx, &expr, ty_width(&var.ty))
            {
                return (
                    true,
                    Json::obj(vec![
                        ("result", Json::s(fmt_var(&val))),
                        ("type", Json::s(var.ty.clone())),
                        ("variablesReference", Json::i(0)),
                    ]),
                    vec![],
                );
            }
        }

        // Otherwise evaluate as an integer expression over the frame's variables.
        let resolve = |name: &str| session.resolve_var(frame_idx, func, name);
        match expr::eval_int(&expr, &resolve) {
            Some(v) => (
                true,
                Json::obj(vec![
                    ("result", Json::s(v.to_string())),
                    ("variablesReference", Json::i(0)),
                ]),
                vec![],
            ),
            None => fail,
        }
    }

    fn on_disconnect(&mut self) -> (bool, Json, Vec<Event>) {
        self.terminated = true;
        self.session = None;
        (true, Json::Null, vec![])
    }

    /// Map a resume's outcome to the DAP event(s) that follow the response. The previous stop's
    /// frame references are now stale, so clear them; the next `stackTrace` assigns fresh ones.
    fn stop_events(&mut self, stop: Stop) -> Vec<Event> {
        if let Some(s) = self.session.as_mut() {
            s.frame_refs.clear();
        }
        let tid = self.stopped_thread_id();
        match stop {
            Stop::Break { reason, .. } => vec![stopped_event(dap_reason(reason), tid)],
            Stop::Finished(_) => {
                self.terminated = true;
                vec![("terminated", Json::obj(vec![]))]
            }
            Stop::Blocked => vec![stopped_event("pause", tid)],
        }
    }

    /// The DAP thread id of the stopped thread (vCPU id + 1); `1` in single-threaded mode.
    fn stopped_thread_id(&self) -> i64 {
        self.session
            .as_ref()
            .and_then(|s| s.inspector.stopped_task())
            .map(|t| t as i64 + 1)
            .unwrap_or(1)
    }
}

/// A `stopped` event for `thread_id`.
fn stopped_event(reason: &'static str, thread_id: i64) -> Event {
    (
        "stopped",
        Json::obj(vec![
            ("reason", Json::s(reason)),
            ("threadId", Json::i(thread_id)),
            ("allThreadsStopped", Json::Bool(true)),
        ]),
    )
}

fn dap_reason(r: StopReason) -> &'static str {
    match r {
        StopReason::Breakpoint => "breakpoint",
        StopReason::Step => "step",
        StopReason::Watchpoint { .. } => "data breakpoint",
        StopReason::CapCall { .. } => "pause",
    }
}

/// `(file, line) → smallest IR pc on that line`, the reverse of `Inspector::source_loc`.
fn build_line_index(di: &DebugInfo) -> BTreeMap<(u32, u32), IrPc> {
    let mut idx: BTreeMap<(u32, u32), IrPc> = BTreeMap::new();
    for l in &di.locs {
        let pc = IrPc {
            module: 0,
            func: l.func,
            block: l.block as usize,
            inst: l.inst as usize,
        };
        idx.entry((l.file, l.line))
            .and_modify(|e| {
                if pc < *e {
                    *e = pc;
                }
            })
            .or_insert(pc);
    }
    idx
}

/// Bind a requested line to the nearest line at/after it that has code (so a breakpoint on a blank
/// or comment line snaps forward, like a native debugger). Returns the actual line + its pc.
fn resolve_line(idx: &BTreeMap<(u32, u32), IrPc>, file: u32, line: u32) -> Option<(u32, IrPc)> {
    idx.range((file, line)..=(file, u32::MAX))
        .next()
        .map(|((_, l), pc)| (*l, *pc))
}

/// Match a DAP source path to a debug-info file index: exact, or the debug path is a suffix of it,
/// or they share a basename (DAP paths are usually absolute; debug paths are often basenames).
fn match_file(files: &[String], path: &str) -> Option<u32> {
    files
        .iter()
        .position(|f| f == path || path.ends_with(f.as_str()) || basename(f) == basename(path))
        .map(|i| i as u32)
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Byte width to read for a window-resident variable, inferred from its C type name. Ignored for
/// promoted (SSA) variables.
fn ty_width(ty: &str) -> usize {
    match ty {
        "_Bool" | "bool" | "char" | "signed char" | "unsigned char" => 1,
        "short" | "unsigned short" => 2,
        "int" | "unsigned int" | "unsigned" | "float" => 4,
        _ => 8, // long, long long, pointers, double, long double
    }
}

fn fmt_var(v: &VarValue) -> String {
    match v {
        VarValue::Value(Value::I32(n)) => n.to_string(),
        VarValue::Value(Value::I64(n)) => n.to_string(),
        VarValue::Value(Value::F32(x)) => x.to_string(),
        VarValue::Value(Value::F64(x)) => x.to_string(),
        VarValue::Value(_) => "<value>".to_string(),
        VarValue::Bytes(b) => {
            if b.len() <= 8 {
                let mut x = 0u64;
                for (i, &byte) in b.iter().enumerate() {
                    x |= (byte as u64) << (8 * i);
                }
                x.to_string()
            } else {
                b.iter().map(|byte| format!("{byte:02x}")).collect()
            }
        }
    }
}

/// A source variable's value as an `i64`, for the expression evaluator: scalars directly, a window
/// byte range as a little-endian integer, a float truncated. `None` for a non-integral value.
fn var_to_i64(v: &VarValue) -> Option<i64> {
    Some(match v {
        VarValue::Value(Value::I32(n)) => *n as i64,
        VarValue::Value(Value::I64(n)) => *n,
        VarValue::Value(Value::F32(x)) => *x as i64,
        VarValue::Value(Value::F64(x)) => *x as i64,
        VarValue::Value(_) => return None,
        VarValue::Bytes(b) => {
            let mut x = 0u64;
            for (i, &byte) in b.iter().take(8).enumerate() {
                x |= (byte as u64) << (8 * i);
            }
            x as i64
        }
    })
}

/// Coerce a JSON array of integers to entry-function arguments, one per declared parameter type.
fn coerce_args(args: Option<&Json>, params: &[ValType]) -> Vec<Value> {
    let ints = args.and_then(|a| a.as_array()).unwrap_or(&[]);
    params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let n = ints.get(i).and_then(|v| v.as_i64()).unwrap_or(0);
            match ty {
                ValType::I64 => Value::I64(n),
                ValType::F32 => Value::F32(n as f32),
                ValType::F64 => Value::F64(n as f64),
                _ => Value::I32(n as i32),
            }
        })
        .collect()
}

/// Run the DAP wire protocol over stdin/stdout: `Content-Length`-framed JSON messages, the
/// transport a real client (VS Code) connects to. Loops until the session terminates or EOF.
pub fn run_stdio() -> std::io::Result<()> {
    use std::io::{BufRead, Read, Write};
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let stdout = std::io::stdout();
    let mut server = DapServer::new();

    loop {
        // Read headers until the blank line; we only need Content-Length.
        let mut content_len: Option<usize> = None;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                return Ok(()); // EOF
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                content_len = rest.trim().parse().ok();
            }
        }
        let Some(len) = content_len else { continue };
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        let Ok(body) = String::from_utf8(buf) else {
            continue;
        };
        let Some(request) = json::parse(&body) else {
            continue;
        };

        for msg in server.handle(&request) {
            let text = msg.to_string();
            let mut out = stdout.lock();
            write!(out, "Content-Length: {}\r\n\r\n{}", text.len(), text)?;
            out.flush()?;
        }
        if server.is_terminated() {
            return Ok(());
        }
    }
}
