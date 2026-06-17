//! A **stateful REPL** built on in-window dynamic linking — the flagship use case made concrete
//! (DYNLINK.md). Each "prompt" defines a function that may **call earlier definitions by name**; the
//! REPL keeps a **symbol table** (`name → installed table slot`) that grows with every definition,
//! resolves each new unit's imports against it, compiles + installs the unit, and registers its name.
//! Later prompts then reach it through the shared `call_indirect` table.
//!
//! This is exactly what a guest-side `vm_dlopen`/`vm_dlsym` REPL will do — the symbol table *is* the
//! dlopen registry, `define` *is* `vm_dlopen` (resolve → verify → compile → install → record), and an
//! `eval` is a `vm_dlsym` + call. It runs **today** on the C1 host-assisted primitives
//! (`svm_run::jit_resolve_and_validate` + the guest-JIT `define_extra`/`install`), with no new cap-op
//! plumbing — so it serves as the *executable spec* for the C `compile_linked` op and the dlopen
//! surface that will turn this harness into a real guest program.
//!
//! Contrast `demos/jit/jit_repl.c`, today's guest REPL: it JITs a **standalone** expression each
//! prompt and throws it away. Here definitions **persist and compose by name** — the dynamic-linking
//! difference, in one test.

use std::collections::HashMap;

use svm_encode::encode_module;
use svm_ir::{Resolved, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_run::jit_resolve_and_validate;
use svm_text::parse_module;
use svm_verify::verify_module;

/// A live REPL: one long-lived JIT module with a reserved `call_indirect` table (definitions install
/// into its padding slots) plus the growing symbol table. `name → (slot, trampoline)`: the slot lets
/// *other* definitions reach it by `call_indirect`; the trampoline lets the REPL itself `eval` it.
struct Repl {
    cm: CompiledModule,
    symbols: HashMap<String, (u32, *const u8)>,
}

impl Repl {
    fn new() -> Self {
        // Module 0 is a trivial host (an identity entry); all the interesting code is defined at
        // runtime. A 16-slot table (log2 = 4) gives definitions room to install.
        let m = parse_module("func (i32) -> (i32) {\nblock0(v0: i32):\n  return v0\n}\n").unwrap();
        verify_module(&m).unwrap();
        let cm = CompiledModule::compile(
            &m,
            0,
            INERT_CAP_THUNK,
            core::ptr::null_mut(),
            DEFAULT_RESERVED_LOG2,
            None,
            None,
            None,
            None,
            Quota::default(),
            4,
        )
        .expect("compile the REPL host module");
        Repl {
            cm,
            symbols: HashMap::new(),
        }
    }

    /// A REPL `def NAME = <unit>`: the unit's text may `call.import` any **earlier** definition by
    /// name. Resolve those names against the live symbol table (→ their table slots), re-verify,
    /// compile into the live window, install into the shared table, and register the name. This is
    /// `vm_dlopen` in miniature — and a reference to an *unknown* name fails closed (panics here).
    fn define(&mut self, name: &str, src: &str) {
        // Serialize the unit with its symbols still unresolved (the `.so` a guest would ship), then
        // let the host bind them — no `resolve_imports_with` in this harness; the host does it.
        let blob = encode_module(&parse_module(src).expect("parse definition"));
        let table = self.symbols.clone();
        let funcs = jit_resolve_and_validate(&blob, None, |n| {
            table.get(n).map(|&(slot, _)| Resolved::Slot(slot))
        })
        .expect("resolve the definition's imports against the REPL symbol table");

        let defs = self
            .cm
            .define_extra(&funcs)
            .expect("compile the definition");
        let slot = self
            .cm
            .install(defs[0].code, defs[0].type_id)
            .expect("install the definition into the shared table");
        // A redefinition would shadow the old name (the old slot stays installed — old callers keep
        // their binding); new callers resolve to the new slot. That's the hot-reload shape, for free.
        self.symbols.insert(name.to_string(), (slot, defs[0].tramp));
    }

    /// A REPL `eval NAME(x)`: `vm_dlsym` the name to its trampoline and call it over the live window.
    fn eval1(&mut self, name: &str, x: i32) -> i32 {
        let &(_, tramp) = self.symbols.get(name).expect("eval of an undefined name");
        let (out, _) = unsafe { self.cm.run_extra(tramp, 1, 1, &[x as i64], None) }
            .expect("invoke the definition");
        match out {
            JitOutcome::Returned(ref s) if s.len() == 1 => s[0] as i32,
            other => panic!("definition {name} did not return one i32: {other:?}"),
        }
    }
}

/// The transcript:
/// ```text
/// > def sq(x)       = x * x
/// > def quad(x)     = sq(sq(x))           # links to sq by name
/// > def quad_plus(x)= quad(x) + sq(x)     # links to quad AND sq by name
/// > sq(5)           => 25
/// > quad(3)         => 81
/// > quad_plus(2)    => 20                 # 2^4 + 2^2 = 16 + 4
/// ```
/// Every later definition reaches earlier ones purely through the symbol table — the essence of a
/// dynamically-linked, incrementally-built program.
#[test]
fn repl_definitions_persist_and_compose_by_name() {
    let mut repl = Repl::new();

    // > def sq(x) = x * x   — a leaf definition (no imports).
    repl.define(
        "sq",
        "func (i32) -> (i32) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i32.mul v0 v0\n\
         \x20 return v1\n\
         }\n",
    );

    // > def quad(x) = sq(sq(x))   — two `call.import \"sq\"`, each resolved to sq's slot.
    repl.define(
        "quad",
        "func (i32) -> (i32) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i32.const 0\n\
         \x20 v2 = call.import \"sq\" (i32) -> (i32) v1 (v0)\n\
         \x20 v3 = i32.const 0\n\
         \x20 v4 = call.import \"sq\" (i32) -> (i32) v3 (v2)\n\
         \x20 return v4\n\
         }\n",
    );

    // > def quad_plus(x) = quad(x) + sq(x)   — links to TWO distinct prior names in one unit.
    repl.define(
        "quad_plus",
        "func (i32) -> (i32) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i32.const 0\n\
         \x20 v2 = call.import \"quad\" (i32) -> (i32) v1 (v0)\n\
         \x20 v3 = i32.const 0\n\
         \x20 v4 = call.import \"sq\" (i32) -> (i32) v3 (v0)\n\
         \x20 v5 = i32.add v2 v4\n\
         \x20 return v5\n\
         }\n",
    );

    // The symbol table grew to three entries, each at its own table slot.
    assert_eq!(
        repl.symbols.len(),
        3,
        "three definitions registered by name"
    );

    assert_eq!(repl.eval1("sq", 5), 25, "sq(5)");
    assert_eq!(repl.eval1("quad", 3), 81, "quad(3) = sq(sq(3)) = 81");
    assert_eq!(
        repl.eval1("quad_plus", 2),
        20,
        "quad_plus(2) = quad(2) + sq(2) = 16 + 4"
    );
}

/// Fail-closed REPL: referencing a name that was never defined is rejected at `define` time (the
/// host can't bind it), never a silent miscompile — so a typo at the prompt is a clean error, not a
/// call into nowhere.
#[test]
#[should_panic(expected = "resolve the definition's imports")]
fn defining_against_an_unknown_name_fails_closed() {
    let mut repl = Repl::new();
    // `cube` references `sq`, which has not been defined — resolution fails closed.
    repl.define(
        "cube",
        "func (i32) -> (i32) {\n\
         block0(v0: i32):\n\
         \x20 v1 = i32.const 0\n\
         \x20 v2 = call.import \"sq\" (i32) -> (i32) v1 (v0)\n\
         \x20 v3 = i32.mul v2 v0\n\
         \x20 return v3\n\
         }\n",
    );
}
