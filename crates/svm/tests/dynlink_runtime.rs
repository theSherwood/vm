//! Dynamic linking, end to end: a **plugin compiled at runtime** calls into the **host program**
//! through the shared `call_indirect` table, with the callee resolved by **name** to its table slot.
//!
//! This is the genuinely *dynamic* case (vs. the static linker in `dynlink.rs`): the host is already
//! compiled and running; the plugin is a *separately compiled* unit (the guest-JIT incremental path,
//! `CompiledModule::define_extra`, §22 Model A — extra code reaches the parent's table by slot); and
//! the plugin was authored against the host function purely by name, the loader binding it to the
//! slot (`Resolved::Slot`) before compiling the plugin. The plugin doesn't share a function-index
//! space with the host — it reaches it only through the table, exactly like a loaded `.so` calling
//! into the program that `dlopen`ed it.

use svm_ir::{Resolved, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_text::parse_module;
use svm_verify::verify_module;

fn compile_host(src: &str) -> CompiledModule {
    let m = parse_module(src).expect("parse host");
    verify_module(&m).expect("verify host");
    CompiledModule::compile(
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
        0,
    )
    .expect("compile host program")
}

#[test]
fn plugin_calls_host_program_by_resolved_slot() {
    // The host program: `F(a,b) = a*2 + b`, at function/table slot 0 (also the entry).
    let mut cm = compile_host(
        "func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
         \x20 v2 = i32.const 2\n\
         \x20 v3 = i32.mul v0 v2\n\
         \x20 v4 = i32.add v3 v1\n\
         \x20 return v4\n\
           }\n\
         }\n",
    );

    // The plugin, authored against `F` purely **by name** — it has no idea what slot F is at.
    let plugin = parse_module(
        "func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
         \x20 v2 = i32.const 0\n\
         \x20 v3 = call.sym \"F\" (i32, i32) -> (i32) v2 (v0, v1)\n\
         \x20 return v3\n\
           }\n\
         }\n",
    )
    .expect("parse plugin");
    assert_eq!(plugin.imports.len(), 1, "the plugin imports F by name");

    // The link step: the loader knows F lives at the host's table slot 0 and binds the plugin's
    // import to it — `call.sym "F"` → `call_indirect 0`.
    let linked = svm_ir::resolve_imports_with(&plugin, |n| (n == "F").then_some(Resolved::Slot(0)))
        .expect("resolve plugin import to the host's slot");
    verify_module(&linked).expect("verify the linked plugin");

    // Compile the plugin **at runtime** against the host's live table, then call it. It dispatches
    // through the shared table to the host's F: F(10,3) = 23 (a direct call to its own funcs[0] would
    // be impossible — the plugin has no F of its own).
    let ptrs = cm
        .define_extra(&linked.funcs)
        .expect("define_extra (compile the plugin)");
    let (out, _) =
        unsafe { cm.run_extra(ptrs[0].tramp, 2, 1, &[10, 3], None) }.expect("run plugin");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[23]),
        "the plugin reached the host's F through the table: expected 23, got {out:?}"
    );
}

/// The symbol-resolved companion to `jit_incremental::install_makes_unit_call_indirectable` (which
/// proves the install *mechanism* — old code `call_indirect`s a runtime slot value into newly-
/// installed code). Here a separately-compiled **client** references a **newly-installed service**
/// purely **by name**: the loader installs the service (getting its table slot), resolves the
/// client's import to that slot (`Resolved::Slot`), and the client — compiled afterwards — reaches
/// the installed service through the shared table. Both are runtime-loaded units; the callee is found
/// by symbol, dispatched by slot. `service(a,b) = a*a + b`, so `service(5,2) = 27`.
#[test]
fn loaded_client_links_to_a_newly_installed_service_by_name() {
    // A host reserving a 16-slot table (log2 = 4) so units can be installed into the padding slots.
    let m =
        parse_module("func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n").unwrap();
    verify_module(&m).unwrap();
    let mut cm = CompiledModule::compile(
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
    .expect("compile host");

    // Compile + install the "service" at runtime → its table slot (slot 1, past the host's 1 func).
    let service = parse_module(
        "func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
         \x20 v2 = i32.mul v0 v0\n\
         \x20 v3 = i32.add v2 v1\n\
         \x20 return v3\n\
           }\n\
         }\n",
    )
    .unwrap();
    verify_module(&service).unwrap();
    let svc = cm.define_extra(&service.funcs).expect("compile service");
    let slot = cm
        .install(svc[0].code, svc[0].type_id)
        .expect("install service");

    // The client references "service" by name; the loader binds it to the install slot, then compiles
    // the client — which now `call_indirect`s the slot into the installed service.
    let client = parse_module(
        "func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
         \x20 v2 = i32.const 0\n\
         \x20 v3 = call.sym \"service\" (i32, i32) -> (i32) v2 (v0, v1)\n\
         \x20 return v3\n\
           }\n\
         }\n",
    )
    .unwrap();
    let linked = svm_ir::resolve_imports_with(&client, |n| {
        (n == "service").then_some(Resolved::Slot(slot))
    })
    .expect("resolve client → install slot");
    verify_module(&linked).expect("verify client");
    let cli = cm.define_extra(&linked.funcs).expect("compile client");
    let (out, _) = unsafe { cm.run_extra(cli[0].tramp, 2, 1, &[5, 2], None) }.expect("run client");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[27]),
        "client reached the newly-installed service by name: expected 27, got {out:?}"
    );
}
