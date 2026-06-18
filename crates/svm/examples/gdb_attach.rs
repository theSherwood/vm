//! Manual acceptance harness for the W5 JIT/DWARF tier, Stage 2c (DEBUGGING.md §7): compile a `-g`
//! module, register its synthesized DWARF with the GDB JIT interface, and run the JIT'd code — so a
//! real gdb can bind a source-line breakpoint inside native JIT'd guest code and show the frame.
//!
//! This is the **manual** half of Stage 2c's acceptance (the CI half is `tests/jit_srcloc.rs`). Run
//! it under gdb in batch mode:
//!
//! ```text
//! cargo build --example gdb_attach -p svm
//! gdb --batch \
//!   -ex 'set breakpoint pending on' \
//!   -ex 'break compute.c:3' \
//!   -ex run -ex bt -ex continue \
//!   target/debug/examples/gdb_attach
//! ```
//!
//! gdb plants its internal breakpoint on `__jit_debug_register_code`; when this program registers,
//! gdb reads the in-memory ELF symfile, resolves `compute.c:3` to the live machine address, and
//! stops there when the JIT'd function executes.

use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{CompiledModule, Quota, INERT_CAP_THUNK};
use svm_text::parse_module;

// Same fixture as the CI test: three computing ops mapped to source lines 2, 3, 4 of "compute.c".
const COMPUTE_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  v3 = i32.const 3
  v4 = i32.mul v2 v3
  v5 = i32.const 2
  v6 = i32.sub v4 v5
  return v6
}

debug.file 0 "compute.c"
debug.loc 0 0 1 0 2 7
debug.loc 0 0 3 0 3 7
debug.loc 0 0 5 0 4 3
"#;

fn main() {
    let m = parse_module(COMPUTE_DBG).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
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
        0,
    )
    .expect("jit compiles");

    // Register with gdb. The guard must outlive the run() below (and is independent of `cm`).
    let _reg = cm.register_with_gdb().expect("a -g module registers");

    // For offline validation of the symfile (readelf / llvm-dwarfdump), dump the exact ELF bytes
    // gdb is handed when $GDB_ATTACH_DUMP is set.
    if let Ok(path) = std::env::var("GDB_ATTACH_DUMP") {
        std::fs::write(&path, cm.elf_object()).expect("dump elf");
        eprintln!("[harness] wrote ELF symfile to {path}");
    }

    for r in cm.src_ranges() {
        eprintln!(
            "[harness] compute.c:{} col {} -> [{:#x}, {:#x})",
            r.line, r.col, r.lo, r.hi
        );
    }

    // Execute the JIT'd function so a source-line breakpoint inside it is actually hit.
    let (outcome, _mem) = cm.run(&[10], None, None, None).expect("run");
    eprintln!("[harness] compute(10) -> {outcome:?}"); // (10+1)*3 - 2 = 31
}
