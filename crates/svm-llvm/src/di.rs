//! The §6 debug-info **data model** — the structured variable/type half of the frontend-neutral
//! waist ([`LlvmDebug`]), consumed by the translator and *built by the in-house textual reader*
//! ([`crate::ll::debug`]). (Historically this module also held an `llvm-sys` walk of the DI metadata
//! graph — the `llvm-ir` AST left it unimplemented; that reader is gone now that the `.ll` reader
//! recovers the same structure from text, so no libLLVM is linked.)

use std::collections::HashMap;

use svm_ir::{TypeDef, TypeId};

/// How a recovered variable's value is located, in terms the translator can correlate to the IR.
pub enum DiLoc {
    /// A `-O0` `llvm.dbg.declare` memory variable: the **alloca ordinal** (the Nth alloca in
    /// textual order, the correlation key into the translator's frame layout) → a `Window` slot.
    Window { alloca_ordinal: usize },
    /// A `-O2`/`-Og` `llvm.dbg.value` binding to **function argument `index`** — a stable SSA value
    /// the translator threads as a block parameter, so it resolves to an `SsaList` over the arg's
    /// live range. (Bindings to non-argument SSA values / constants / `poison` are skipped — the
    /// optimizer didn't keep a recoverable location there; instruction-result bindings are a
    /// follow-up.)
    Arg { index: u32 },
}

/// One source variable recovered from the DI metadata: its name, structured [`TypeId`], render
/// name, and how to locate it ([`DiLoc`]).
pub struct DiVar {
    pub name: String,
    pub loc: DiLoc,
    pub type_id: Option<TypeId>,
    /// The neutral render name (e.g. `"int"`, `"struct Point"`) — always present, even when the
    /// structured `type_id` could not be built.
    pub ty: String,
    /// The §6 lexical scope `(start_line, end_line)` when the variable lives in a `DILexicalBlock`
    /// (the shadowing case); `None` ⇒ scoped to the subprogram (function-wide). `end_line` is the
    /// last source line of instructions in the block (`DILexicalBlock` carries no end line).
    pub scope: Option<(u32, u32)>,
}

/// A module-scoped source global recovered from a global's `!dbg` `DIGlobalVariableExpression`:
/// its source `name`, the LLVM `symbol` (the correlation key into the translator's globals layout →
/// a window address), and its structured [`TypeId`].
pub struct DiGlobal {
    pub symbol: String,
    pub name: String,
    pub type_id: Option<TypeId>,
    pub ty: String,
}

/// The reader's result: a structured `types` table (the §6 `TypeDef` graph), the per-function local
/// variable lists (keyed by LLVM function name = the translator's `Function::name`), and the
/// module-scoped globals.
#[derive(Default)]
pub struct LlvmDebug {
    pub types: Vec<TypeDef>,
    pub vars: HashMap<String, Vec<DiVar>>,
    pub globals: Vec<DiGlobal>,
    /// `DISubprogram` source names, keyed by the function's LLVM value (linkage) name — the §6
    /// function-name table the translator correlates to IR function indices. Every function with a
    /// subprogram, whether or not it has tracked variables.
    pub func_names: HashMap<String, String>,
}
