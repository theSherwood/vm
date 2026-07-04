//! The `blockaddress` recovery **data model** — the computed-`goto` half of the on-ramp.
//!
//! `Constant::BlockAddress` is payloadless in the IR AST (the `(@f, %bb)` operands are dropped), so a
//! `blockaddress`'s target block index is recovered out of band into [`BlockAddrs`] and correlated to
//! the AST leaves positionally. The in-house textual reader ([`crate::ll::parse`]) fills this from the
//! `.ll` text; this module historically also held an `llvm-sys` `.bc` reader, gone now that the
//! textual reader recovers the same structure (no libLLVM linked).

use std::collections::HashMap;

/// Per global-variable **name**, the block-index labels of the `blockaddress` constants in its
/// initializer, in the depth-first order `const_bytes` visits them.
#[derive(Default)]
pub struct BlockAddrs {
    pub per_global: HashMap<String, Vec<u32>>,
    /// Operand-position `blockaddress`es — clang's jump-threading can thread one through a φ (an
    /// instruction operand, not a global). Keyed positionally `(func_idx, block_idx, phi_ord,
    /// incoming_idx)` → target block index — the ordinal-correlation discipline (φ results / blocks are
    /// usually *unnamed*, so name-keying is impossible). `func_idx` is the **defined**-function index
    /// (declarations skipped), matching `lib.rs`'s `defined`/`name2idx`; `phi_ord` counts φs within the
    /// block; `incoming_idx` indexes the φ's `incoming_values`.
    pub phi: HashMap<(u32, u32, u32, u32), u32>,
}
