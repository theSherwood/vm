//! `blockaddress` recovery via `llvm-sys` (the computed-`goto` half of the on-ramp).
//!
//! `llvm-ir` 0.11.3 erases the operands of a `blockaddress` constant — `Constant::BlockAddress` is
//! payloadless (the C-API getters were thought unavailable). But the LLVM-C API *does* expose them
//! (`LLVMGetOperand` + `LLVMValueAsBasicBlock`), so — exactly as [`crate::di`] does for the debug-info
//! graph — we re-parse the `.bc` through `llvm-sys` and recover, for each global initializer, the
//! `blockaddress` constants it holds.
//!
//! A `blockaddress(@f, %bb)` lowers to the **index of `%bb` within `@f`** (matching `block_idx` on the
//! `llvm-ir` side — both walk a function's basic blocks in definition order). That small integer is
//! what the guest stores in its dispatch table (`static void *tbl[] = {&&l0, &&l1, …}`) and what an
//! `indirectbr` (`goto *p`) consumes as a `br_table` index (see `translate_indirectbr` in `lib.rs`).
//!
//! The labels are returned **per global, in depth-first initializer order** — the same order
//! `const_bytes` serializes the initializer — so the serializer pops them positionally (the `di.rs`
//! ordinal-correlation discipline), no fragile name/offset matching.

use std::collections::HashMap;
use std::ffi::{c_char, CString};

use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::LLVMValueKind;

/// Per global-variable **name**, the block-index labels of the `blockaddress` constants in its
/// initializer, in the depth-first order `const_bytes` visits them.
#[derive(Default)]
pub struct BlockAddrs {
    pub per_global: HashMap<String, Vec<u32>>,
    /// Operand-position `blockaddress`es — clang's jump-threading can thread one through a φ (an
    /// instruction operand, not a global). Keyed positionally `(func_idx, block_idx, phi_ord,
    /// incoming_idx)` → target block index — the `di.rs` ordinal discipline (φ results / blocks are
    /// usually *unnamed*, so name-keying is impossible). `func_idx` is the **defined**-function index
    /// (declarations skipped), matching `lib.rs`'s `defined`/`name2idx`; `phi_ord` counts φs within the
    /// block; `incoming_idx` indexes the φ's `incoming_values`.
    pub phi: HashMap<(u32, u32, u32, u32), u32>,
}

/// Recover the module's `blockaddress` labels from the bitcode at `path`. `None` if the file can't be
/// read/parsed or holds no `blockaddress` (the common case — the cost is one extra parse only when a
/// program actually uses computed `goto`; for everything else the map is empty and unused).
pub fn read_block_addrs(path: &str) -> Option<BlockAddrs> {
    unsafe { read_unsafe(path) }
}

unsafe fn read_unsafe(path: &str) -> Option<BlockAddrs> {
    let ctx = LLVMContextCreate();
    let result = (|| {
        let cpath = CString::new(path).ok()?;
        let mut buf: LLVMMemoryBufferRef = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        if LLVMCreateMemoryBufferWithContentsOfFile(cpath.as_ptr(), &mut buf, &mut err) != 0 {
            return None;
        }
        let mut module: LLVMModuleRef = std::ptr::null_mut();
        if LLVMParseBitcodeInContext2(ctx, buf, &mut module) != 0 {
            return None;
        }
        let mut out = BlockAddrs::default();
        let mut g = LLVMGetFirstGlobal(module);
        while !g.is_null() {
            let init = LLVMGetInitializer(g);
            if !init.is_null() {
                let mut labels = Vec::new();
                collect(init, &mut labels);
                if !labels.is_empty() {
                    out.per_global.insert(value_name(g), labels);
                }
            }
            g = LLVMGetNextGlobal(g);
        }
        collect_phi_addrs(module, &mut out.phi);
        (!out.per_global.is_empty() || !out.phi.is_empty()).then_some(out)
    })();
    LLVMContextDispose(ctx);
    result
}

/// Depth-first walk of a constant initializer, mirroring `const_bytes`' recursion (arrays / vectors /
/// structs recurse into their elements in order; every other constant is a leaf). A `blockaddress`
/// leaf contributes its target block index; any other leaf contributes nothing.
unsafe fn collect(v: LLVMValueRef, out: &mut Vec<u32>) {
    match LLVMGetValueKind(v) {
        LLVMValueKind::LLVMConstantArrayValueKind
        | LLVMValueKind::LLVMConstantStructValueKind
        | LLVMValueKind::LLVMConstantVectorValueKind => {
            let n = LLVMGetNumOperands(v);
            for i in 0..n {
                collect(LLVMGetOperand(v, i as u32), out);
            }
        }
        // A `ConstantDataArray`/`ConstantAggregateZero`/scalar leaf — no nested operands to a
        // `blockaddress` (a packed data array holds only ints/floats; a zeroinitializer holds none).
        _ => {
            if !LLVMIsABlockAddress(v).is_null() {
                if let Some(idx) = block_index(v) {
                    out.push(idx);
                }
            }
        }
    }
}

/// Walk every defined function's φ nodes, recording each `blockaddress` incoming value keyed by its
/// position `(func_idx, block_idx, phi_ord, incoming_idx)` — the order `branch_args` reconstructs on
/// the `llvm-ir` side. `func_idx` counts only **defined** functions (those with a body), in module
/// order, exactly as `lib.rs`'s `defined` list does.
unsafe fn collect_phi_addrs(module: LLVMModuleRef, out: &mut HashMap<(u32, u32, u32, u32), u32>) {
    let mut func_idx = 0u32;
    let mut f = LLVMGetFirstFunction(module);
    while !f.is_null() {
        let mut bb = LLVMGetFirstBasicBlock(f);
        if bb.is_null() {
            // A declaration (no body) — not in the `defined` index space; skip without consuming one.
            f = LLVMGetNextFunction(f);
            continue;
        }
        let mut block_idx = 0u32;
        while !bb.is_null() {
            let mut phi_ord = 0u32;
            let mut inst = LLVMGetFirstInstruction(bb);
            while !inst.is_null() {
                if !LLVMIsAPHINode(inst).is_null() {
                    let n = LLVMCountIncoming(inst);
                    for i in 0..n {
                        let val = LLVMGetIncomingValue(inst, i);
                        if !LLVMIsABlockAddress(val).is_null() {
                            if let Some(target) = block_index(val) {
                                out.insert((func_idx, block_idx, phi_ord, i), target);
                            }
                        }
                    }
                    phi_ord += 1;
                }
                inst = LLVMGetNextInstruction(inst);
            }
            block_idx += 1;
            bb = LLVMGetNextBasicBlock(bb);
        }
        func_idx += 1;
        f = LLVMGetNextFunction(f);
    }
}

/// The index of the basic block a `blockaddress` targets, within its parent function — matching the
/// `llvm-ir` `block_idx` order (both enumerate the function's blocks in definition order). The
/// `BlockAddress` constant's operands are `[function, basic-block]`.
unsafe fn block_index(ba: LLVMValueRef) -> Option<u32> {
    let bb_val = LLVMGetOperand(ba, 1);
    if LLVMValueIsBasicBlock(bb_val) == 0 {
        return None;
    }
    let bb = LLVMValueAsBasicBlock(bb_val);
    let func = LLVMGetBasicBlockParent(bb);
    let mut idx = 0u32;
    let mut cur = LLVMGetFirstBasicBlock(func);
    while !cur.is_null() {
        if cur == bb {
            return Some(idx);
        }
        idx += 1;
        cur = LLVMGetNextBasicBlock(cur);
    }
    None
}

unsafe fn value_name(v: LLVMValueRef) -> String {
    let mut len = 0usize;
    let p = LLVMGetValueName2(v, &mut len);
    if p.is_null() || len == 0 {
        String::new()
    } else {
        String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len)).into_owned()
    }
}
