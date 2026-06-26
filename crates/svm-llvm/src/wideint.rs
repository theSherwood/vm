//! Fail-closed guard for **wide integer constants** that `llvm-ir` 0.11.3 silently truncates (I14).
//!
//! Upstream reads every integer constant through `LLVMConstIntGetZExtValue`, which returns a **`u64`**.
//! For a `bits > 64` literal that drops the high bits — and on a *no-asserts* libLLVM (Ubuntu's
//! `llvm-18` is `--assertion-mode OFF`) it does so **silently**, so an `i128` constant `>= 2^64` or
//! negative would reach the on-ramp as just its low word and **miscompile** (e.g. `x % (2^64+1)` lowered
//! as `x % 1`). The truncation already happened by the time we hold the `llvm-ir` AST, so — exactly as
//! [`crate::blockaddr`] / [`crate::di`] re-read the `.bc` through `llvm-sys` for what the AST can't
//! express — we walk the module's values and **reject the whole translate** (a clean `Unsupported`,
//! never a miscompile) if any integer constant wider than 64 bits is outside `[0, 2^64)`. Constants
//! that *do* fit round-trip exactly from their (exact) low word, so only the genuinely-unrepresentable
//! case fail-closes. (Supporting such constants would need the high word — i.e. patching `llvm-ir`,
//! deliberately not done; see I14.)

use std::ffi::{c_char, CStr, CString};

use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
use llvm_sys::core::*;
use llvm_sys::prelude::*;

/// If the bitcode at `path` contains an integer constant wider than 64 bits whose value is outside
/// `[0, 2^64)` (negative, or needing the high word), return a short description of the first such
/// constant (for the `Unsupported` message). `None` when the module has none (the common case) or
/// can't be read/parsed — the real parse in `translate_bc_path` reports any parse error.
pub fn out_of_range_constant(path: &str) -> Option<String> {
    unsafe { scan(path) }
}

unsafe fn scan(path: &str) -> Option<String> {
    let ctx = LLVMContextCreate();
    let found = (|| {
        let cpath = CString::new(path).ok()?;
        let mut buf: LLVMMemoryBufferRef = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        if LLVMCreateMemoryBufferWithContentsOfFile(cpath.as_ptr(), &mut buf, &mut err) != 0 {
            return None;
        }
        // `LLVMParseBitcodeInContext2` takes ownership of `buf`.
        let mut module: LLVMModuleRef = std::ptr::null_mut();
        if LLVMParseBitcodeInContext2(ctx, buf, &mut module) != 0 {
            return None;
        }
        // Every instruction operand (across all functions) and every global initializer.
        let mut f = LLVMGetFirstFunction(module);
        while !f.is_null() {
            let mut bb = LLVMGetFirstBasicBlock(f);
            while !bb.is_null() {
                let mut inst = LLVMGetFirstInstruction(bb);
                while !inst.is_null() {
                    for o in 0..LLVMGetNumOperands(inst) {
                        if let Some(s) = check_const(LLVMGetOperand(inst, o as u32)) {
                            return Some(s);
                        }
                    }
                    inst = LLVMGetNextInstruction(inst);
                }
                bb = LLVMGetNextBasicBlock(bb);
            }
            f = LLVMGetNextFunction(f);
        }
        let mut g = LLVMGetFirstGlobal(module);
        while !g.is_null() {
            let init = LLVMGetInitializer(g);
            if !init.is_null() {
                if let Some(s) = check_const(init) {
                    return Some(s);
                }
            }
            g = LLVMGetNextGlobal(g);
        }
        None
    })();
    LLVMContextDispose(ctx);
    found
}

/// If `v` is a **constant**, check it (and nested constant aggregates/expressions) for an out-of-range
/// wide integer; a non-constant operand (another instruction, an argument) is ignored. Bounded: the
/// constant DAG is finite and `ConstantInt`/leaves have no operands.
unsafe fn check_const(v: LLVMValueRef) -> Option<String> {
    if v.is_null() || LLVMIsAConstant(v).is_null() {
        return None;
    }
    if !LLVMIsAConstantInt(v).is_null() {
        let bits = LLVMGetIntTypeWidth(LLVMTypeOf(v));
        if bits > 64 && wide_value(v).is_some_and(|val| !(0..1i128 << 64).contains(&val)) {
            // Render `iN <value>` for the diagnostic (the textual form is already what we parsed).
            let s = LLVMPrintValueToString(v);
            let text = CStr::from_ptr(s).to_string_lossy().trim().to_string();
            LLVMDisposeMessage(s);
            return Some(text);
        }
        return None;
    }
    // A **global value** (global var / function / alias) is an *address*, not nested literal data —
    // don't descend (its operands include its own initializer, which can reference it back → infinite
    // recursion on self-referential statics like vtables). Only constant aggregates / expressions below.
    if !LLVMIsAGlobalValue(v).is_null() {
        return None;
    }
    // A constant aggregate (`{…}`, `[…]`, `<…>`) or constant expression: recurse its elements.
    for o in 0..LLVMGetNumOperands(v) {
        if let Some(s) = check_const(LLVMGetOperand(v, o as u32)) {
            return Some(s);
        }
    }
    None
}

/// The full value of a `ConstantInt` (`bits > 64`) as an `i128`, parsed from its textual form
/// (`iN <signed-decimal>`) — LLVM-C 18 has no wide-int getter, and a high-bit-set value prints as a
/// negative decimal, so every `i128` literal fits an `i128` and parses cleanly. `None` on the
/// (unexpected) parse failure, treated as "in range" so a malformed render never fails-closed spuriously.
unsafe fn wide_value(v: LLVMValueRef) -> Option<i128> {
    let s = LLVMPrintValueToString(v);
    let text = CStr::from_ptr(s).to_string_lossy().into_owned();
    LLVMDisposeMessage(s);
    text.split_whitespace().last()?.parse::<i128>().ok()
}
