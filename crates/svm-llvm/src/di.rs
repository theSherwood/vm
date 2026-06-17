//! The LLVM **debug-info variable/type reader** (DEBUGGING.md §6 / D-DBG-7, slice 25): a direct
//! `llvm-sys` walk of the DI metadata graph that the pinned `llvm-ir` AST reader does not expose
//! (`Metadata::from_llvm_ref` is `unimplemented!`, `MetadataOperand` is payloadless). It recovers
//! each source local's **name + structured type** and correlates it to the SVM IR, feeding the
//! *variable/type half* of the frontend-neutral waist — the LLVM analog of the wasm producer's
//! `dwarf_info` ingest.
//!
//! **Scope (this slice): `-O0 -g` memory variables.** At `-O0` clang keeps every C local as an
//! `alloca` plus an `llvm.dbg.declare(addr, !DILocalVariable, !DIExpression)`. The address operand
//! pins the alloca, whose **ordinal** (the Nth alloca in textual order) is stable across this parse
//! and the translator's own walk — so it correlates to the alloca's data-stack frame slot, yielding
//! a [`VarLoc::Window`]. (The `-O2`/`-Og` `llvm.dbg.value` SSA-location-list case — where LLVM
//! solves S2's promotion-vs-inspectability for free — is a follow-up slice; it maps onto the
//! existing `SsaList`/`WindowVia` machinery the wasm producer already exercises.)
//!
//! **What the LLVM-C DI API does / doesn't give us.** Typed getters cover `name`, `size`, `offset`,
//! and the DWARF `tag`; the **`baseType`/`elements` edges and the base-type `encoding` are not
//! exposed**, so the type graph is walked through the generic MDNode-operand bridge
//! (`LLVMMetadataAsValue` → `LLVMGetMDNodeOperands` → `LLVMValueAsMetadata`) at the positional
//! indices LLVM 18 uses, and `encoding` is inferred from the primitive's name (the C-type heuristic,
//! consistent with the existing `ty_width` name-fallback). Both are pinned by the LLVM-18 dependency
//! and covered by tests; an off-version reader is rejected upstream (`Error::Parse`).

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_char;

use llvm_sys::bit_reader::LLVMParseBitcodeInContext2;
use llvm_sys::core::*;
use llvm_sys::debuginfo::*;
use llvm_sys::prelude::*;

use svm_ir::{Encoding, Field, TypeDef, TypeId};

/// DWARF tags (the subset the reader dispatches on).
mod tag {
    pub const ARRAY_TYPE: u16 = 0x01;
    pub const MEMBER: u16 = 0x0d;
    pub const POINTER_TYPE: u16 = 0x0f;
    pub const UNION_TYPE: u16 = 0x17;
    pub const TYPEDEF: u16 = 0x16;
    pub const CONST_TYPE: u16 = 0x26;
    pub const VOLATILE_TYPE: u16 = 0x35;
    pub const RESTRICT_TYPE: u16 = 0x37;
}

/// One source variable recovered from an `llvm.dbg.declare`: its name, the **alloca ordinal** (the
/// correlation key into the translator's frame layout), and its structured [`TypeId`].
pub struct DiVar {
    pub name: String,
    pub alloca_ordinal: usize,
    pub type_id: Option<TypeId>,
    /// The neutral render name (e.g. `"int"`, `"struct Point"`) — always present, even when the
    /// structured `type_id` could not be built.
    pub ty: String,
}

/// The reader's result: a structured `types` table (the §6 `TypeDef` graph) plus the per-function
/// variable lists, keyed by LLVM function name (= the translator's `Function::name`).
#[derive(Default)]
pub struct LlvmDebug {
    pub types: Vec<TypeDef>,
    pub vars: HashMap<String, Vec<DiVar>>,
}

/// Read the `-g` debug variables/types from a bitcode file, or `None` when the module carries no
/// debug info (or could not be parsed — the translator's own `llvm-ir` parse reports that error).
pub fn read_debug(path: &str) -> Option<LlvmDebug> {
    unsafe { read_debug_unsafe(path) }
}

unsafe fn read_debug_unsafe(path: &str) -> Option<LlvmDebug> {
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
        // No debug metadata at all ⇒ nothing for this reader (the source-line half still comes from
        // the `llvm-ir` `DebugLoc` path in `lib.rs`).
        if LLVMGetModuleDebugMetadataVersion(module) == 0 {
            return None;
        }

        let mut out = LlvmDebug::default();
        let mut interner: HashMap<usize, TypeId> = HashMap::new();

        let mut f = LLVMGetFirstFunction(module);
        while !f.is_null() {
            if !LLVMGetSubprogram(f).is_null() {
                let fname = value_name(f);
                let vars = read_function_vars(ctx, f, &mut out.types, &mut interner);
                if !vars.is_empty() {
                    out.vars.insert(fname, vars);
                }
            }
            f = LLVMGetNextFunction(f);
        }
        if out.vars.is_empty() {
            None
        } else {
            Some(out)
        }
    })();
    LLVMContextDispose(ctx);
    result
}

/// Walk one function: number its allocas (textual order = the translator's order), then turn each
/// `llvm.dbg.declare` into a [`DiVar`] correlated by alloca ordinal.
unsafe fn read_function_vars(
    ctx: LLVMContextRef,
    f: LLVMValueRef,
    types: &mut Vec<TypeDef>,
    interner: &mut HashMap<usize, TypeId>,
) -> Vec<DiVar> {
    // Pass 1: alloca → ordinal (the Nth alloca in the function, in block/instruction order).
    let mut alloca_ord: HashMap<usize, usize> = HashMap::new();
    let mut n_alloca = 0usize;
    for_each_inst(f, |inst| {
        if LLVMIsAAllocaInst(inst) == inst {
            alloca_ord.insert(inst as usize, n_alloca);
            n_alloca += 1;
        }
    });

    // Pass 2: each `llvm.dbg.declare(addr, var, expr)` → a window variable at the alloca's slot.
    let mut vars = Vec::new();
    for_each_inst(f, |inst| {
        if LLVMIsACallInst(inst) != inst {
            return;
        }
        let callee = LLVMGetCalledValue(inst);
        if callee.is_null() || value_name(callee) != "llvm.dbg.declare" {
            return;
        }
        if LLVMGetNumOperands(inst) < 2 {
            return;
        }
        // op0 = address: a MetadataAsValue wrapping LocalAsMetadata(alloca); its single MDNode
        // operand is the alloca value, matched by pointer identity to the ordinal map.
        let arg0 = LLVMGetOperand(inst, 0);
        let Some(alloca) = single_md_value(arg0) else {
            return;
        };
        let Some(&ordinal) = alloca_ord.get(&(alloca as usize)) else {
            return; // address isn't a tracked alloca (e.g. a dbg.declare of a field) — skip
        };
        // op1 = the !DILocalVariable.
        let var_md = LLVMValueAsMetadata(LLVMGetOperand(inst, 1));
        if var_md.is_null()
            || !matches!(
                LLVMGetMetadataKind(var_md),
                LLVMMetadataKind::LLVMDILocalVariableMetadataKind
            )
        {
            return;
        }
        let name = op_string(ctx, var_md, 1).unwrap_or_default();
        if name.is_empty() {
            return;
        }
        let type_md = op_md(ctx, var_md, 3);
        let type_id = type_md.and_then(|t| intern_type(ctx, t, types, interner));
        let ty = type_id
            .map(|id| render_name(&types[id as usize]))
            .unwrap_or_else(|| "void".to_string());
        vars.push(DiVar {
            name,
            alloca_ordinal: ordinal,
            type_id,
            ty,
        });
    });
    vars
}

/// Recursively intern a `DIType` metadata node into the §6 `TypeDef` graph, returning its index.
/// Cycle-safe: a node reserves its `TypeId` with an `Opaque` placeholder before recursing, so a
/// `struct Point *` whose pointee is `struct Point` terminates (the wasm `dwarf_info` pattern).
unsafe fn intern_type(
    ctx: LLVMContextRef,
    md: LLVMMetadataRef,
    types: &mut Vec<TypeDef>,
    interner: &mut HashMap<usize, TypeId>,
) -> Option<TypeId> {
    if md.is_null() {
        return None;
    }
    let kind = LLVMGetMetadataKind(md);
    let is_type = matches!(
        kind,
        LLVMMetadataKind::LLVMDIBasicTypeMetadataKind
            | LLVMMetadataKind::LLVMDIDerivedTypeMetadataKind
            | LLVMMetadataKind::LLVMDICompositeTypeMetadataKind
    );
    if !is_type {
        return None;
    }
    if let Some(&id) = interner.get(&(md as usize)) {
        return Some(id);
    }

    let tag = LLVMGetDINodeTag(md);
    let name = di_name(md);
    let size_bytes = (LLVMDITypeGetSizeInBits(md) / 8) as u32;

    // Transparent aliases (typedef/const/volatile/restrict) resolve to their underlying type — no
    // new node, so `int` and `const int` share a `TypeId`.
    if matches!(
        tag,
        tag::TYPEDEF | tag::CONST_TYPE | tag::VOLATILE_TYPE | tag::RESTRICT_TYPE
    ) {
        return op_md(ctx, md, 3).and_then(|u| intern_type(ctx, u, types, interner));
    }

    // Reserve this node's id up front (placeholder) so cyclic graphs terminate.
    let id = types.len() as TypeId;
    interner.insert(md as usize, id);
    types.push(TypeDef::Opaque {
        name: name.clone(),
        size: size_bytes,
    });

    let def = match kind {
        LLVMMetadataKind::LLVMDIBasicTypeMetadataKind => TypeDef::Base {
            encoding: infer_encoding(&name),
            name,
            size: size_bytes,
        },
        LLVMMetadataKind::LLVMDIDerivedTypeMetadataKind if tag == tag::POINTER_TYPE => {
            let pointee = op_md(ctx, md, 3)
                .and_then(|p| intern_type(ctx, p, types, interner))
                .unwrap_or(id); // void* (no baseType) points at the placeholder
            let pname = format!("{} *", render_name(&types[pointee as usize]));
            TypeDef::Pointer {
                name: pname,
                pointee,
                size: if size_bytes == 0 { 8 } else { size_bytes },
            }
        }
        LLVMMetadataKind::LLVMDICompositeTypeMetadataKind if tag == tag::ARRAY_TYPE => {
            let elem = op_md(ctx, md, 3).and_then(|e| intern_type(ctx, e, types, interner));
            let elem = match elem {
                Some(e) => e,
                None => {
                    types[id as usize] = TypeDef::Opaque {
                        name,
                        size: size_bytes,
                    };
                    return Some(id);
                }
            };
            let elem_bits = types[elem as usize].size() as u64 * 8;
            let count = if elem_bits == 0 {
                0
            } else {
                (LLVMDITypeGetSizeInBits(md) / elem_bits) as u32
            };
            let ename = render_name(&types[elem as usize]);
            TypeDef::Array {
                name: format!("{ename}[{count}]"),
                elem,
                count,
            }
        }
        LLVMMetadataKind::LLVMDICompositeTypeMetadataKind => {
            // struct / union: gather the member DIDerivedTypes from the elements tuple (op 4).
            let kw = if tag == tag::UNION_TYPE {
                "union"
            } else {
                "struct"
            };
            let agg_name = if name.is_empty() {
                kw.to_string()
            } else {
                format!("{kw} {name}")
            };
            let mut fields = Vec::new();
            if let Some(elems) = op_md(ctx, md, 4) {
                for m in tuple_elems(ctx, elems) {
                    if LLVMGetDINodeTag(m) != tag::MEMBER {
                        continue;
                    }
                    let fname = di_name(m);
                    let foff = (LLVMDITypeGetOffsetInBits(m) / 8) as u32;
                    if let Some(fty) =
                        op_md(ctx, m, 3).and_then(|b| intern_type(ctx, b, types, interner))
                    {
                        fields.push(Field {
                            name: fname,
                            offset: foff,
                            ty: fty,
                        });
                    }
                }
            }
            TypeDef::Aggregate {
                name: agg_name,
                size: size_bytes,
                fields,
            }
        }
        // A derived type we don't model (e.g. member reached directly) → opaque.
        _ => TypeDef::Opaque {
            name,
            size: size_bytes,
        },
    };
    types[id as usize] = def;
    Some(id)
}

/// The §6 render name of an already-interned type.
fn render_name(t: &TypeDef) -> String {
    match t {
        TypeDef::Base { name, .. }
        | TypeDef::Pointer { name, .. }
        | TypeDef::Array { name, .. }
        | TypeDef::Aggregate { name, .. }
        | TypeDef::Opaque { name, .. } => name.clone(),
    }
}

/// Infer a base type's neutral encoding from its C name (the LLVM-C API exposes no encoding getter).
fn infer_encoding(name: &str) -> Encoding {
    if name.contains("float") || name.contains("double") {
        Encoding::Float
    } else if name == "_Bool" || name == "bool" {
        Encoding::Bool
    } else if name.starts_with("unsigned") || name == "_Bool" {
        Encoding::Unsigned
    } else {
        Encoding::Signed
    }
}

// ---- LLVM-C metadata helpers --------------------------------------------------------------------

/// Run `g` over every instruction of `f`, in block then instruction order.
unsafe fn for_each_inst(f: LLVMValueRef, mut g: impl FnMut(LLVMValueRef)) {
    let mut bb = LLVMGetFirstBasicBlock(f);
    while !bb.is_null() {
        let mut inst = LLVMGetFirstInstruction(bb);
        while !inst.is_null() {
            g(inst);
            inst = LLVMGetNextInstruction(inst);
        }
        bb = LLVMGetNextBasicBlock(bb);
    }
}

/// The `name` of an LLVM value (empty for unnamed/numbered values).
unsafe fn value_name(v: LLVMValueRef) -> String {
    let mut len = 0usize;
    let p = LLVMGetValueName2(v, &mut len);
    if p.is_null() || len == 0 {
        String::new()
    } else {
        String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len)).into_owned()
    }
}

/// `LLVMDITypeGetName` as an owned `String` (empty for anonymous types).
unsafe fn di_name(md: LLVMMetadataRef) -> String {
    let mut len = 0usize;
    let p = LLVMDITypeGetName(md, &mut len);
    if p.is_null() || len == 0 {
        String::new()
    } else {
        String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len)).into_owned()
    }
}

/// The single wrapped value of a `MetadataAsValue(LocalAsMetadata(v))` (a `dbg.declare` address).
unsafe fn single_md_value(mav: LLVMValueRef) -> Option<LLVMValueRef> {
    if mav.is_null() || LLVMGetMDNodeNumOperands(mav) != 1 {
        return None;
    }
    let mut d = [std::ptr::null_mut(); 1];
    LLVMGetMDNodeOperands(mav, d.as_mut_ptr());
    if d[0].is_null() {
        None
    } else {
        Some(d[0])
    }
}

/// Operand `i` of a metadata node as a metadata ref (`None` if absent/null/not metadata).
unsafe fn op_md(ctx: LLVMContextRef, md: LLVMMetadataRef, i: usize) -> Option<LLVMMetadataRef> {
    let val = LLVMMetadataAsValue(ctx, md);
    let n = LLVMGetMDNodeNumOperands(val) as usize;
    if i >= n {
        return None;
    }
    let mut ops = vec![std::ptr::null_mut(); n];
    LLVMGetMDNodeOperands(val, ops.as_mut_ptr());
    let op = ops[i];
    if op.is_null() {
        return None;
    }
    let cmd = LLVMValueAsMetadata(op);
    if cmd.is_null() {
        None
    } else {
        Some(cmd)
    }
}

/// Operand `i` of a metadata node as a UTF-8 string (when it is an `MDString`).
unsafe fn op_string(ctx: LLVMContextRef, md: LLVMMetadataRef, i: usize) -> Option<String> {
    let val = LLVMMetadataAsValue(ctx, md);
    let n = LLVMGetMDNodeNumOperands(val) as usize;
    if i >= n {
        return None;
    }
    let mut ops = vec![std::ptr::null_mut(); n];
    LLVMGetMDNodeOperands(val, ops.as_mut_ptr());
    let op = ops[i];
    if op.is_null() {
        return None;
    }
    let mut len = 0u32;
    let p = LLVMGetMDString(op, &mut len);
    if p.is_null() {
        None
    } else {
        Some(
            String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len as usize))
                .into_owned(),
        )
    }
}

/// The metadata operands of an `MDTuple` (the elements/members list).
unsafe fn tuple_elems(ctx: LLVMContextRef, md: LLVMMetadataRef) -> Vec<LLVMMetadataRef> {
    let val = LLVMMetadataAsValue(ctx, md);
    let n = LLVMGetMDNodeNumOperands(val) as usize;
    let mut ops = vec![std::ptr::null_mut(); n];
    LLVMGetMDNodeOperands(val, ops.as_mut_ptr());
    ops.into_iter()
        .filter(|o| !o.is_null())
        .map(|o| LLVMValueAsMetadata(o))
        .filter(|m| !m.is_null())
        .collect()
}
