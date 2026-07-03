//! Build the §6 structured debug info (the [`di::LlvmDebug`](crate::di::LlvmDebug) the translator
//! threads through its `di` argument) from the textual `.ll` metadata table — the in-house replacement
//! for [`di`](crate::di)'s `llvm-sys` DI walk. It covers the **type graph**, **module globals**, and
//! **local variables** (`llvm.dbg.declare` → `Window` at `-O0`; `dbg.value` → argument `Arg`) — the
//! lexical-block scoping of shadowed variables is the one remaining follow-up.
//!
//! It mirrors `di.rs` field-for-field so the two readers produce byte-identical `LlvmDebug` (the
//! `assert_ll_parity_debug` gate): the same interning order (globals walked in module order, the type
//! recursion reserving a placeholder id before descending), the same `infer_encoding(name)` heuristic
//! (LLVM-C exposes no encoding getter, so `di.rs` guesses from the name — we match that, *not* the
//! text's `encoding:` field), and the same array count = `size_bits / elem_bits`.

use std::collections::{BTreeMap, HashMap, HashSet};

use svm_ir::{Encoding, Field, TypeDef, TypeId};

use super::ast::{Function, Instruction, Module, Name};
use super::parse::{DbgIntrinsic, DiNode, MetaVal};
use crate::di::{DiGlobal, DiLoc, DiVar, LlvmDebug};

/// DWARF tag names the type reader dispatches on (the textual form of `di.rs`'s numeric `tag` module).
mod tag {
    pub const POINTER: &str = "DW_TAG_pointer_type";
    pub const ARRAY: &str = "DW_TAG_array_type";
    pub const UNION: &str = "DW_TAG_union_type";
    pub const MEMBER: &str = "DW_TAG_member";
    pub const TYPEDEF: &str = "DW_TAG_typedef";
    pub const CONST: &str = "DW_TAG_const_type";
    pub const VOLATILE: &str = "DW_TAG_volatile_type";
    pub const RESTRICT: &str = "DW_TAG_restrict_type";
}

type Meta = BTreeMap<u64, DiNode>;

/// Build the structured debug info from the parsed metadata table, the source globals (each a
/// `(symbol, DIGlobalVariableExpression id)` in module order), and the function-name table. Returns
/// `None` when nothing was recovered (mirrors `di::read_debug`'s empty case → `debug_info: None`).
pub(crate) fn build(
    meta: &Meta,
    global_dbg: &[(String, u64)],
    dbg_intrinsics: &[DbgIntrinsic],
    module: &Module,
    func_names: BTreeMap<String, String>,
) -> Option<LlvmDebug> {
    let mut types: Vec<TypeDef> = Vec::new();
    let mut interner: HashMap<u64, TypeId> = HashMap::new();
    // Functions first, then globals — the same walk order `di::read_debug` uses, so the type graph
    // interns in an identical order (and the `TypeId`s match).
    let mut vars: HashMap<String, Vec<DiVar>> = HashMap::new();
    for f in &module.functions {
        let fvars = read_function_vars(f, dbg_intrinsics, meta, &mut types, &mut interner);
        if !fvars.is_empty() {
            vars.insert(f.name.clone(), fvars);
        }
    }
    let globals = read_globals(meta, global_dbg, &mut types, &mut interner);
    if vars.is_empty() && globals.is_empty() && func_names.is_empty() {
        return None;
    }
    Some(LlvmDebug {
        types,
        vars,
        globals,
        func_names: func_names.into_iter().collect(),
    })
}

/// Recover one function's source locals from its captured `dbg.declare`/`dbg.value` intrinsics:
/// `dbg.declare` correlates the address to an **alloca ordinal** (the Nth alloca in block/instruction
/// order → a `Window` slot), `dbg.value` correlates the value to a **function argument** (→ `Arg`).
/// Mirrors `di::read_function_vars` (dedup by `!DILocalVariable` identity, first binding wins).
fn read_function_vars(
    f: &Function,
    dbg_intrinsics: &[DbgIntrinsic],
    meta: &Meta,
    types: &mut Vec<TypeDef>,
    interner: &mut HashMap<u64, TypeId>,
) -> Vec<DiVar> {
    // alloca → ordinal (Nth alloca, in block then instruction order).
    let mut alloca_ord: HashMap<&Name, usize> = HashMap::new();
    let mut n_alloca = 0usize;
    for bb in &f.basic_blocks {
        for inst in &bb.instrs {
            if let Instruction::Alloca(a) = inst {
                alloca_ord.insert(&a.dest, n_alloca);
                n_alloca += 1;
            }
        }
    }
    // argument → index (the `dbg.value` correlation key).
    let mut param_index: HashMap<&Name, u32> = HashMap::new();
    for (i, p) in f.parameters.iter().enumerate() {
        param_index.insert(&p.name, i as u32);
    }

    let mut seen: HashSet<u64> = HashSet::new();
    let mut vars = Vec::new();
    for d in dbg_intrinsics.iter().filter(|d| d.func == f.name) {
        let loc = if d.declare {
            match d.value.as_ref().and_then(|v| alloca_ord.get(v)) {
                Some(&ordinal) => DiLoc::Window {
                    alloca_ordinal: ordinal,
                },
                None => continue, // not a tracked alloca (e.g. a field) — skip
            }
        } else {
            match d.value.as_ref().and_then(|v| param_index.get(v)) {
                Some(&index) => DiLoc::Arg { index },
                None => continue, // not an argument (constant/poison/instruction result) — skip
            }
        };
        // The `!DILocalVariable` → name + structured type.
        let Some(var) = meta.get(&d.var) else {
            continue;
        };
        if var.kind() != Some("DILocalVariable") {
            continue;
        }
        let name = var.field("name").and_then(MetaVal::as_str).unwrap_or("");
        if name.is_empty() || !seen.insert(d.var) {
            continue; // unnamed, or this DI variable already recorded (first binding wins)
        }
        let type_id = var
            .field("type")
            .and_then(MetaVal::as_ref_id)
            .and_then(|t| intern_type(meta, t, types, interner));
        let ty = type_id
            .map(|id| render_name(&types[id as usize]))
            .unwrap_or_else(|| "void".to_string());
        vars.push(DiVar {
            name: name.to_string(),
            loc,
            type_id,
            ty,
            // §6 lexical-block scoping (shadowed-variable case) is a follow-up; a subprogram-scoped
            // variable — every non-nested local — is function-wide (`None`), which is the common case.
            scope: None,
        });
    }
    vars
}

/// Recover each source global from its `!dbg` `DIGlobalVariableExpression` → `DIGlobalVariable` (name
/// + structured type), interning the type graph in module-walk order (matching `di::read_globals`).
fn read_globals(
    meta: &Meta,
    global_dbg: &[(String, u64)],
    types: &mut Vec<TypeDef>,
    interner: &mut HashMap<u64, TypeId>,
) -> Vec<DiGlobal> {
    let mut out = Vec::new();
    for (symbol, gve_id) in global_dbg {
        let Some(var_id) = meta
            .get(gve_id)
            .and_then(|n| n.field("var"))
            .and_then(MetaVal::as_ref_id)
        else {
            continue;
        };
        let Some(var) = meta.get(&var_id) else {
            continue;
        };
        let name = var
            .field("name")
            .and_then(MetaVal::as_str)
            .unwrap_or(symbol.as_str())
            .to_string();
        let type_id = var
            .field("type")
            .and_then(MetaVal::as_ref_id)
            .and_then(|t| intern_type(meta, t, types, interner));
        let ty = type_id
            .map(|id| render_name(&types[id as usize]))
            .unwrap_or_else(|| "void".to_string());
        out.push(DiGlobal {
            symbol: symbol.clone(),
            name,
            type_id,
            ty,
        });
    }
    out
}

/// Recursively intern a `DIType` metadata node into the §6 `TypeDef` graph, returning its index.
/// Cycle-safe (reserves an `Opaque` placeholder before recursing); mirrors `di::intern_type`.
fn intern_type(
    meta: &Meta,
    id: u64,
    types: &mut Vec<TypeDef>,
    interner: &mut HashMap<u64, TypeId>,
) -> Option<TypeId> {
    let node = meta.get(&id)?;
    let kind = node.kind()?;
    if !matches!(kind, "DIBasicType" | "DIDerivedType" | "DICompositeType") {
        return None;
    }
    if let Some(&tid) = interner.get(&id) {
        return Some(tid);
    }
    let tag = node.field("tag").and_then(MetaVal::as_word);
    let name = node
        .field("name")
        .and_then(MetaVal::as_str)
        .unwrap_or("")
        .to_string();
    let size_bits = node.field("size").and_then(MetaVal::as_int).unwrap_or(0);
    let size_bytes = (size_bits / 8) as u32;

    // Transparent aliases (typedef/const/volatile/restrict) resolve to their underlying type — no new
    // node, so `int` and `const int` share a `TypeId`.
    let is_alias = matches!(tag, Some(t)
        if t == tag::TYPEDEF || t == tag::CONST || t == tag::VOLATILE || t == tag::RESTRICT);
    if is_alias {
        return node
            .field("baseType")
            .and_then(MetaVal::as_ref_id)
            .and_then(|u| intern_type(meta, u, types, interner));
    }

    // Reserve this node's id up front (placeholder) so cyclic graphs terminate.
    let tid = types.len() as TypeId;
    interner.insert(id, tid);
    types.push(TypeDef::Opaque {
        name: name.clone(),
        size: size_bytes,
    });

    let def = match kind {
        "DIBasicType" => TypeDef::Base {
            encoding: infer_encoding(&name),
            name,
            size: size_bytes,
        },
        "DIDerivedType" if tag == Some(tag::POINTER) => {
            let pointee = node
                .field("baseType")
                .and_then(MetaVal::as_ref_id)
                .and_then(|p| intern_type(meta, p, types, interner))
                .unwrap_or(tid); // void* (no baseType) points at the placeholder
            let pname = format!("{} *", render_name(&types[pointee as usize]));
            TypeDef::Pointer {
                name: pname,
                pointee,
                size: if size_bytes == 0 { 8 } else { size_bytes },
            }
        }
        "DICompositeType" if tag == Some(tag::ARRAY) => {
            let elem = node
                .field("baseType")
                .and_then(MetaVal::as_ref_id)
                .and_then(|e| intern_type(meta, e, types, interner));
            let elem = match elem {
                Some(e) => e,
                None => {
                    types[tid as usize] = TypeDef::Opaque {
                        name,
                        size: size_bytes,
                    };
                    return Some(tid);
                }
            };
            let elem_bits = types[elem as usize].size() as u64 * 8;
            let count = size_bits.checked_div(elem_bits).unwrap_or(0) as u32;
            let ename = render_name(&types[elem as usize]);
            TypeDef::Array {
                name: format!("{ename}[{count}]"),
                elem,
                count,
            }
        }
        "DICompositeType" => {
            // struct / union: gather the member `DIDerivedType`s from the `elements` tuple.
            let kw = if tag == Some(tag::UNION) {
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
            if let Some(elems) = node.field("elements").and_then(MetaVal::as_ref_id) {
                if let Some(DiNode::Tuple(members)) = meta.get(&elems) {
                    for &m in members {
                        let Some(mnode) = meta.get(&m) else { continue };
                        if mnode.field("tag").and_then(MetaVal::as_word) != Some(tag::MEMBER) {
                            continue;
                        }
                        let fname = mnode
                            .field("name")
                            .and_then(MetaVal::as_str)
                            .unwrap_or("")
                            .to_string();
                        let foff = (mnode.field("offset").and_then(MetaVal::as_int).unwrap_or(0)
                            / 8) as u32;
                        if let Some(fty) = mnode
                            .field("baseType")
                            .and_then(MetaVal::as_ref_id)
                            .and_then(|b| intern_type(meta, b, types, interner))
                        {
                            fields.push(Field {
                                name: fname,
                                offset: foff,
                                ty: fty,
                            });
                        }
                    }
                }
            }
            TypeDef::Aggregate {
                name: agg_name,
                size: size_bytes,
                fields,
            }
        }
        _ => TypeDef::Opaque {
            name,
            size: size_bytes,
        },
    };
    types[tid as usize] = def;
    Some(tid)
}

/// The §6 render name of an already-interned type (mirrors `di::render_name`).
fn render_name(t: &TypeDef) -> String {
    match t {
        TypeDef::Base { name, .. }
        | TypeDef::Pointer { name, .. }
        | TypeDef::Array { name, .. }
        | TypeDef::Aggregate { name, .. }
        | TypeDef::Opaque { name, .. } => name.clone(),
    }
}

/// Infer a base type's neutral encoding from its C name — `di.rs`'s heuristic, reproduced so the two
/// readers agree (the LLVM-C DI API exposes no encoding getter, so the bitcode reader also guesses).
fn infer_encoding(name: &str) -> Encoding {
    if name.contains("float") || name.contains("double") {
        Encoding::Float
    } else if name == "_Bool" || name == "bool" {
        Encoding::Bool
    } else if name.starts_with("unsigned") {
        Encoding::Unsigned
    } else {
        Encoding::Signed
    }
}
