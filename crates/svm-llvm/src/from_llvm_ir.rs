//! Conversion shim: an `llvm_ir::Module` (read from bitcode) → our owned [`crate::ll::Module`]
//! ([`ast`](crate::ll::ast)). This bridges the **bitcode reader** to the swapped translator, which now
//! consumes `crate::ll` types (LLVM.md §8 Q1b): the textual-`.ll` reader produces a `ll::Module`
//! directly, and this shim makes the legacy bitcode path produce the *same* shape, so the existing
//! differential corpus keeps running and becomes the parity oracle for the `.ll` reader.
//!
//! It is **transient** — once `llvm-ir`/`llvm-sys` are dropped (PR4) the bitcode path and this file go
//! with them, leaving the textual reader as the only ingest.
//!
//! Faithful + fail-closed: each construct maps field-for-field to its `ll` mirror (dropping the
//! `nuw`/`nsw`/attribute/calling-conv fields the translator never reads), and anything outside the
//! modeled subset (a non-folded constant expression, a funclet-EH construct, a target-ext type) is a
//! clean [`Error::Unsupported`] — exactly what the translator itself returns for the same input, so a
//! program that translates today is converted losslessly and one that doesn't fails the same way.

use either::Either;

use crate::ll::ast as ll;
use crate::Error;

type R<T> = Result<T, Error>;

fn unsup<T>(what: impl Into<String>) -> R<T> {
    Err(Error::Unsupported(what.into()))
}

/// Convert a whole bitcode module to its `ll` mirror.
pub fn convert_module(m: &llvm_ir::Module) -> R<ll::Module> {
    let mut types = ll::Types::new();
    // Named-struct definitions, so `ll::Types::named_struct_def` resolves during translation.
    for name in m.types.all_struct_names() {
        let def = match m.types.named_struct_def(name) {
            Some(llvm_ir::types::NamedStructDef::Defined(t)) => {
                ll::NamedStructDef::Defined(cvt_type(t)?)
            }
            // Opaque, or (defensively) a name without a def — both map to opaque.
            _ => ll::NamedStructDef::Opaque,
        };
        types.add_named_struct_def(name.clone(), def);
    }

    let functions = m.functions.iter().map(cvt_function).collect::<R<_>>()?;
    let func_declarations = m
        .func_declarations
        .iter()
        .map(cvt_func_decl)
        .collect::<R<_>>()?;
    let global_vars = m.global_vars.iter().map(cvt_global_var).collect::<R<_>>()?;
    let global_aliases = m
        .global_aliases
        .iter()
        .map(cvt_global_alias)
        .collect::<R<_>>()?;

    Ok(ll::Module {
        name: m.name.clone(),
        source_file_name: m.source_file_name.clone(),
        target_triple: m.target_triple.clone(),
        functions,
        func_declarations,
        global_vars,
        global_aliases,
        types,
    })
}

// ---- names / types / debug locs ----------------------------------------------------------------

fn cvt_name(n: &llvm_ir::Name) -> ll::Name {
    match n {
        llvm_ir::Name::Name(s) => ll::Name::Name(s.clone()),
        llvm_ir::Name::Number(k) => ll::Name::Number(*k),
    }
}

fn cvt_fp(fp: llvm_ir::types::FPType) -> ll::FPType {
    use llvm_ir::types::FPType as F;
    match fp {
        F::Half => ll::FPType::Half,
        F::BFloat => ll::FPType::BFloat,
        F::Single => ll::FPType::Single,
        F::Double => ll::FPType::Double,
        F::FP128 => ll::FPType::FP128,
        F::X86_FP80 => ll::FPType::X86_FP80,
        F::PPC_FP128 => ll::FPType::PPC_FP128,
    }
}

fn cvt_type(t: &llvm_ir::TypeRef) -> R<ll::TypeRef> {
    use llvm_ir::types::Type as T;
    let ty = match t.as_ref() {
        T::VoidType => ll::Type::VoidType,
        T::IntegerType { bits } => ll::Type::IntegerType { bits: *bits },
        // LLVM 15+: opaque pointer (the pin), so only an address space.
        T::PointerType { addr_space } => ll::Type::PointerType {
            addr_space: *addr_space,
        },
        T::FPType(fp) => ll::Type::FPType(cvt_fp(*fp)),
        T::FuncType {
            result_type,
            param_types,
            is_var_arg,
        } => ll::Type::FuncType {
            result_type: cvt_type(result_type)?,
            param_types: param_types.iter().map(cvt_type).collect::<R<_>>()?,
            is_var_arg: *is_var_arg,
        },
        T::VectorType {
            element_type,
            num_elements,
            scalable,
        } => ll::Type::VectorType {
            element_type: cvt_type(element_type)?,
            num_elements: *num_elements,
            scalable: *scalable,
        },
        T::ArrayType {
            element_type,
            num_elements,
        } => ll::Type::ArrayType {
            element_type: cvt_type(element_type)?,
            num_elements: *num_elements,
        },
        T::StructType {
            element_types,
            is_packed,
        } => ll::Type::StructType {
            element_types: element_types.iter().map(cvt_type).collect::<R<_>>()?,
            is_packed: *is_packed,
        },
        T::NamedStructType { name } => ll::Type::NamedStructType { name: name.clone() },
        T::X86_MMXType => ll::Type::X86_MMXType,
        T::X86_AMXType => ll::Type::X86_AMXType,
        T::MetadataType => ll::Type::MetadataType,
        T::LabelType => ll::Type::LabelType,
        T::TokenType => ll::Type::TokenType,
        T::TargetExtType => return unsup("target extension type"),
    };
    Ok(ll::TypeRef::new(ty))
}

fn cvt_debugloc(d: &Option<llvm_ir::debugloc::DebugLoc>) -> Option<ll::DebugLoc> {
    d.as_ref().map(|d| ll::DebugLoc {
        line: d.line,
        col: d.col,
        filename: d.filename.clone(),
        directory: d.directory.clone(),
    })
}

// ---- constants ---------------------------------------------------------------------------------

fn cvt_float(f: &llvm_ir::constant::Float) -> ll::Float {
    use llvm_ir::constant::Float as F;
    match f {
        F::Half => ll::Float::Half,
        F::BFloat => ll::Float::BFloat,
        F::Single(x) => ll::Float::Single(*x),
        F::Double(x) => ll::Float::Double(*x),
        F::Quadruple => ll::Float::Quadruple,
        F::X86_FP80 => ll::Float::X86_FP80,
        F::PPC_FP128 => ll::Float::PPC_FP128,
    }
}

fn cvt_cunop(u: &impl ConstUnopFields) -> R<ll::ConstUnaryOp> {
    Ok(ll::ConstUnaryOp {
        operand: cvt_constant(u.operand())?,
        to_type: cvt_type(u.to_type())?,
    })
}

fn cvt_cbinop(b: &impl ConstBinopFields) -> R<ll::ConstBinaryOp> {
    Ok(ll::ConstBinaryOp {
        operand0: cvt_constant(b.operand0())?,
        operand1: cvt_constant(b.operand1())?,
    })
}

fn cvt_constant(c: &llvm_ir::ConstantRef) -> R<ll::ConstantRef> {
    use llvm_ir::constant::Constant as C;
    let out = match c.as_ref() {
        // I14: `llvm-ir` already truncated a `bits > 64` value to its low word, but `translate_bc_path`
        // rejects such modules up front (`wideint`), so anything reaching here fits in `u64` and widens
        // exactly. (The textual reader carries the full `u128` natively — the point of the migration.)
        C::Int { bits, value } => ll::Constant::Int {
            bits: *bits,
            value: *value as u128,
        },
        C::Float(f) => ll::Constant::Float(cvt_float(f)),
        C::Null(t) => ll::Constant::Null(cvt_type(t)?),
        C::AggregateZero(t) => ll::Constant::AggregateZero(cvt_type(t)?),
        C::Struct {
            name,
            values,
            is_packed,
        } => ll::Constant::Struct {
            name: name.clone(),
            values: values.iter().map(cvt_constant).collect::<R<_>>()?,
            is_packed: *is_packed,
        },
        C::Array {
            element_type,
            elements,
        } => ll::Constant::Array {
            element_type: cvt_type(element_type)?,
            elements: elements.iter().map(cvt_constant).collect::<R<_>>()?,
        },
        C::Vector(v) => ll::Constant::Vector(v.iter().map(cvt_constant).collect::<R<_>>()?),
        C::Undef(t) => ll::Constant::Undef(cvt_type(t)?),
        C::Poison(t) => ll::Constant::Poison(cvt_type(t)?),
        C::BlockAddress => ll::Constant::BlockAddress,
        C::GlobalReference { name, ty } => ll::Constant::GlobalReference {
            name: cvt_name(name),
            ty: cvt_type(ty)?,
        },
        C::TokenNone => ll::Constant::TokenNone,
        C::GetElementPtr(g) => ll::Constant::GetElementPtr(ll::ConstGetElementPtr {
            address: cvt_constant(&g.address)?,
            indices: g.indices.iter().map(cvt_constant).collect::<R<_>>()?,
            in_bounds: g.in_bounds,
        }),
        C::Trunc(u) => ll::Constant::Trunc(cvt_cunop(u)?),
        C::PtrToInt(u) => ll::Constant::PtrToInt(cvt_cunop(u)?),
        C::IntToPtr(u) => ll::Constant::IntToPtr(cvt_cunop(u)?),
        C::BitCast(u) => ll::Constant::BitCast(cvt_cunop(u)?),
        C::AddrSpaceCast(u) => ll::Constant::AddrSpaceCast(cvt_cunop(u)?),
        C::Add(b) => ll::Constant::Add(cvt_cbinop(b)?),
        C::Sub(b) => ll::Constant::Sub(cvt_cbinop(b)?),
        C::Mul(b) => ll::Constant::Mul(cvt_cbinop(b)?),
        // Any other constant expression (`Xor`/`Shl`/`ICmp`/`FCmp`/`ExtractElement`/… as a constant) is
        // outside the folded subset; the translator rejects the same forms via its `const_eval`/
        // `const_bytes` catch-alls, so fail closed here too.
        other => return unsup(format!("constant expression {other:?}")),
    };
    Ok(ll::ConstantRef::new(out))
}

// ---- operands ----------------------------------------------------------------------------------

fn cvt_operand(o: &llvm_ir::Operand) -> R<ll::Operand> {
    Ok(match o {
        llvm_ir::Operand::LocalOperand { name, ty } => ll::Operand::LocalOperand {
            name: cvt_name(name),
            ty: cvt_type(ty)?,
        },
        llvm_ir::Operand::ConstantOperand(c) => ll::Operand::ConstantOperand(cvt_constant(c)?),
        llvm_ir::Operand::MetadataOperand => ll::Operand::MetadataOperand,
    })
}

/// The callee of a `call`/`invoke`: inline asm (`Left`) or an operand (`Right`).
fn cvt_callee(
    f: &Either<llvm_ir::instruction::InlineAssembly, llvm_ir::Operand>,
) -> R<ll::Either<ll::InlineAssembly, ll::Operand>> {
    Ok(match f {
        Either::Left(asm) => ll::Either::Left(ll::InlineAssembly {
            ty: cvt_type(&asm.ty)?,
        }),
        Either::Right(op) => ll::Either::Right(cvt_operand(op)?),
    })
}

/// Call/invoke arguments — the attribute list is dropped (the translator reads only the operand).
fn cvt_args(
    args: &[(llvm_ir::Operand, Vec<llvm_ir::function::ParameterAttribute>)],
) -> R<Vec<(ll::Operand, Vec<ll::ParameterAttribute>)>> {
    args.iter()
        .map(|(op, _attrs)| Ok((cvt_operand(op)?, Vec::new())))
        .collect()
}

// ---- instructions ------------------------------------------------------------------------------

fn cvt_binop(b: &impl BinopFields) -> R<ll::BinaryOp> {
    Ok(ll::BinaryOp {
        operand0: cvt_operand(b.operand0())?,
        operand1: cvt_operand(b.operand1())?,
        dest: cvt_name(b.dest()),
        debugloc: cvt_debugloc(b.debugloc()),
    })
}

fn cvt_unop_same(
    operand: &llvm_ir::Operand,
    dest: &llvm_ir::Name,
    dl: &Option<llvm_ir::debugloc::DebugLoc>,
) -> R<ll::UnaryOp> {
    // `fneg`/`freeze`: the result type equals the operand type — carry it as `to_type`.
    let operand = cvt_operand(operand)?;
    let to_type = ll_operand_type(&operand);
    Ok(ll::UnaryOp {
        operand,
        to_type,
        dest: cvt_name(dest),
        debugloc: cvt_debugloc(dl),
    })
}

fn cvt_unop_typed(
    operand: &llvm_ir::Operand,
    to_type: &llvm_ir::TypeRef,
    dest: &llvm_ir::Name,
    dl: &Option<llvm_ir::debugloc::DebugLoc>,
) -> R<ll::UnaryOp> {
    Ok(ll::UnaryOp {
        operand: cvt_operand(operand)?,
        to_type: cvt_type(to_type)?,
        dest: cvt_name(dest),
        debugloc: cvt_debugloc(dl),
    })
}

/// The static type of an already-converted operand (cheap — `LocalOperand` carries it; a constant
/// carries it structurally). Used to fill `fneg`/`freeze`'s `to_type` without a `Types` round-trip.
fn ll_operand_type(o: &ll::Operand) -> ll::TypeRef {
    let types = ll::Types::new();
    ll::Typed::get_type(o, &types)
}

fn cvt_instruction(i: &llvm_ir::instruction::Instruction) -> R<ll::Instruction> {
    use llvm_ir::instruction::Instruction as I;
    Ok(match i {
        I::Add(b) => ll::Instruction::Add(cvt_binop(b)?),
        I::Sub(b) => ll::Instruction::Sub(cvt_binop(b)?),
        I::Mul(b) => ll::Instruction::Mul(cvt_binop(b)?),
        I::UDiv(b) => ll::Instruction::UDiv(cvt_binop(b)?),
        I::SDiv(b) => ll::Instruction::SDiv(cvt_binop(b)?),
        I::URem(b) => ll::Instruction::URem(cvt_binop(b)?),
        I::SRem(b) => ll::Instruction::SRem(cvt_binop(b)?),
        I::And(b) => ll::Instruction::And(cvt_binop(b)?),
        I::Or(b) => ll::Instruction::Or(cvt_binop(b)?),
        I::Xor(b) => ll::Instruction::Xor(cvt_binop(b)?),
        I::Shl(b) => ll::Instruction::Shl(cvt_binop(b)?),
        I::LShr(b) => ll::Instruction::LShr(cvt_binop(b)?),
        I::AShr(b) => ll::Instruction::AShr(cvt_binop(b)?),
        I::FAdd(b) => ll::Instruction::FAdd(cvt_binop(b)?),
        I::FSub(b) => ll::Instruction::FSub(cvt_binop(b)?),
        I::FMul(b) => ll::Instruction::FMul(cvt_binop(b)?),
        I::FDiv(b) => ll::Instruction::FDiv(cvt_binop(b)?),
        I::FRem(b) => ll::Instruction::FRem(cvt_binop(b)?),
        I::FNeg(i) => ll::Instruction::FNeg(cvt_unop_same(&i.operand, &i.dest, &i.debugloc)?),
        I::ExtractElement(i) => ll::Instruction::ExtractElement(ll::ExtractElement {
            vector: cvt_operand(&i.vector)?,
            index: cvt_operand(&i.index)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::InsertElement(i) => ll::Instruction::InsertElement(ll::InsertElement {
            vector: cvt_operand(&i.vector)?,
            element: cvt_operand(&i.element)?,
            index: cvt_operand(&i.index)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::ShuffleVector(i) => ll::Instruction::ShuffleVector(ll::ShuffleVector {
            operand0: cvt_operand(&i.operand0)?,
            operand1: cvt_operand(&i.operand1)?,
            dest: cvt_name(&i.dest),
            mask: cvt_constant(&i.mask)?,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::ExtractValue(i) => ll::Instruction::ExtractValue(ll::ExtractValue {
            aggregate: cvt_operand(&i.aggregate)?,
            indices: i.indices.clone(),
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::InsertValue(i) => ll::Instruction::InsertValue(ll::InsertValue {
            aggregate: cvt_operand(&i.aggregate)?,
            element: cvt_operand(&i.element)?,
            indices: i.indices.clone(),
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Alloca(i) => ll::Instruction::Alloca(ll::Alloca {
            allocated_type: cvt_type(&i.allocated_type)?,
            num_elements: cvt_operand(&i.num_elements)?,
            dest: cvt_name(&i.dest),
            alignment: i.alignment,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Load(i) => ll::Instruction::Load(ll::Load {
            address: cvt_operand(&i.address)?,
            dest: cvt_name(&i.dest),
            loaded_ty: cvt_type(&i.loaded_ty)?,
            volatile: i.volatile,
            atomicity: cvt_atomicity_opt(&i.atomicity),
            alignment: i.alignment,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Store(i) => ll::Instruction::Store(ll::Store {
            address: cvt_operand(&i.address)?,
            value: cvt_operand(&i.value)?,
            volatile: i.volatile,
            atomicity: cvt_atomicity_opt(&i.atomicity),
            alignment: i.alignment,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Fence(i) => ll::Instruction::Fence(ll::Fence {
            atomicity: cvt_atomicity(&i.atomicity),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::CmpXchg(i) => ll::Instruction::CmpXchg(ll::CmpXchg {
            address: cvt_operand(&i.address)?,
            expected: cvt_operand(&i.expected)?,
            replacement: cvt_operand(&i.replacement)?,
            dest: cvt_name(&i.dest),
            volatile: i.volatile,
            atomicity: cvt_atomicity(&i.atomicity),
            failure_memory_ordering: cvt_mem_ordering(i.failure_memory_ordering),
            weak: i.weak,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::AtomicRMW(i) => ll::Instruction::AtomicRMW(ll::AtomicRMW {
            operation: cvt_rmw_op(i.operation)?,
            address: cvt_operand(&i.address)?,
            value: cvt_operand(&i.value)?,
            dest: cvt_name(&i.dest),
            volatile: i.volatile,
            atomicity: cvt_atomicity(&i.atomicity),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::GetElementPtr(i) => ll::Instruction::GetElementPtr(ll::GetElementPtr {
            address: cvt_operand(&i.address)?,
            indices: i.indices.iter().map(cvt_operand).collect::<R<_>>()?,
            dest: cvt_name(&i.dest),
            in_bounds: i.in_bounds,
            source_element_type: cvt_type(&i.source_element_type)?,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Trunc(i) => ll::Instruction::Trunc(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::ZExt(i) => ll::Instruction::ZExt(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::SExt(i) => ll::Instruction::SExt(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::FPTrunc(i) => ll::Instruction::FPTrunc(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::FPExt(i) => ll::Instruction::FPExt(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::FPToUI(i) => ll::Instruction::FPToUI(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::FPToSI(i) => ll::Instruction::FPToSI(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::UIToFP(i) => ll::Instruction::UIToFP(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::SIToFP(i) => ll::Instruction::SIToFP(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::PtrToInt(i) => ll::Instruction::PtrToInt(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::IntToPtr(i) => ll::Instruction::IntToPtr(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::BitCast(i) => ll::Instruction::BitCast(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::AddrSpaceCast(i) => ll::Instruction::AddrSpaceCast(cvt_unop_typed(
            &i.operand,
            &i.to_type,
            &i.dest,
            &i.debugloc,
        )?),
        I::ICmp(i) => ll::Instruction::ICmp(ll::ICmp {
            predicate: cvt_int_pred(i.predicate),
            operand0: cvt_operand(&i.operand0)?,
            operand1: cvt_operand(&i.operand1)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::FCmp(i) => ll::Instruction::FCmp(ll::FCmp {
            predicate: cvt_fp_pred(i.predicate),
            operand0: cvt_operand(&i.operand0)?,
            operand1: cvt_operand(&i.operand1)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Phi(i) => ll::Instruction::Phi(ll::Phi {
            incoming_values: i
                .incoming_values
                .iter()
                .map(|(op, n)| Ok((cvt_operand(op)?, cvt_name(n))))
                .collect::<R<_>>()?,
            dest: cvt_name(&i.dest),
            to_type: cvt_type(&i.to_type)?,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Select(i) => ll::Instruction::Select(ll::Select {
            condition: cvt_operand(&i.condition)?,
            true_value: cvt_operand(&i.true_value)?,
            false_value: cvt_operand(&i.false_value)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::Freeze(i) => ll::Instruction::Freeze(cvt_unop_same(&i.operand, &i.dest, &i.debugloc)?),
        I::Call(i) => ll::Instruction::Call(ll::Call {
            function: cvt_callee(&i.function)?,
            function_ty: cvt_type(&i.function_ty)?,
            arguments: cvt_args(&i.arguments)?,
            dest: i.dest.as_ref().map(cvt_name),
            is_tail_call: i.is_tail_call,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::VAArg(i) => ll::Instruction::VAArg(ll::VAArg {
            arg_list: cvt_operand(&i.arg_list)?,
            cur_type: cvt_type(&i.cur_type)?,
            dest: cvt_name(&i.dest),
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        I::LandingPad(i) => ll::Instruction::LandingPad(ll::LandingPad {
            result_type: cvt_type(&i.result_type)?,
            clauses: i.clauses.iter().map(|_| ll::LandingPadClause {}).collect(),
            dest: cvt_name(&i.dest),
            cleanup: i.cleanup,
            debugloc: cvt_debugloc(&i.debugloc),
        }),
        // Windows funclet EH — not in the (Itanium) corpus; the translator rejects it too.
        I::CatchPad(_) => return unsup("catchpad (funclet EH)"),
        I::CleanupPad(_) => return unsup("cleanuppad (funclet EH)"),
    })
}

// ---- terminators -------------------------------------------------------------------------------

fn cvt_terminator(t: &llvm_ir::terminator::Terminator) -> R<ll::Terminator> {
    use llvm_ir::terminator::Terminator as T;
    Ok(match t {
        T::Ret(r) => ll::Terminator::Ret(ll::Ret {
            return_operand: r.return_operand.as_ref().map(cvt_operand).transpose()?,
            debugloc: cvt_debugloc(&r.debugloc),
        }),
        T::Br(b) => ll::Terminator::Br(ll::Br {
            dest: cvt_name(&b.dest),
            debugloc: cvt_debugloc(&b.debugloc),
        }),
        T::CondBr(b) => ll::Terminator::CondBr(ll::CondBr {
            condition: cvt_operand(&b.condition)?,
            true_dest: cvt_name(&b.true_dest),
            false_dest: cvt_name(&b.false_dest),
            debugloc: cvt_debugloc(&b.debugloc),
        }),
        T::Switch(s) => ll::Terminator::Switch(ll::Switch {
            operand: cvt_operand(&s.operand)?,
            dests: s
                .dests
                .iter()
                .map(|(c, n)| Ok((cvt_constant(c)?, cvt_name(n))))
                .collect::<R<_>>()?,
            default_dest: cvt_name(&s.default_dest),
            debugloc: cvt_debugloc(&s.debugloc),
        }),
        T::IndirectBr(ib) => ll::Terminator::IndirectBr(ll::IndirectBr {
            operand: cvt_operand(&ib.operand)?,
            possible_dests: ib.possible_dests.iter().map(cvt_name).collect(),
            debugloc: cvt_debugloc(&ib.debugloc),
        }),
        T::Invoke(inv) => ll::Terminator::Invoke(ll::Invoke {
            function: cvt_callee(&inv.function)?,
            function_ty: cvt_type(&inv.function_ty)?,
            arguments: cvt_args(&inv.arguments)?,
            result: cvt_name(&inv.result),
            return_label: cvt_name(&inv.return_label),
            exception_label: cvt_name(&inv.exception_label),
            debugloc: cvt_debugloc(&inv.debugloc),
        }),
        T::Resume(r) => ll::Terminator::Resume(ll::Resume {
            operand: cvt_operand(&r.operand)?,
            debugloc: cvt_debugloc(&r.debugloc),
        }),
        T::Unreachable(u) => ll::Terminator::Unreachable(ll::Unreachable {
            debugloc: cvt_debugloc(&u.debugloc),
        }),
        // Funclet EH / callbr — not in the corpus; the translator rejects them via its `other` arm.
        T::CleanupRet(_) => return unsup("cleanupret (funclet EH)"),
        T::CatchRet(_) => return unsup("catchret (funclet EH)"),
        T::CatchSwitch(_) => return unsup("catchswitch (funclet EH)"),
        T::CallBr(_) => return unsup("callbr (inline-asm goto)"),
    })
}

// ---- atomics / predicates / rmw ----------------------------------------------------------------

fn cvt_mem_ordering(o: llvm_ir::instruction::MemoryOrdering) -> ll::MemoryOrdering {
    use llvm_ir::instruction::MemoryOrdering as M;
    match o {
        M::Unordered => ll::MemoryOrdering::Unordered,
        M::Monotonic => ll::MemoryOrdering::Monotonic,
        M::Acquire => ll::MemoryOrdering::Acquire,
        M::Release => ll::MemoryOrdering::Release,
        M::AcquireRelease => ll::MemoryOrdering::AcquireRelease,
        M::SequentiallyConsistent => ll::MemoryOrdering::SequentiallyConsistent,
        M::NotAtomic => ll::MemoryOrdering::NotAtomic,
    }
}

fn cvt_sync_scope(s: llvm_ir::instruction::SynchronizationScope) -> ll::SynchronizationScope {
    use llvm_ir::instruction::SynchronizationScope as S;
    match s {
        S::SingleThread => ll::SynchronizationScope::SingleThread,
        S::System => ll::SynchronizationScope::System,
    }
}

fn cvt_atomicity(a: &llvm_ir::instruction::Atomicity) -> ll::Atomicity {
    ll::Atomicity {
        synch_scope: cvt_sync_scope(a.synch_scope),
        mem_ordering: cvt_mem_ordering(a.mem_ordering),
    }
}

fn cvt_atomicity_opt(a: &Option<llvm_ir::instruction::Atomicity>) -> Option<ll::Atomicity> {
    a.as_ref().map(cvt_atomicity)
}

fn cvt_rmw_op(op: llvm_ir::instruction::RMWBinOp) -> R<ll::RMWBinOp> {
    use llvm_ir::instruction::RMWBinOp as L;
    Ok(match op {
        L::Xchg => ll::RMWBinOp::Xchg,
        L::Add => ll::RMWBinOp::Add,
        L::Sub => ll::RMWBinOp::Sub,
        L::And => ll::RMWBinOp::And,
        L::Nand => ll::RMWBinOp::Nand,
        L::Or => ll::RMWBinOp::Or,
        L::Xor => ll::RMWBinOp::Xor,
        L::Max => ll::RMWBinOp::Max,
        L::Min => ll::RMWBinOp::Min,
        L::UMax => ll::RMWBinOp::UMax,
        L::UMin => ll::RMWBinOp::UMin,
        L::FAdd => ll::RMWBinOp::FAdd,
        L::FSub => ll::RMWBinOp::FSub,
        L::FMax => ll::RMWBinOp::FMax,
        L::FMin => ll::RMWBinOp::FMin,
    })
}

fn cvt_int_pred(p: llvm_ir::IntPredicate) -> ll::IntPredicate {
    use llvm_ir::IntPredicate as P;
    match p {
        P::EQ => ll::IntPredicate::EQ,
        P::NE => ll::IntPredicate::NE,
        P::UGT => ll::IntPredicate::UGT,
        P::UGE => ll::IntPredicate::UGE,
        P::ULT => ll::IntPredicate::ULT,
        P::ULE => ll::IntPredicate::ULE,
        P::SGT => ll::IntPredicate::SGT,
        P::SGE => ll::IntPredicate::SGE,
        P::SLT => ll::IntPredicate::SLT,
        P::SLE => ll::IntPredicate::SLE,
    }
}

fn cvt_fp_pred(p: llvm_ir::FPPredicate) -> ll::FPPredicate {
    use llvm_ir::FPPredicate as P;
    match p {
        P::False => ll::FPPredicate::False,
        P::OEQ => ll::FPPredicate::OEQ,
        P::OGT => ll::FPPredicate::OGT,
        P::OGE => ll::FPPredicate::OGE,
        P::OLT => ll::FPPredicate::OLT,
        P::OLE => ll::FPPredicate::OLE,
        P::ONE => ll::FPPredicate::ONE,
        P::ORD => ll::FPPredicate::ORD,
        P::UNO => ll::FPPredicate::UNO,
        P::UEQ => ll::FPPredicate::UEQ,
        P::UGT => ll::FPPredicate::UGT,
        P::UGE => ll::FPPredicate::UGE,
        P::ULT => ll::FPPredicate::ULT,
        P::ULE => ll::FPPredicate::ULE,
        P::UNE => ll::FPPredicate::UNE,
        P::True => ll::FPPredicate::True,
    }
}

// ---- module structure --------------------------------------------------------------------------

fn cvt_param(p: &llvm_ir::function::Parameter) -> R<ll::Parameter> {
    Ok(ll::Parameter {
        name: cvt_name(&p.name),
        ty: cvt_type(&p.ty)?,
    })
}

fn cvt_block(bb: &llvm_ir::BasicBlock) -> R<ll::BasicBlock> {
    Ok(ll::BasicBlock {
        name: cvt_name(&bb.name),
        instrs: bb.instrs.iter().map(cvt_instruction).collect::<R<_>>()?,
        term: cvt_terminator(&bb.term)?,
    })
}

fn cvt_function(f: &llvm_ir::Function) -> R<ll::Function> {
    Ok(ll::Function {
        name: f.name.clone(),
        parameters: f.parameters.iter().map(cvt_param).collect::<R<_>>()?,
        is_var_arg: f.is_var_arg,
        return_type: cvt_type(&f.return_type)?,
        basic_blocks: f.basic_blocks.iter().map(cvt_block).collect::<R<_>>()?,
    })
}

fn cvt_func_decl(d: &llvm_ir::function::FunctionDeclaration) -> R<ll::FunctionDeclaration> {
    Ok(ll::FunctionDeclaration {
        name: d.name.clone(),
        parameters: d.parameters.iter().map(cvt_param).collect::<R<_>>()?,
        is_var_arg: d.is_var_arg,
        return_type: cvt_type(&d.return_type)?,
    })
}

fn cvt_global_var(g: &llvm_ir::module::GlobalVariable) -> R<ll::GlobalVariable> {
    Ok(ll::GlobalVariable {
        name: cvt_name(&g.name),
        ty: cvt_type(&g.ty)?,
        initializer: g.initializer.as_ref().map(cvt_constant).transpose()?,
        is_constant: g.is_constant,
        alignment: g.alignment,
    })
}

fn cvt_global_alias(g: &llvm_ir::module::GlobalAlias) -> R<ll::GlobalAlias> {
    Ok(ll::GlobalAlias {
        name: cvt_name(&g.name),
        aliasee: cvt_constant(&g.aliasee)?,
        ty: cvt_type(&g.ty)?,
    })
}

// ---- binop field access (uniform over `llvm-ir`'s per-op binop structs) -------------------------

/// `llvm-ir` gives each integer/float binary op its own struct (`Add`, `Sub`, …) rather than a shared
/// one, so this trait lets [`cvt_binop`] read the common fields uniformly.
trait BinopFields {
    fn operand0(&self) -> &llvm_ir::Operand;
    fn operand1(&self) -> &llvm_ir::Operand;
    fn dest(&self) -> &llvm_ir::Name;
    fn debugloc(&self) -> &Option<llvm_ir::debugloc::DebugLoc>;
}

macro_rules! impl_binop_fields {
    ($($ty:ty),+ $(,)?) => {
        $(impl BinopFields for $ty {
            fn operand0(&self) -> &llvm_ir::Operand { &self.operand0 }
            fn operand1(&self) -> &llvm_ir::Operand { &self.operand1 }
            fn dest(&self) -> &llvm_ir::Name { &self.dest }
            fn debugloc(&self) -> &Option<llvm_ir::debugloc::DebugLoc> { &self.debugloc }
        })+
    };
}
impl_binop_fields!(
    llvm_ir::instruction::Add,
    llvm_ir::instruction::Sub,
    llvm_ir::instruction::Mul,
    llvm_ir::instruction::UDiv,
    llvm_ir::instruction::SDiv,
    llvm_ir::instruction::URem,
    llvm_ir::instruction::SRem,
    llvm_ir::instruction::And,
    llvm_ir::instruction::Or,
    llvm_ir::instruction::Xor,
    llvm_ir::instruction::Shl,
    llvm_ir::instruction::LShr,
    llvm_ir::instruction::AShr,
    llvm_ir::instruction::FAdd,
    llvm_ir::instruction::FSub,
    llvm_ir::instruction::FMul,
    llvm_ir::instruction::FDiv,
    llvm_ir::instruction::FRem,
);

// ---- constant-expression field access (`llvm-ir` gives each const op its own struct too) ---------

/// Common fields of `llvm-ir`'s constant **unary** ops (`Trunc`/`PtrToInt`/`IntToPtr`/`BitCast`/
/// `AddrSpaceCast`): an operand and the explicit result type.
trait ConstUnopFields {
    fn operand(&self) -> &llvm_ir::ConstantRef;
    fn to_type(&self) -> &llvm_ir::TypeRef;
}

macro_rules! impl_cunop_fields {
    ($($ty:ty),+ $(,)?) => {
        $(impl ConstUnopFields for $ty {
            fn operand(&self) -> &llvm_ir::ConstantRef { &self.operand }
            fn to_type(&self) -> &llvm_ir::TypeRef { &self.to_type }
        })+
    };
}
impl_cunop_fields!(
    llvm_ir::constant::Trunc,
    llvm_ir::constant::PtrToInt,
    llvm_ir::constant::IntToPtr,
    llvm_ir::constant::BitCast,
    llvm_ir::constant::AddrSpaceCast,
);

/// Common fields of `llvm-ir`'s constant **binary** ops (`Add`/`Sub`/`Mul`).
trait ConstBinopFields {
    fn operand0(&self) -> &llvm_ir::ConstantRef;
    fn operand1(&self) -> &llvm_ir::ConstantRef;
}

macro_rules! impl_cbinop_fields {
    ($($ty:ty),+ $(,)?) => {
        $(impl ConstBinopFields for $ty {
            fn operand0(&self) -> &llvm_ir::ConstantRef { &self.operand0 }
            fn operand1(&self) -> &llvm_ir::ConstantRef { &self.operand1 }
        })+
    };
}
impl_cbinop_fields!(
    llvm_ir::constant::Add,
    llvm_ir::constant::Sub,
    llvm_ir::constant::Mul,
);
