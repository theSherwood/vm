//! The AST the [`parse`](super::parse) reader produces and the translator (`lib.rs`) consumes —
//! the slice of `llvm-ir`'s data model we depend on, mirrored with the **same variant/field names**
//! so the translator's pattern-matches are unchanged, but **owned by us** (no libLLVM link, no
//! version ceiling) and with the **I14 fix baked in**: an integer constant carries its full value as
//! a `u128`, not a truncating `u64`.
//!
//! This file is data definitions + the few pure-Rust accessors the translator calls (`get_type`,
//! `try_get_result`, `named_struct_def`, …). It has no FFI and no parsing — parsing lives in
//! [`parse`](super::parse).
//!
//! Scope note: we mirror only the **subset the on-ramp consumes**. Where `llvm-ir` carries extra
//! fields the translator never reads (instruction `nuw`/`nsw`/`exact` flags, calling conventions,
//! call/parameter attributes), we omit them — the translator's `..`/named-field matches don't need
//! them and dropping them keeps us free of `llvm-ir`'s attribute/CC types. The pin is **LLVM 18**, so
//! the LLVM-18 shape is what we model (opaque pointers; explicit `loaded_ty`/`function_ty`).

use std::collections::HashMap;
use std::sync::Arc;

// ---- either ------------------------------------------------------------------------------------

/// A two-variant sum, mirroring the `either::Either` surface the translator uses (`as_ref`/`right`)
/// without taking the dependency. `Call`/`Invoke`'s callee is `Either<InlineAssembly, Operand>`.
#[derive(PartialEq, Clone, Debug)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    /// Borrow both sides — `Either<&L, &R>`. Mirrors `either::Either::as_ref`.
    pub fn as_ref(&self) -> Either<&L, &R> {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(r),
        }
    }
    /// The `Right` value, or `None`. Mirrors `either::Either::right`.
    pub fn right(self) -> Option<R> {
        match self {
            Either::Right(r) => Some(r),
            Either::Left(_) => None,
        }
    }
    /// The `Left` value, or `None`. Mirrors `either::Either::left`.
    pub fn left(self) -> Option<L> {
        match self {
            Either::Left(l) => Some(l),
            Either::Right(_) => None,
        }
    }
}

// ---- names -------------------------------------------------------------------------------------

/// An LLVM value/block name: a textual `%foo`/`@foo`/`%"quoted"` name, or a sequential number for an
/// unnamed value (`%3`). Mirrors `llvm_ir::Name`.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Name {
    /// A textual name (the inner `String` is the name without the leading sigil).
    Name(Box<String>),
    /// An unnamed value, given its sequential number.
    Number(usize),
}

impl Name {
    pub fn from_string(s: String) -> Self {
        Name::Name(Box::new(s))
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Name::Name(Box::new(s))
    }
}
impl From<usize> for Name {
    fn from(n: usize) -> Self {
        Name::Number(n)
    }
}

// ---- types -------------------------------------------------------------------------------------

/// Floating-point widths we model (the on-ramp handles f32/f64; the others exist so a type parses and
/// then fails closed downstream). Mirrors `llvm_ir::types::FPType` (names kept verbatim).
#[allow(non_camel_case_types)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum FPType {
    Half,
    BFloat,
    Single,
    Double,
    FP128,
    X86_FP80,
    PPC_FP128,
}

/// A pointer address space (LLVM `addrspace(N)`; `0` is the default). Mirrors `llvm_ir::types::AddrSpace`.
pub type AddrSpace = u32;

/// An interned, ref-counted handle to a [`Type`]. Mirrors `llvm_ir::types::TypeRef`.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct TypeRef(Arc<Type>);

impl TypeRef {
    pub fn new(ty: Type) -> Self {
        TypeRef(Arc::new(ty))
    }
}

/// The pointed-to [`Type`] — the translator calls `.as_ref()` pervasively (via this trait).
impl AsRef<Type> for TypeRef {
    fn as_ref(&self) -> &Type {
        &self.0
    }
}

/// Deref to the pointed-to [`Type`], mirroring `llvm_ir::types::TypeRef`: the translator passes
/// `&TypeRef` where `&Type` is expected (deref coercion) all over.
impl std::ops::Deref for TypeRef {
    type Target = Type;
    fn deref(&self) -> &Type {
        &self.0
    }
}

/// Display delegates to the pointed-to [`Type`] (mirrors `llvm_ir::types::TypeRef: Display`).
impl std::fmt::Display for TypeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

/// An LLVM type. The pointer form is the **opaque** LLVM-15+ shape (just an address space) — our pin
/// is LLVM 18. Mirrors `llvm_ir::types::Type` (the `llvm-18` feature subset; names kept verbatim).
#[allow(non_camel_case_types)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Type {
    VoidType,
    IntegerType {
        bits: u32,
    },
    /// Opaque pointer (LLVM 15+): carries only its address space.
    PointerType {
        addr_space: AddrSpace,
    },
    FPType(FPType),
    FuncType {
        result_type: TypeRef,
        param_types: Vec<TypeRef>,
        is_var_arg: bool,
    },
    VectorType {
        element_type: TypeRef,
        num_elements: usize,
        scalable: bool,
    },
    ArrayType {
        element_type: TypeRef,
        num_elements: usize,
    },
    /// A literal (anonymous) struct.
    StructType {
        element_types: Vec<TypeRef>,
        is_packed: bool,
    },
    /// A named struct; resolve its body via [`Types::named_struct_def`].
    NamedStructType {
        name: String,
    },
    X86_MMXType,
    X86_AMXType,
    MetadataType,
    LabelType,
    TokenType,
}

/// A concise textual form, used by the translator only in fail-closed error messages (`unsup("type
/// {ty}")`). Not the canonical LLVM syntax — just legible. Mirrors that `llvm_ir::Type: Display`.
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::VoidType => write!(f, "void"),
            Type::IntegerType { bits } => write!(f, "i{bits}"),
            Type::PointerType { addr_space } if *addr_space == 0 => write!(f, "ptr"),
            Type::PointerType { addr_space } => write!(f, "ptr addrspace({addr_space})"),
            Type::FPType(fp) => write!(f, "{}", fp.name()),
            Type::FuncType {
                result_type,
                is_var_arg,
                ..
            } => write!(
                f,
                "{result_type} (…{})",
                if *is_var_arg { ", ..." } else { "" }
            ),
            Type::VectorType {
                element_type,
                num_elements,
                scalable,
            } if *scalable => write!(f, "<vscale x {num_elements} x {}>", element_type.as_ref()),
            Type::VectorType {
                element_type,
                num_elements,
                ..
            } => write!(f, "<{num_elements} x {}>", element_type.as_ref()),
            Type::ArrayType {
                element_type,
                num_elements,
            } => write!(f, "[{num_elements} x {}]", element_type.as_ref()),
            Type::StructType { element_types, .. } => {
                write!(f, "{{ {} fields }}", element_types.len())
            }
            Type::NamedStructType { name } => write!(f, "%{name}"),
            Type::X86_MMXType => write!(f, "x86_mmx"),
            Type::X86_AMXType => write!(f, "x86_amx"),
            Type::MetadataType => write!(f, "metadata"),
            Type::LabelType => write!(f, "label"),
            Type::TokenType => write!(f, "token"),
        }
    }
}

impl FPType {
    /// The LLVM keyword for this float type (for diagnostics).
    pub fn name(self) -> &'static str {
        match self {
            FPType::Half => "half",
            FPType::BFloat => "bfloat",
            FPType::Single => "float",
            FPType::Double => "double",
            FPType::FP128 => "fp128",
            FPType::X86_FP80 => "x86_fp80",
            FPType::PPC_FP128 => "ppc_fp128",
        }
    }
}

/// The definition of a named struct. Mirrors `llvm_ir::types::NamedStructDef`: `Defined` carries a
/// `TypeRef` to the body's `StructType`; `Opaque` is a forward declaration with no body.
#[derive(Clone, Debug)]
pub enum NamedStructDef {
    Opaque,
    Defined(TypeRef),
}

/// The module type table. Mirrors the read-only `llvm_ir::types::Types` (`type_of`/`named_struct_def`
/// plus the `&self` constructors the translator's [`Typed`] impls call). Unlike `llvm-ir` we do *not*
/// intern, since structural [`PartialEq`] on [`TypeRef`] makes equal types compare equal regardless —
/// a constructor just hands back a fresh [`TypeRef`] (an AOT translate builds a bounded set). Named
/// struct *definitions* are registered as the parser reads `%struct.Foo = type { … }`.
#[derive(Clone, Debug, Default)]
pub struct Types {
    /// Named-struct definitions (`%struct.Foo = type { … }`), by name.
    named_struct_defs: HashMap<String, NamedStructDef>,
}

impl Types {
    pub fn new() -> Self {
        Types::default()
    }

    /// The type of anything that is [`Typed`] (`module.types.type_of(x)`). Mirrors `Types::type_of`.
    pub fn type_of<T: Typed + ?Sized>(&self, t: &T) -> TypeRef {
        t.get_type(self)
    }

    pub fn void(&self) -> TypeRef {
        TypeRef::new(Type::VoidType)
    }
    pub fn int(&self, bits: u32) -> TypeRef {
        TypeRef::new(Type::IntegerType { bits })
    }
    pub fn bool(&self) -> TypeRef {
        self.int(1)
    }
    pub fn pointer(&self, addr_space: AddrSpace) -> TypeRef {
        TypeRef::new(Type::PointerType { addr_space })
    }
    pub fn fp(&self, t: FPType) -> TypeRef {
        TypeRef::new(Type::FPType(t))
    }
    pub fn vector_of(&self, element_type: TypeRef, num_elements: usize, scalable: bool) -> TypeRef {
        TypeRef::new(Type::VectorType {
            element_type,
            num_elements,
            scalable,
        })
    }
    pub fn array_of(&self, element_type: TypeRef, num_elements: usize) -> TypeRef {
        TypeRef::new(Type::ArrayType {
            element_type,
            num_elements,
        })
    }
    pub fn struct_of(&self, element_types: Vec<TypeRef>, is_packed: bool) -> TypeRef {
        TypeRef::new(Type::StructType {
            element_types,
            is_packed,
        })
    }
    pub fn func_type(
        &self,
        result_type: TypeRef,
        param_types: Vec<TypeRef>,
        is_var_arg: bool,
    ) -> TypeRef {
        TypeRef::new(Type::FuncType {
            result_type,
            param_types,
            is_var_arg,
        })
    }
    pub fn named_struct(&self, name: String) -> TypeRef {
        TypeRef::new(Type::NamedStructType { name })
    }
    pub fn metadata_type(&self) -> TypeRef {
        TypeRef::new(Type::MetadataType)
    }
    pub fn label_type(&self) -> TypeRef {
        TypeRef::new(Type::LabelType)
    }
    pub fn token_type(&self) -> TypeRef {
        TypeRef::new(Type::TokenType)
    }

    /// Register (or replace) a named struct's definition. The parser calls this as it reads a
    /// `%struct.Foo = type { … }` line (`Opaque` for an opaque/forward declaration).
    pub fn add_named_struct_def(&mut self, name: String, def: NamedStructDef) {
        self.named_struct_defs.insert(name, def);
    }

    /// The definition of a named struct (`module.types.named_struct_def(name)`), or `None` if the
    /// name is unknown. Mirrors `llvm_ir::types::Types::named_struct_def`.
    pub fn named_struct_def(&self, name: &str) -> Option<&NamedStructDef> {
        self.named_struct_defs.get(name)
    }
}

/// The translator's `Typed::get_type(types)` — the static type of a value. Mirrors `llvm_ir::types::Typed`.
pub trait Typed {
    fn get_type(&self, types: &Types) -> TypeRef;
}

impl Typed for TypeRef {
    fn get_type(&self, _types: &Types) -> TypeRef {
        self.clone()
    }
}

// ---- constants ---------------------------------------------------------------------------------

/// An interned, ref-counted [`Constant`]. Mirrors `llvm_ir::constant::ConstantRef`.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ConstantRef(Arc<Constant>);

impl ConstantRef {
    pub fn new(c: Constant) -> Self {
        ConstantRef(Arc::new(c))
    }
}
impl AsRef<Constant> for ConstantRef {
    fn as_ref(&self) -> &Constant {
        &self.0
    }
}

/// Deref to the pointed-to [`Constant`], mirroring `llvm_ir::constant::ConstantRef` (the translator
/// passes `&ConstantRef` where `&Constant` is expected).
impl std::ops::Deref for ConstantRef {
    type Target = Constant;
    fn deref(&self) -> &Constant {
        &self.0
    }
}

impl Typed for ConstantRef {
    fn get_type(&self, types: &Types) -> TypeRef {
        self.0.get_type(types)
    }
}

/// A floating-point constant payload. Mirrors `llvm_ir::constant::Float` (the f32/f64 cases the
/// on-ramp handles; wider widths carry their raw bits so they parse then fail closed).
#[allow(non_camel_case_types)]
#[derive(PartialEq, Clone, Debug)]
pub enum Float {
    Half,
    BFloat,
    Single(f32),
    Double(f64),
    Quadruple,
    X86_FP80,
    PPC_FP128,
}

// Float is only ever compared/hashed via the enclosing Constant; give it Eq/Hash over the bit
// patterns so ConstantRef can derive them (NaN-bit-identity is fine for an IR constant table).
impl Eq for Float {}
#[allow(clippy::derived_hash_with_manual_eq)]
impl std::hash::Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Float::Single(f) => f.to_bits().hash(state),
            Float::Double(f) => f.to_bits().hash(state),
            _ => {}
        }
    }
}

impl Typed for Float {
    fn get_type(&self, types: &Types) -> TypeRef {
        types.fp(match self {
            Float::Half => FPType::Half,
            Float::BFloat => FPType::BFloat,
            Float::Single(_) => FPType::Single,
            Float::Double(_) => FPType::Double,
            Float::Quadruple => FPType::FP128,
            Float::X86_FP80 => FPType::X86_FP80,
            Float::PPC_FP128 => FPType::PPC_FP128,
        })
    }
}

/// An LLVM constant. Mirrors `llvm_ir::constant::Constant`, **with the I14 fix**: `Int.value` is a
/// full `u128` (the two's-complement bit pattern), never a truncating `u64`. We model the constant
/// expressions the on-ramp folds (`const_eval`); the others fail closed at the parser.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum Constant {
    Int {
        bits: u32,
        value: u128,
    },
    Float(Float),
    Null(TypeRef),
    AggregateZero(TypeRef),
    Struct {
        name: Option<String>,
        values: Vec<ConstantRef>,
        is_packed: bool,
    },
    Array {
        element_type: TypeRef,
        elements: Vec<ConstantRef>,
    },
    Vector(Vec<ConstantRef>),
    Undef(TypeRef),
    Poison(TypeRef),
    BlockAddress, // payload recovered structurally by the parser (see super::parse); mirrors llvm-ir
    GlobalReference {
        name: Name,
        ty: TypeRef,
    },
    TokenNone,
    /// Constant-expression GEP (`getelementptr` in a constant). `in_bounds` + the address + indices.
    GetElementPtr(ConstGetElementPtr),
    Trunc(ConstUnaryOp),
    ZExt(ConstUnaryOp),
    SExt(ConstUnaryOp),
    PtrToInt(ConstUnaryOp),
    IntToPtr(ConstUnaryOp),
    BitCast(ConstUnaryOp),
    AddrSpaceCast(ConstUnaryOp),
    Add(ConstBinaryOp),
    Sub(ConstBinaryOp),
    Mul(ConstBinaryOp),
}

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ConstUnaryOp {
    pub operand: ConstantRef,
    pub to_type: TypeRef,
}

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ConstBinaryOp {
    pub operand0: ConstantRef,
    pub operand1: ConstantRef,
}

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct ConstGetElementPtr {
    pub address: ConstantRef,
    pub indices: Vec<ConstantRef>,
    pub in_bounds: bool,
}

impl Typed for Constant {
    fn get_type(&self, types: &Types) -> TypeRef {
        match self {
            Constant::Int { bits, .. } => types.int(*bits),
            Constant::Float(f) => types.type_of(f),
            Constant::Null(t) => t.clone(),
            Constant::AggregateZero(t) => t.clone(),
            Constant::Struct {
                values, is_packed, ..
            } => types.struct_of(
                values.iter().map(|v| types.type_of(v)).collect(),
                *is_packed,
            ),
            Constant::Array {
                element_type,
                elements,
            } => types.array_of(element_type.clone(), elements.len()),
            Constant::Vector(v) => types.vector_of(types.type_of(&v[0]), v.len(), false),
            Constant::Undef(t) => t.clone(),
            Constant::Poison(t) => t.clone(),
            Constant::BlockAddress => types.label_type(),
            // LLVM 15+: a global reference is an opaque pointer (address space 0).
            Constant::GlobalReference { .. } => types.pointer(0),
            Constant::TokenNone => types.token_type(),
            Constant::GetElementPtr(_) => types.pointer(0),
            Constant::Trunc(u)
            | Constant::ZExt(u)
            | Constant::SExt(u)
            | Constant::PtrToInt(u)
            | Constant::IntToPtr(u)
            | Constant::BitCast(u)
            | Constant::AddrSpaceCast(u) => u.to_type.clone(),
            Constant::Add(b) | Constant::Sub(b) | Constant::Mul(b) => types.type_of(&b.operand0),
        }
    }
}

// ---- operands ----------------------------------------------------------------------------------

/// An instruction operand. Mirrors `llvm_ir::Operand`.
#[derive(PartialEq, Clone, Debug)]
pub enum Operand {
    LocalOperand { name: Name, ty: TypeRef },
    ConstantOperand(ConstantRef),
    MetadataOperand,
}

impl Operand {
    /// The inner [`Constant`] if this operand is a constant, else `None`. Mirrors `Operand::as_constant`.
    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Operand::ConstantOperand(cref) => Some(cref.as_ref()),
            _ => None,
        }
    }
}

impl Typed for Operand {
    fn get_type(&self, types: &Types) -> TypeRef {
        match self {
            Operand::LocalOperand { ty, .. } => ty.clone(),
            Operand::ConstantOperand(c) => types.type_of(c),
            Operand::MetadataOperand => types.metadata_type(),
        }
    }
}

// ---- predicates --------------------------------------------------------------------------------

/// Integer comparison predicate. Mirrors `llvm_ir::IntPredicate`.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum IntPredicate {
    EQ,
    NE,
    UGT,
    UGE,
    ULT,
    ULE,
    SGT,
    SGE,
    SLT,
    SLE,
}

/// Floating-point comparison predicate. Mirrors `llvm_ir::FPPredicate`.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum FPPredicate {
    False,
    OEQ,
    OGT,
    OGE,
    OLT,
    OLE,
    ONE,
    ORD,
    UNO,
    UEQ,
    UGT,
    UGE,
    ULT,
    ULE,
    UNE,
    True,
}

// ---- atomics -----------------------------------------------------------------------------------

/// Memory ordering on an atomic operation. Mirrors `llvm_ir::instruction::MemoryOrdering`.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum MemoryOrdering {
    Unordered,
    Monotonic,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
    NotAtomic,
}

/// The synchronization scope of an atomic operation. Mirrors `llvm_ir::instruction::SynchronizationScope`.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum SynchronizationScope {
    SingleThread,
    System,
}

/// The atomicity of a memory operation (`load atomic`/`store atomic`/`cmpxchg`/`atomicrmw`).
/// Mirrors `llvm_ir::instruction::Atomicity`. The translator only checks presence (`Option::is_some`),
/// but the fields are kept faithful.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct Atomicity {
    pub synch_scope: SynchronizationScope,
    pub mem_ordering: MemoryOrdering,
}

/// The binary operation of an `atomicrmw`. Mirrors `llvm_ir::instruction::RMWBinOp` (the LLVM-18
/// subset; the LLVM-19 `UIncWrap`/`UDecWrap` are excluded by the pin).
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum RMWBinOp {
    Xchg,
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
    FAdd,
    FSub,
    FMax,
    FMin,
}

// ---- inline asm / attributes / EH (modeled minimally) ------------------------------------------

/// An inline-assembly callee. The on-ramp does not *execute* asm (that would be opaque, unmaskable
/// machine code — the whole sandbox thesis is that everything is re-verified); instead a fixed
/// **recognize-and-lower** allowlist matches the handful of template strings known C headers emit
/// (compiler barriers, `popcnt`, the x86 atomics) and re-emits their semantics as ordinary verified
/// IR, failing closed on anything else. So it carries the `template`/`constraints` the recognizer
/// keys on. Mirrors the consumed shape of `llvm_ir::instruction::InlineAssembly` plus the two strings
/// the LLVM-C API exposes (`LLVMGetInlineAsmAsmString`/`…ConstraintString`).
#[derive(PartialEq, Clone, Debug)]
pub struct InlineAssembly {
    pub ty: TypeRef,
    /// The asm template with `\XX` escapes still encoded (as the `.ll` prints them); the recognizer
    /// decodes before matching.
    pub template: String,
    /// The constraint string (`"=q,=*m,0,*m,~{memory},…"`) — pins operand in/out/clobber roles.
    pub constraints: String,
}

/// A call/parameter attribute. The on-ramp ignores these (it reads only the operand of each
/// argument), so the type is an uninhabited placeholder: argument attribute lists are always empty.
/// Mirrors the *position* of `llvm_ir::function::ParameterAttribute` without modeling its variants.
#[derive(PartialEq, Clone, Debug)]
pub enum ParameterAttribute {}

/// A `landingpad` clause. Like `llvm-ir`, the LLVM C API does not expose the fields, so this is empty;
/// the on-ramp reads exceptions through its own EH slots, not the clause list.
#[derive(PartialEq, Clone, Debug)]
pub struct LandingPadClause {}

// ---- debug locations ---------------------------------------------------------------------------

/// A source location attached to an instruction (`!dbg`). Mirrors `llvm_ir::debugloc::DebugLoc`.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DebugLoc {
    pub line: u32,
    pub col: Option<u32>,
    pub filename: String,
    pub directory: Option<String>,
}

/// The `HasDebugLoc` accessor trait the translator uses. Mirrors `llvm_ir::debugloc::HasDebugLoc`.
pub trait HasDebugLoc {
    fn get_debug_loc(&self) -> &Option<DebugLoc>;
}

// ---- module structure --------------------------------------------------------------------------

/// A function parameter (`%name : ty`). Mirrors the `Parameter` fields the translator reads.
#[derive(PartialEq, Clone, Debug)]
pub struct Parameter {
    pub name: Name,
    pub ty: TypeRef,
}

/// A defined function. Mirrors the `llvm_ir::Function` fields the translator reads (the rest —
/// linkage/visibility/attributes — are not consumed by the on-ramp, so omitted).
#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub is_var_arg: bool,
    pub return_type: TypeRef,
    pub basic_blocks: Vec<BasicBlock>,
}

/// A declared-but-not-defined function (a prototype). The translator only needs its name + signature
/// to resolve calls; mirrors the consumed subset of `llvm_ir::FunctionDeclaration`.
#[derive(Clone, Debug)]
pub struct FunctionDeclaration {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub is_var_arg: bool,
    pub return_type: TypeRef,
}

/// A global variable. Mirrors the consumed subset of `llvm_ir::GlobalVariable`.
#[derive(Clone, Debug)]
pub struct GlobalVariable {
    pub name: Name,
    pub ty: TypeRef,
    pub initializer: Option<ConstantRef>,
    pub is_constant: bool,
    pub alignment: u32,
}

/// A global alias (`@a = alias … @b`). Mirrors the consumed subset of `llvm_ir::GlobalAlias`.
#[derive(Clone, Debug)]
pub struct GlobalAlias {
    pub name: Name,
    pub aliasee: ConstantRef,
    pub ty: TypeRef,
}

/// A basic block. Mirrors `llvm_ir::BasicBlock`.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub name: Name,
    pub instrs: Vec<Instruction>,
    pub term: Terminator,
}

/// A whole module. Mirrors the consumed subset of `llvm_ir::Module` (`func_declarations` separate from
/// `functions`, exactly like upstream).
#[derive(Clone, Debug)]
pub struct Module {
    pub name: String,
    pub source_file_name: String,
    pub target_triple: Option<String>,
    pub functions: Vec<Function>,
    pub func_declarations: Vec<FunctionDeclaration>,
    pub global_vars: Vec<GlobalVariable>,
    pub global_aliases: Vec<GlobalAlias>,
    pub types: Types,
}

// ---- instructions ------------------------------------------------------------------------------

/// A non-terminator instruction. Mirrors `llvm_ir::instruction::Instruction` (LLVM-18 subset). Each
/// variant is a struct holding the operand/dest fields the translator reads.
#[derive(PartialEq, Clone, Debug)]
pub enum Instruction {
    // Integer binary ops
    Add(BinaryOp),
    Sub(BinaryOp),
    Mul(BinaryOp),
    UDiv(BinaryOp),
    SDiv(BinaryOp),
    URem(BinaryOp),
    SRem(BinaryOp),
    // Bitwise binary ops
    And(BinaryOp),
    Or(BinaryOp),
    Xor(BinaryOp),
    Shl(BinaryOp),
    LShr(BinaryOp),
    AShr(BinaryOp),
    // Floating-point binary ops
    FAdd(BinaryOp),
    FSub(BinaryOp),
    FMul(BinaryOp),
    FDiv(BinaryOp),
    FRem(BinaryOp),
    FNeg(UnaryOp),
    // Vector ops
    ExtractElement(ExtractElement),
    InsertElement(InsertElement),
    ShuffleVector(ShuffleVector),
    // Aggregate ops
    ExtractValue(ExtractValue),
    InsertValue(InsertValue),
    // Memory ops
    Alloca(Alloca),
    Load(Load),
    Store(Store),
    Fence(Fence),
    CmpXchg(CmpXchg),
    AtomicRMW(AtomicRMW),
    GetElementPtr(GetElementPtr),
    // Conversion ops
    Trunc(UnaryOp),
    ZExt(UnaryOp),
    SExt(UnaryOp),
    FPTrunc(UnaryOp),
    FPExt(UnaryOp),
    FPToUI(UnaryOp),
    FPToSI(UnaryOp),
    UIToFP(UnaryOp),
    SIToFP(UnaryOp),
    PtrToInt(UnaryOp),
    IntToPtr(UnaryOp),
    BitCast(UnaryOp),
    AddrSpaceCast(UnaryOp),
    // Other ops
    ICmp(ICmp),
    FCmp(FCmp),
    Phi(Phi),
    Select(Select),
    Freeze(UnaryOp),
    Call(Call),
    VAArg(VAArg),
    LandingPad(LandingPad),
    CatchPad(CatchPad),
    CleanupPad(CleanupPad),
}

/// A binary-operation body (`<dest> = <op> <ty> <operand0>, <operand1>`). The `nuw`/`nsw`/`exact`
/// flags `llvm-ir` carries are dropped — the on-ramp defines wrap semantics and never reads them.
#[derive(PartialEq, Clone, Debug)]
pub struct BinaryOp {
    pub operand0: Operand,
    pub operand1: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

/// A unary op with the result type the same as the operand (`fneg`/`freeze`) **or** an
/// explicitly-typed conversion (`trunc`/`zext`/…/`bitcast`). `to_type` is meaningful for the
/// conversions; for `fneg`/`freeze` it equals the operand type (the parser fills it in).
#[derive(PartialEq, Clone, Debug)]
pub struct UnaryOp {
    pub operand: Operand,
    pub to_type: TypeRef,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ExtractElement {
    pub vector: Operand,
    pub index: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct InsertElement {
    pub vector: Operand,
    pub element: Operand,
    pub index: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ShuffleVector {
    pub operand0: Operand,
    pub operand1: Operand,
    pub dest: Name,
    pub mask: ConstantRef,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ExtractValue {
    pub aggregate: Operand,
    pub indices: Vec<u32>,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct InsertValue {
    pub aggregate: Operand,
    pub element: Operand,
    pub indices: Vec<u32>,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Alloca {
    pub allocated_type: TypeRef,
    pub num_elements: Operand,
    pub dest: Name,
    pub alignment: u32,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Load {
    pub address: Operand,
    pub dest: Name,
    /// LLVM 15+: the loaded type is explicit (opaque pointers carry no pointee).
    pub loaded_ty: TypeRef,
    pub volatile: bool,
    pub atomicity: Option<Atomicity>,
    pub alignment: u32,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Store {
    pub address: Operand,
    pub value: Operand,
    pub volatile: bool,
    pub atomicity: Option<Atomicity>,
    pub alignment: u32,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Fence {
    pub atomicity: Atomicity,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CmpXchg {
    pub address: Operand,
    pub expected: Operand,
    pub replacement: Operand,
    pub dest: Name,
    pub volatile: bool,
    pub atomicity: Atomicity,
    pub failure_memory_ordering: MemoryOrdering,
    pub weak: bool,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct AtomicRMW {
    pub operation: RMWBinOp,
    pub address: Operand,
    pub value: Operand,
    pub dest: Name,
    pub volatile: bool,
    pub atomicity: Atomicity,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct GetElementPtr {
    pub address: Operand,
    pub indices: Vec<Operand>,
    pub dest: Name,
    pub in_bounds: bool,
    /// LLVM 14+: the source element type the indices walk (opaque pointers carry no pointee).
    pub source_element_type: TypeRef,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct ICmp {
    pub predicate: IntPredicate,
    pub operand0: Operand,
    pub operand1: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct FCmp {
    pub predicate: FPPredicate,
    pub operand0: Operand,
    pub operand1: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Phi {
    pub incoming_values: Vec<(Operand, Name)>,
    pub dest: Name,
    pub to_type: TypeRef,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Select {
    pub condition: Operand,
    pub true_value: Operand,
    pub false_value: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Call {
    pub function: Either<InlineAssembly, Operand>,
    /// LLVM 15+: the callee's function type (the indices/return type the on-ramp reads for an
    /// indirect call), since an opaque pointer callee carries no signature.
    pub function_ty: TypeRef,
    pub arguments: Vec<(Operand, Vec<ParameterAttribute>)>,
    pub dest: Option<Name>,
    pub is_tail_call: bool,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct VAArg {
    pub arg_list: Operand,
    pub cur_type: TypeRef,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct LandingPad {
    pub result_type: TypeRef,
    pub clauses: Vec<LandingPadClause>,
    pub dest: Name,
    pub cleanup: bool,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CatchPad {
    pub catch_switch: Operand,
    pub args: Vec<Operand>,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CleanupPad {
    pub parent_pad: Operand,
    pub args: Vec<Operand>,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

impl Instruction {
    /// The destination name this instruction defines, if any (a `void`-typed effectful instruction has
    /// none). Mirrors `llvm_ir::instruction::Instruction::try_get_result`.
    pub fn try_get_result(&self) -> Option<&Name> {
        match self {
            Instruction::Add(i)
            | Instruction::Sub(i)
            | Instruction::Mul(i)
            | Instruction::UDiv(i)
            | Instruction::SDiv(i)
            | Instruction::URem(i)
            | Instruction::SRem(i)
            | Instruction::And(i)
            | Instruction::Or(i)
            | Instruction::Xor(i)
            | Instruction::Shl(i)
            | Instruction::LShr(i)
            | Instruction::AShr(i)
            | Instruction::FAdd(i)
            | Instruction::FSub(i)
            | Instruction::FMul(i)
            | Instruction::FDiv(i)
            | Instruction::FRem(i) => Some(&i.dest),
            Instruction::FNeg(i)
            | Instruction::Trunc(i)
            | Instruction::ZExt(i)
            | Instruction::SExt(i)
            | Instruction::FPTrunc(i)
            | Instruction::FPExt(i)
            | Instruction::FPToUI(i)
            | Instruction::FPToSI(i)
            | Instruction::UIToFP(i)
            | Instruction::SIToFP(i)
            | Instruction::PtrToInt(i)
            | Instruction::IntToPtr(i)
            | Instruction::BitCast(i)
            | Instruction::AddrSpaceCast(i)
            | Instruction::Freeze(i) => Some(&i.dest),
            Instruction::ExtractElement(i) => Some(&i.dest),
            Instruction::InsertElement(i) => Some(&i.dest),
            Instruction::ShuffleVector(i) => Some(&i.dest),
            Instruction::ExtractValue(i) => Some(&i.dest),
            Instruction::InsertValue(i) => Some(&i.dest),
            Instruction::Alloca(i) => Some(&i.dest),
            Instruction::Load(i) => Some(&i.dest),
            Instruction::Store(_) => None,
            Instruction::Fence(_) => None,
            Instruction::CmpXchg(i) => Some(&i.dest),
            Instruction::AtomicRMW(i) => Some(&i.dest),
            Instruction::GetElementPtr(i) => Some(&i.dest),
            Instruction::ICmp(i) => Some(&i.dest),
            Instruction::FCmp(i) => Some(&i.dest),
            Instruction::Phi(i) => Some(&i.dest),
            Instruction::Select(i) => Some(&i.dest),
            Instruction::Call(i) => i.dest.as_ref(),
            Instruction::VAArg(i) => Some(&i.dest),
            Instruction::LandingPad(i) => Some(&i.dest),
            Instruction::CatchPad(i) => Some(&i.dest),
            Instruction::CleanupPad(i) => Some(&i.dest),
        }
    }
}

impl HasDebugLoc for Instruction {
    fn get_debug_loc(&self) -> &Option<DebugLoc> {
        match self {
            Instruction::Add(i)
            | Instruction::Sub(i)
            | Instruction::Mul(i)
            | Instruction::UDiv(i)
            | Instruction::SDiv(i)
            | Instruction::URem(i)
            | Instruction::SRem(i)
            | Instruction::And(i)
            | Instruction::Or(i)
            | Instruction::Xor(i)
            | Instruction::Shl(i)
            | Instruction::LShr(i)
            | Instruction::AShr(i)
            | Instruction::FAdd(i)
            | Instruction::FSub(i)
            | Instruction::FMul(i)
            | Instruction::FDiv(i)
            | Instruction::FRem(i) => i.get_debug_loc(),
            Instruction::FNeg(i)
            | Instruction::Trunc(i)
            | Instruction::ZExt(i)
            | Instruction::SExt(i)
            | Instruction::FPTrunc(i)
            | Instruction::FPExt(i)
            | Instruction::FPToUI(i)
            | Instruction::FPToSI(i)
            | Instruction::UIToFP(i)
            | Instruction::SIToFP(i)
            | Instruction::PtrToInt(i)
            | Instruction::IntToPtr(i)
            | Instruction::BitCast(i)
            | Instruction::AddrSpaceCast(i)
            | Instruction::Freeze(i) => i.get_debug_loc(),
            Instruction::ExtractElement(i) => i.get_debug_loc(),
            Instruction::InsertElement(i) => i.get_debug_loc(),
            Instruction::ShuffleVector(i) => i.get_debug_loc(),
            Instruction::ExtractValue(i) => i.get_debug_loc(),
            Instruction::InsertValue(i) => i.get_debug_loc(),
            Instruction::Alloca(i) => i.get_debug_loc(),
            Instruction::Load(i) => i.get_debug_loc(),
            Instruction::Store(i) => i.get_debug_loc(),
            Instruction::Fence(i) => i.get_debug_loc(),
            Instruction::CmpXchg(i) => i.get_debug_loc(),
            Instruction::AtomicRMW(i) => i.get_debug_loc(),
            Instruction::GetElementPtr(i) => i.get_debug_loc(),
            Instruction::ICmp(i) => i.get_debug_loc(),
            Instruction::FCmp(i) => i.get_debug_loc(),
            Instruction::Phi(i) => i.get_debug_loc(),
            Instruction::Select(i) => i.get_debug_loc(),
            Instruction::Call(i) => i.get_debug_loc(),
            Instruction::VAArg(i) => i.get_debug_loc(),
            Instruction::LandingPad(i) => i.get_debug_loc(),
            Instruction::CatchPad(i) => i.get_debug_loc(),
            Instruction::CleanupPad(i) => i.get_debug_loc(),
        }
    }
}

impl Instruction {
    /// Parser-side setter for the source location, resolved from an instruction's `!dbg` attachment
    /// (the textual reader's analog of the bitcode reader's per-instruction `DebugLoc`). Mirrors the
    /// variant grouping of [`Instruction::get_debug_loc`].
    pub fn set_debug_loc(&mut self, dl: Option<DebugLoc>) {
        use Instruction::*;
        let slot = match self {
            Add(i) | Sub(i) | Mul(i) | UDiv(i) | SDiv(i) | URem(i) | SRem(i) | And(i) | Or(i)
            | Xor(i) | Shl(i) | LShr(i) | AShr(i) | FAdd(i) | FSub(i) | FMul(i) | FDiv(i)
            | FRem(i) => &mut i.debugloc,
            FNeg(i) | Trunc(i) | ZExt(i) | SExt(i) | FPTrunc(i) | FPExt(i) | FPToUI(i)
            | FPToSI(i) | UIToFP(i) | SIToFP(i) | PtrToInt(i) | IntToPtr(i) | BitCast(i)
            | AddrSpaceCast(i) | Freeze(i) => &mut i.debugloc,
            ExtractElement(i) => &mut i.debugloc,
            InsertElement(i) => &mut i.debugloc,
            ShuffleVector(i) => &mut i.debugloc,
            ExtractValue(i) => &mut i.debugloc,
            InsertValue(i) => &mut i.debugloc,
            Alloca(i) => &mut i.debugloc,
            Load(i) => &mut i.debugloc,
            Store(i) => &mut i.debugloc,
            Fence(i) => &mut i.debugloc,
            CmpXchg(i) => &mut i.debugloc,
            AtomicRMW(i) => &mut i.debugloc,
            GetElementPtr(i) => &mut i.debugloc,
            ICmp(i) => &mut i.debugloc,
            FCmp(i) => &mut i.debugloc,
            Phi(i) => &mut i.debugloc,
            Select(i) => &mut i.debugloc,
            Call(i) => &mut i.debugloc,
            VAArg(i) => &mut i.debugloc,
            LandingPad(i) => &mut i.debugloc,
            CatchPad(i) => &mut i.debugloc,
            CleanupPad(i) => &mut i.debugloc,
        };
        *slot = dl;
    }
}

/// The element type of a vector `TypeRef`, or a panic (the operand of a vector op must be a vector —
/// mirrors `llvm-ir`'s behavior; the on-ramp only calls this on verified-valid IR).
fn vec_element_type(ty: &TypeRef) -> TypeRef {
    match ty.as_ref() {
        Type::VectorType { element_type, .. } => element_type.clone(),
        ty => panic!("Expected a vector type, got {ty:?}"),
    }
}

/// Walk an aggregate type by `extractvalue`/`insertvalue` indices to the leaf field type. Mirrors
/// `llvm-ir`'s `ev_type` (literal array/struct only; named-struct resolution is not needed for the
/// register-coerced aggregates clang emits).
fn ev_type(cur_type: TypeRef, indices: &[u32]) -> TypeRef {
    match indices.split_first() {
        None => cur_type,
        Some((&index, rest)) => match cur_type.as_ref() {
            Type::ArrayType { element_type, .. } => ev_type(element_type.clone(), rest),
            Type::StructType { element_types, .. } => ev_type(
                element_types
                    .get(index as usize)
                    .expect("ExtractValue index out of range")
                    .clone(),
                rest,
            ),
            _ => panic!("ExtractValue from a non-aggregate type {cur_type:?}"),
        },
    }
}

/// `icmp`/`fcmp` result type: `i1`, or `<N x i1>` for a vector compare. Mirrors `llvm-ir`.
fn cmp_result_type(operand_ty: TypeRef, types: &Types) -> TypeRef {
    match operand_ty.as_ref() {
        Type::VectorType {
            num_elements,
            scalable,
            ..
        } => types.vector_of(types.bool(), *num_elements, *scalable),
        _ => types.bool(),
    }
}

impl Typed for Instruction {
    fn get_type(&self, types: &Types) -> TypeRef {
        match self {
            // result type == operand0 type
            Instruction::Add(i)
            | Instruction::Sub(i)
            | Instruction::Mul(i)
            | Instruction::UDiv(i)
            | Instruction::SDiv(i)
            | Instruction::URem(i)
            | Instruction::SRem(i)
            | Instruction::And(i)
            | Instruction::Or(i)
            | Instruction::Xor(i)
            | Instruction::Shl(i)
            | Instruction::LShr(i)
            | Instruction::AShr(i)
            | Instruction::FAdd(i)
            | Instruction::FSub(i)
            | Instruction::FMul(i)
            | Instruction::FDiv(i)
            | Instruction::FRem(i) => types.type_of(&i.operand0),
            // result type == operand type / the explicit `to_type`
            Instruction::FNeg(i) | Instruction::Freeze(i) => types.type_of(&i.operand),
            Instruction::Trunc(i)
            | Instruction::ZExt(i)
            | Instruction::SExt(i)
            | Instruction::FPTrunc(i)
            | Instruction::FPExt(i)
            | Instruction::FPToUI(i)
            | Instruction::FPToSI(i)
            | Instruction::UIToFP(i)
            | Instruction::SIToFP(i)
            | Instruction::PtrToInt(i)
            | Instruction::IntToPtr(i)
            | Instruction::BitCast(i)
            | Instruction::AddrSpaceCast(i) => i.to_type.clone(),
            Instruction::ExtractElement(i) => vec_element_type(&types.type_of(&i.vector)),
            Instruction::InsertElement(i) => types.type_of(&i.vector),
            Instruction::ShuffleVector(i) => {
                let elem = vec_element_type(&types.type_of(&i.operand0));
                match types.type_of(&i.mask).as_ref() {
                    Type::VectorType {
                        num_elements,
                        scalable,
                        ..
                    } => types.vector_of(elem, *num_elements, *scalable),
                    ty => panic!("Expected a ShuffleVector mask to be a vector, got {ty:?}"),
                }
            }
            Instruction::ExtractValue(i) => ev_type(types.type_of(&i.aggregate), &i.indices),
            Instruction::InsertValue(i) => types.type_of(&i.aggregate),
            // LLVM 15+: alloca/gep produce an opaque pointer.
            Instruction::Alloca(_) | Instruction::GetElementPtr(_) => types.pointer(0),
            Instruction::Load(i) => i.loaded_ty.clone(),
            Instruction::Store(_) | Instruction::Fence(_) => types.void(),
            Instruction::CmpXchg(i) => {
                let ty = types.type_of(&i.expected);
                types.struct_of(vec![ty, types.bool()], false)
            }
            Instruction::AtomicRMW(i) => types.type_of(&i.value),
            Instruction::ICmp(i) => cmp_result_type(types.type_of(&i.operand0), types),
            Instruction::FCmp(i) => cmp_result_type(types.type_of(&i.operand0), types),
            Instruction::Phi(i) => i.to_type.clone(),
            Instruction::Select(i) => types.type_of(&i.true_value),
            // LLVM 15+: the result type is read off the explicit `function_ty`.
            Instruction::Call(i) => match i.function_ty.as_ref() {
                Type::FuncType { result_type, .. } => result_type.clone(),
                ty => panic!("Expected Call.function_ty to be a FuncType, got {ty:?}"),
            },
            Instruction::VAArg(i) => i.cur_type.clone(),
            Instruction::LandingPad(i) => i.result_type.clone(),
            Instruction::CatchPad(_) | Instruction::CleanupPad(_) => types.token_type(),
        }
    }
}

macro_rules! impl_has_debug_loc {
    ($($ty:ty),+ $(,)?) => {
        $(impl HasDebugLoc for $ty {
            fn get_debug_loc(&self) -> &Option<DebugLoc> {
                &self.debugloc
            }
        })+
    };
}
impl_has_debug_loc!(
    BinaryOp,
    UnaryOp,
    ExtractElement,
    InsertElement,
    ShuffleVector,
    ExtractValue,
    InsertValue,
    Alloca,
    Load,
    Store,
    Fence,
    CmpXchg,
    AtomicRMW,
    GetElementPtr,
    ICmp,
    FCmp,
    Phi,
    Select,
    Call,
    VAArg,
    LandingPad,
    CatchPad,
    CleanupPad,
);

// ---- terminators -------------------------------------------------------------------------------

/// A block terminator. Mirrors `llvm_ir::terminator::Terminator` (the full LLVM-18 set; the on-ramp
/// translates `Ret`/`Br`/`CondBr`/`Switch`/`IndirectBr`/`Invoke`/`Resume`/`Unreachable` and fails
/// closed on the funclet-EH ones via its `other => unsup(..)` arm).
#[derive(PartialEq, Clone, Debug)]
pub enum Terminator {
    Ret(Ret),
    Br(Br),
    CondBr(CondBr),
    Switch(Switch),
    IndirectBr(IndirectBr),
    Invoke(Invoke),
    Resume(Resume),
    Unreachable(Unreachable),
    CleanupRet(CleanupRet),
    CatchRet(CatchRet),
    CatchSwitch(CatchSwitch),
    CallBr(CallBr),
}

#[derive(PartialEq, Clone, Debug)]
pub struct Ret {
    pub return_operand: Option<Operand>,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Br {
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CondBr {
    pub condition: Operand,
    pub true_dest: Name,
    pub false_dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Switch {
    pub operand: Operand,
    pub dests: Vec<(ConstantRef, Name)>,
    pub default_dest: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct IndirectBr {
    pub operand: Operand,
    pub possible_dests: Vec<Name>,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Invoke {
    pub function: Either<InlineAssembly, Operand>,
    pub function_ty: TypeRef,
    pub arguments: Vec<(Operand, Vec<ParameterAttribute>)>,
    pub result: Name,
    pub return_label: Name,
    pub exception_label: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Resume {
    pub operand: Operand,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct Unreachable {
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CleanupRet {
    pub cleanup_pad: Operand,
    pub unwind_dest: Option<Name>,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CatchRet {
    pub catch_pad: Operand,
    pub successor: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CatchSwitch {
    pub parent_pad: Operand,
    pub catch_handlers: Vec<Name>,
    pub default_unwind_dest: Option<Name>,
    pub result: Name,
    pub debugloc: Option<DebugLoc>,
}

#[derive(PartialEq, Clone, Debug)]
pub struct CallBr {
    pub function: Either<InlineAssembly, Operand>,
    pub function_ty: TypeRef,
    pub arguments: Vec<(Operand, Vec<ParameterAttribute>)>,
    pub result: Name,
    pub return_label: Name,
    pub debugloc: Option<DebugLoc>,
}

impl_has_debug_loc!(
    Ret,
    Br,
    CondBr,
    Switch,
    IndirectBr,
    Invoke,
    Resume,
    Unreachable,
    CleanupRet,
    CatchRet,
    CatchSwitch,
    CallBr,
);
