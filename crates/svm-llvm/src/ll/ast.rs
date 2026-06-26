//! The AST the [`parse`](super::parse) reader produces and the translator (`lib.rs`) consumes —
//! the slice of `llvm-ir`'s data model we depend on, mirrored with the **same variant/field names**
//! so the translator's pattern-matches are unchanged, but **owned by us** (no libLLVM link, no
//! version ceiling) and with the **I14 fix baked in**: an integer constant carries its full value as
//! a `u128`, not a truncating `u64`.
//!
//! This file is data definitions + the few pure-Rust accessors the translator calls (`get_type`,
//! `try_get_result`, `named_struct_def`, …). It has no FFI and no parsing — parsing lives in
//! [`parse`](super::parse).

use std::collections::HashMap;
use std::sync::Arc;

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

/// The module type table: the interner that hands out cached [`TypeRef`]s and resolves named structs.
/// Mirrors the subset of `llvm_ir::types::Types` the translator uses (`named_struct_def`, plus the
/// constructors the parser needs). Named-struct *bodies* are registered as they are parsed.
#[derive(Clone, Debug, Default)]
pub struct Types {
    /// Named-struct definitions (`%struct.Foo = type { … }`), by name. `None` body ⇒ opaque.
    named_structs: HashMap<String, Option<Vec<TypeRef>>>,
    /// Cache for the common nullary/parameterized types so equal types share a `TypeRef`.
    cache: HashMap<Type, TypeRef>,
}

impl Types {
    pub fn new() -> Self {
        Types::default()
    }

    /// Intern a [`Type`], returning a cached [`TypeRef`] (so structurally-equal types are pointer-equal
    /// and cheap to clone/compare).
    pub fn get(&mut self, ty: Type) -> TypeRef {
        if let Some(r) = self.cache.get(&ty) {
            return r.clone();
        }
        let r = TypeRef::new(ty.clone());
        self.cache.insert(ty, r.clone());
        r
    }

    pub fn int(&mut self, bits: u32) -> TypeRef {
        self.get(Type::IntegerType { bits })
    }
    pub fn void(&mut self) -> TypeRef {
        self.get(Type::VoidType)
    }
    pub fn pointer(&mut self, addr_space: AddrSpace) -> TypeRef {
        self.get(Type::PointerType { addr_space })
    }
    pub fn fp(&mut self, t: FPType) -> TypeRef {
        self.get(Type::FPType(t))
    }

    /// Declare (or redeclare) a named struct. `body == None` ⇒ opaque/forward declaration.
    pub fn register_named_struct(&mut self, name: String, body: Option<Vec<TypeRef>>) {
        self.named_structs.insert(name, body);
    }

    /// The field types of a named struct (`module.types.named_struct_def(name)`), or `None` if the
    /// name is unknown or opaque. Mirrors `llvm_ir::types::Types::named_struct_def`.
    pub fn named_struct_def(&self, name: &str) -> Option<&Vec<TypeRef>> {
        self.named_structs.get(name).and_then(|b| b.as_ref())
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

/// An LLVM constant. Mirrors `llvm_ir::constant::Constant`, **with the I14 fix**: `Int.value` is a
/// full `u128` (the two's-complement bit pattern), never a truncating `u64`.
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

// ---- operands ----------------------------------------------------------------------------------

/// An instruction operand. Mirrors `llvm_ir::Operand`.
#[derive(PartialEq, Clone, Debug)]
pub enum Operand {
    LocalOperand { name: Name, ty: TypeRef },
    ConstantOperand(ConstantRef),
    MetadataOperand,
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

// ---- instructions & terminators ----------------------------------------------------------------
//
// NOTE (PR1, in progress): the `Instruction` and `Terminator` enums are populated incrementally,
// driven by compiling `lib.rs` against this module (the translator's matches are the exact spec).
// The trivial vertical slice below (binary ops + ret/br) is the seed; the full ~48-variant set lands
// as coverage grows under the parity harness. See super::mod docs.

/// An instruction (non-terminator). Mirrors `llvm_ir::instruction::Instruction`. Each variant is a
/// struct with the operand/dest fields the translator reads.
#[derive(PartialEq, Clone, Debug)]
pub enum Instruction {
    Add(BinaryOp),
    Sub(BinaryOp),
    Mul(BinaryOp),
    UDiv(BinaryOp),
    SDiv(BinaryOp),
    URem(BinaryOp),
    SRem(BinaryOp),
    And(BinaryOp),
    Or(BinaryOp),
    Xor(BinaryOp),
    Shl(BinaryOp),
    LShr(BinaryOp),
    AShr(BinaryOp),
}

/// A binary-operation instruction body (`<dest> = <op> <ty> <operand0>, <operand1>`). Mirrors the
/// field names of `llvm_ir`'s binop instruction structs (`Add`, `Sub`, …).
#[derive(PartialEq, Clone, Debug)]
pub struct BinaryOp {
    pub operand0: Operand,
    pub operand1: Operand,
    pub dest: Name,
    pub debugloc: Option<DebugLoc>,
}

impl Instruction {
    /// The destination name this instruction defines, if any (a `void`-typed effectful instruction has
    /// none). Mirrors `llvm_ir::instruction::Instruction::try_get_result`.
    pub fn try_get_result(&self) -> Option<&Name> {
        match self {
            Instruction::Add(b)
            | Instruction::Sub(b)
            | Instruction::Mul(b)
            | Instruction::UDiv(b)
            | Instruction::SDiv(b)
            | Instruction::URem(b)
            | Instruction::SRem(b)
            | Instruction::And(b)
            | Instruction::Or(b)
            | Instruction::Xor(b)
            | Instruction::Shl(b)
            | Instruction::LShr(b)
            | Instruction::AShr(b) => Some(&b.dest),
        }
    }
}

/// A block terminator. Mirrors `llvm_ir::terminator::Terminator` (seed subset; grows under parity).
#[derive(PartialEq, Clone, Debug)]
pub enum Terminator {
    Ret(Ret),
    Br(Br),
    CondBr(CondBr),
    Unreachable(Unreachable),
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
pub struct Unreachable {
    pub debugloc: Option<DebugLoc>,
}
