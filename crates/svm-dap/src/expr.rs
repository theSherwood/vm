//! A tiny **C-expression evaluator** for DAP `evaluate` and conditional breakpoints
//! (DEBUGGING.md W5). It supports integer (`42`, `0x2a`) and floating (`1.5`, `2e3`) literals,
//! variable names, parentheses, unary `- ! ~`, the usual C arithmetic / bitwise / comparison /
//! logical binary operators with C precedence and **float promotion**, **short-circuit** `&&`/`||`,
//! and **postfix member / index / arrow access** (`a.b`, `arr[i]`, `p->x`) over the §6 structured
//! `TypeRef` layout. Comparisons and logical ops yield `0`/`1`.
//!
//! The evaluator is frontend-agnostic: it knows only [`Value`] (an integer, a float, or an opaque
//! typed [`Place`](Value::Place) = window address + `TypeId`) and calls back into a [`Resolver`]
//! for the semantics (name lookup, member/index/deref, scalar load). The caller (`svm-dap`)
//! supplies the resolver that reads the interpreter's window through the neutral debug info — so
//! this module has no dependency on `svm-ir`/`svm-interp` or any frontend.

/// An evaluated value: an integer, a float, or a typed location (`Place`) the resolver can navigate
/// / read. `type_id` indexes the caller's structured type table (opaque to this module).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Place { addr: u64, type_id: u32 },
}

/// The semantic callbacks the evaluator needs — implemented by the caller against its debug info +
/// memory. Navigation returns `None` for a type mismatch (e.g. `.x` on a non-struct) or a missing
/// member, which fails the whole expression cleanly.
pub trait Resolver {
    /// A bare identifier → its value (a `Place` for a window var, an `Int`/`Float` for a promoted
    /// scalar).
    fn ident(&mut self, name: &str) -> Option<Value>;
    /// `base.name` — a struct/union member.
    fn member(&mut self, base: &Value, name: &str) -> Option<Value>;
    /// `base[index]` — an array element or pointer offset.
    fn index(&mut self, base: &Value, index: i64) -> Option<Value>;
    /// `*base` — dereference a pointer (used to desugar `base->name`).
    fn deref(&mut self, base: &Value) -> Option<Value>;
    /// Resolve a value the operators can compute on: a scalar `Place` is read to an `Int`/`Float`;
    /// an `Int`/`Float` passes through; an aggregate/array `Place` ⇒ `None`.
    fn load(&mut self, v: &Value) -> Option<Value>;
}

/// Evaluate `expr` to a [`Value`] against `resolver`. `None` on a parse error, an unknown name, a
/// type mismatch, or integer division by zero.
pub fn eval(expr: &str, resolver: &mut dyn Resolver) -> Option<Value> {
    let tokens = tokenize(expr)?;
    let mut p = Parser {
        tokens: &tokens,
        pos: 0,
        resolver,
    };
    let v = p.parse(0, true)?;
    (p.pos == p.tokens.len()).then_some(v)
}

/// Evaluate `expr` to an `i64`, resolving identifiers through `resolve` — the scalar path for
/// conditional breakpoints (and existing callers). Member/index access is unavailable here (no type
/// info); use [`eval`] with a full [`Resolver`] for that. A float result is truncated to `i64`.
pub fn eval_int(expr: &str, resolve: &dyn Fn(&str) -> Option<i64>) -> Option<i64> {
    struct Simple<'a>(&'a dyn Fn(&str) -> Option<i64>);
    impl Resolver for Simple<'_> {
        fn ident(&mut self, name: &str) -> Option<Value> {
            (self.0)(name).map(Value::Int)
        }
        fn member(&mut self, _: &Value, _: &str) -> Option<Value> {
            None
        }
        fn index(&mut self, _: &Value, _: i64) -> Option<Value> {
            None
        }
        fn deref(&mut self, _: &Value) -> Option<Value> {
            None
        }
        fn load(&mut self, v: &Value) -> Option<Value> {
            match v {
                Value::Int(_) | Value::Float(_) => Some(*v),
                Value::Place { .. } => None,
            }
        }
    }
    match eval(expr, &mut Simple(resolve))? {
        Value::Int(n) => Some(n),
        Value::Float(x) => Some(x as i64),
        Value::Place { .. } => None,
    }
}

/// A number for the arithmetic operators — the int/float split C promotion works over.
#[derive(Clone, Copy)]
enum Num {
    I(i64),
    F(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::I(n) => n as f64,
            Num::F(x) => x,
        }
    }
    fn truthy(self) -> bool {
        match self {
            Num::I(n) => n != 0,
            Num::F(x) => x != 0.0,
        }
    }
}

#[derive(PartialEq)]
enum Tok {
    Num(i64),
    Real(f64),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
    Dot,
    Arrow,
    LBracket,
    RBracket,
}

fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        match b[i] {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b'[' => {
                out.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                out.push(Tok::RBracket);
                i += 1;
            }
            b'.' => {
                out.push(Tok::Dot);
                i += 1;
            }
            // `->` (arrow) before the `-` operator, so it doesn't tokenize as minus-then-greater.
            b'-' if i + 1 < b.len() && b[i + 1] == b'>' => {
                out.push(Tok::Arrow);
                i += 2;
            }
            b'0' if i + 1 < b.len() && (b[i + 1] == b'x' || b[i + 1] == b'X') => {
                let start = i + 2;
                i = start;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
                let n = i64::from_str_radix(std::str::from_utf8(&b[start..i]).ok()?, 16).ok()?;
                out.push(Tok::Num(n));
            }
            b'0'..=b'9' => {
                let start = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                // A `.` or exponent after the digits makes it a float literal. (`.` here is always
                // fractional — you can't member-access an integer literal.)
                let mut is_float = false;
                if i < b.len() && b[i] == b'.' {
                    is_float = true;
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    is_float = true;
                    i += 1;
                    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                        i += 1;
                    }
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let lit = std::str::from_utf8(&b[start..i]).ok()?;
                if is_float {
                    out.push(Tok::Real(lit.parse().ok()?));
                } else {
                    out.push(Tok::Num(lit.parse().ok()?));
                }
            }
            c if c == b'_' || c.is_ascii_alphabetic() => {
                let start = i;
                while i < b.len() && (b[i] == b'_' || b[i].is_ascii_alphanumeric()) {
                    i += 1;
                }
                out.push(Tok::Ident(
                    std::str::from_utf8(&b[start..i]).ok()?.to_string(),
                ));
            }
            _ => {
                // Operators, longest match first.
                let two = b.get(i..i + 2).and_then(|s| std::str::from_utf8(s).ok());
                let op = match two {
                    Some(t @ ("==" | "!=" | "<=" | ">=" | "&&" | "||" | "<<" | ">>")) => {
                        i += 2;
                        static_op(t)
                    }
                    _ => {
                        let one = std::str::from_utf8(&b[i..i + 1]).ok()?;
                        match one {
                            "+" | "-" | "*" | "/" | "%" | "<" | ">" | "&" | "|" | "^" | "~"
                            | "!" => {
                                i += 1;
                                static_op(one)
                            }
                            _ => return None, // unknown character
                        }
                    }
                };
                out.push(Tok::Op(op));
            }
        }
    }
    Some(out)
}

/// Intern an operator lexeme as a `&'static str` (it is always one of the known operators).
fn static_op(op: &str) -> &'static str {
    match op {
        "==" => "==",
        "!=" => "!=",
        "<=" => "<=",
        ">=" => ">=",
        "&&" => "&&",
        "||" => "||",
        "<<" => "<<",
        ">>" => ">>",
        "+" => "+",
        "-" => "-",
        "*" => "*",
        "/" => "/",
        "%" => "%",
        "<" => "<",
        ">" => ">",
        "&" => "&",
        "|" => "|",
        "^" => "^",
        "~" => "~",
        "!" => "!",
        _ => unreachable!("static_op called with a non-operator lexeme"),
    }
}

struct Parser<'a> {
    tokens: &'a [Tok],
    pos: usize,
    resolver: &'a mut dyn Resolver,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    /// Resolve a value to a [`Num`] for the operators. When `!live` (a short-circuited branch),
    /// returns a placeholder without touching the resolver, so the dead branch can't error.
    fn num_of(&mut self, v: &Value, live: bool) -> Option<Num> {
        if !live {
            return Some(Num::I(0));
        }
        match self.resolver.load(v)? {
            Value::Int(n) => Some(Num::I(n)),
            Value::Float(x) => Some(Num::F(x)),
            Value::Place { .. } => None,
        }
    }

    /// Resolve a value to an `i64` (for array indices and shift counts); a float is truncated.
    fn int_of(&mut self, v: &Value, live: bool) -> Option<i64> {
        Some(match self.num_of(v, live)? {
            Num::I(n) => n,
            Num::F(x) => x as i64,
        })
    }

    /// Precedence-climbing parse. Operands flow as [`Value`]s (so a postfix chain can stay a
    /// `Place`); binary ops coerce via [`Num`] with C float promotion. `&&`/`||` short-circuit by
    /// parsing the dead operand with `live = false` (consumed for position, never evaluated).
    fn parse(&mut self, min_prec: u8, live: bool) -> Option<Value> {
        let mut lhs = self.unary(live)?;
        while let Some(Tok::Op(op)) = self.peek() {
            let Some(prec) = prec_of(op) else { break };
            if prec < min_prec {
                break;
            }
            let op = *op;
            self.pos += 1;
            if op == "&&" || op == "||" {
                let lt = self.num_of(&lhs, live)?.truthy();
                // Only evaluate the rhs if the result isn't already decided.
                let rhs_live = live && if op == "&&" { lt } else { !lt };
                let rhs = self.parse(prec + 1, rhs_live)?;
                lhs = if !live {
                    Value::Int(0)
                } else if op == "&&" {
                    Value::Int((lt && self.num_of(&rhs, true)?.truthy()) as i64)
                } else {
                    Value::Int((lt || self.num_of(&rhs, true)?.truthy()) as i64)
                };
            } else {
                let rhs = self.parse(prec + 1, live)?; // left-associative
                if live {
                    let l = self.num_of(&lhs, true)?;
                    let r = self.num_of(&rhs, true)?;
                    lhs = apply(op, l, r)?;
                } else {
                    lhs = Value::Int(0);
                }
            }
        }
        Some(lhs)
    }

    fn unary(&mut self, live: bool) -> Option<Value> {
        if let Some(Tok::Op(op)) = self.peek() {
            if matches!(*op, "-" | "!" | "~") {
                let op = *op;
                self.pos += 1;
                let v = self.unary(live)?;
                if !live {
                    return Some(Value::Int(0));
                }
                let n = self.num_of(&v, true)?;
                return Some(match op {
                    "-" => match n {
                        Num::I(i) => Value::Int(i.wrapping_neg()),
                        Num::F(x) => Value::Float(-x),
                    },
                    "!" => Value::Int(!n.truthy() as i64),
                    // `~` is integer-only.
                    _ => match n {
                        Num::I(i) => Value::Int(!i),
                        Num::F(_) => return None,
                    },
                });
            }
        }
        self.postfix(live)
    }

    /// A primary followed by zero or more postfix accessors: `.field`, `->field`, `[index]`. In a
    /// dead (`!live`) branch the accessors are consumed but not resolved.
    fn postfix(&mut self, live: bool) -> Option<Value> {
        let mut v = self.primary(live)?;
        loop {
            match self.peek() {
                Some(Tok::Dot) => {
                    self.pos += 1;
                    let name = self.ident_tok()?;
                    v = if live {
                        self.resolver.member(&v, &name)?
                    } else {
                        Value::Int(0)
                    };
                }
                Some(Tok::Arrow) => {
                    self.pos += 1;
                    let name = self.ident_tok()?;
                    v = if live {
                        let target = self.resolver.deref(&v)?; // base->name == (*base).name
                        self.resolver.member(&target, &name)?
                    } else {
                        Value::Int(0)
                    };
                }
                Some(Tok::LBracket) => {
                    self.pos += 1;
                    let idx = self.parse(0, live)?;
                    let idx = self.int_of(&idx, live)?;
                    matches!(self.peek(), Some(Tok::RBracket)).then_some(())?;
                    self.pos += 1;
                    v = if live {
                        self.resolver.index(&v, idx)?
                    } else {
                        Value::Int(0)
                    };
                }
                _ => break,
            }
        }
        Some(v)
    }

    /// Consume an identifier token, returning its text (for `.field` / `->field`).
    fn ident_tok(&mut self) -> Option<String> {
        match self.peek()? {
            Tok::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                Some(name)
            }
            _ => None,
        }
    }

    fn primary(&mut self, live: bool) -> Option<Value> {
        match self.peek()? {
            Tok::Num(n) => {
                let n = *n;
                self.pos += 1;
                Some(Value::Int(n))
            }
            Tok::Real(x) => {
                let x = *x;
                self.pos += 1;
                Some(Value::Float(x))
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                // A name in a dead branch isn't resolved (it need not exist — C short-circuit).
                if live {
                    self.resolver.ident(&name)
                } else {
                    Some(Value::Int(0))
                }
            }
            Tok::LParen => {
                self.pos += 1;
                let v = self.parse(0, live)?;
                matches!(self.peek(), Some(Tok::RParen)).then_some(())?;
                self.pos += 1;
                Some(v)
            }
            _ => None,
        }
    }
}

/// Binary-operator precedence (higher binds tighter), C-like. `None` ⇒ not a binary operator.
fn prec_of(op: &str) -> Option<u8> {
    Some(match op {
        "||" => 1,
        "&&" => 2,
        "|" => 3,
        "^" => 4,
        "&" => 5,
        "==" | "!=" => 6,
        "<" | "<=" | ">" | ">=" => 7,
        "<<" | ">>" => 8,
        "+" | "-" => 9,
        "*" | "/" | "%" => 10,
        _ => return None,
    })
}

/// Apply a binary operator with C numeric promotion: if either side is a float, arithmetic and
/// comparisons run in `f64`; the bitwise/shift/`%` operators are integer-only (a float operand ⇒
/// `None`). Comparisons yield an `Int` `0`/`1`.
fn apply(op: &str, l: Num, r: Num) -> Option<Value> {
    // Integer-only operators.
    if matches!(op, "%" | "<<" | ">>" | "&" | "|" | "^") {
        let (Num::I(l), Num::I(r)) = (l, r) else {
            return None;
        };
        return Some(Value::Int(match op {
            "%" => (r != 0).then(|| l.wrapping_rem(r))?,
            "<<" => l.wrapping_shl(r as u32),
            ">>" => l.wrapping_shr(r as u32),
            "&" => l & r,
            "|" => l | r,
            "^" => l ^ r,
            _ => unreachable!(),
        }));
    }
    // Integer arithmetic/comparison when both sides are integers.
    if let (Num::I(l), Num::I(r)) = (l, r) {
        return Some(match op {
            "+" => Value::Int(l.wrapping_add(r)),
            "-" => Value::Int(l.wrapping_sub(r)),
            "*" => Value::Int(l.wrapping_mul(r)),
            "/" => Value::Int((r != 0).then(|| l.wrapping_div(r))?),
            "==" => Value::Int((l == r) as i64),
            "!=" => Value::Int((l != r) as i64),
            "<" => Value::Int((l < r) as i64),
            "<=" => Value::Int((l <= r) as i64),
            ">" => Value::Int((l > r) as i64),
            ">=" => Value::Int((l >= r) as i64),
            _ => return None,
        });
    }
    // Otherwise promote to f64 (division by zero yields inf/NaN, as in C).
    let (l, r) = (l.as_f64(), r.as_f64());
    Some(match op {
        "+" => Value::Float(l + r),
        "-" => Value::Float(l - r),
        "*" => Value::Float(l * r),
        "/" => Value::Float(l / r),
        "==" => Value::Int((l == r) as i64),
        "!=" => Value::Int((l != r) as i64),
        "<" => Value::Int((l < r) as i64),
        "<=" => Value::Int((l <= r) as i64),
        ">" => Value::Int((l > r) as i64),
        ">=" => Value::Int((l >= r) as i64),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{eval, eval_int, Resolver, Value};

    fn ev(e: &str) -> Option<i64> {
        eval_int(e, &|name| match name {
            "i" => Some(3),
            "n" => Some(10),
            _ => None,
        })
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(ev("1 + 2 * 3"), Some(7));
        assert_eq!(ev("(1 + 2) * 3"), Some(9));
        assert_eq!(ev("-5 + 0x10"), Some(11));
        assert_eq!(ev("10 % 3"), Some(1));
        assert_eq!(ev("1 << 4"), Some(16));
    }

    #[test]
    fn variables_and_comparisons() {
        assert_eq!(ev("i"), Some(3));
        assert_eq!(ev("i == 3"), Some(1));
        assert_eq!(ev("i > n"), Some(0));
        assert_eq!(ev("i < n && n > 0"), Some(1));
        assert_eq!(ev("i * 2 + 1"), Some(7));
        assert_eq!(ev("!(i == 3)"), Some(0));
    }

    #[test]
    fn errors_are_none() {
        assert_eq!(ev("nope"), None); // unknown variable
        assert_eq!(ev("1 +"), None); // parse error
        assert_eq!(ev("1 / 0"), None); // division by zero
        assert_eq!(ev("1 @ 2"), None); // bad token
    }

    #[test]
    fn short_circuit_skips_the_dead_branch() {
        // The dead operand isn't evaluated, so its error (div-by-zero) / unknown name is suppressed.
        assert_eq!(ev("i != 3 && 1 / 0"), Some(0)); // lhs false ⇒ rhs `1/0` skipped
        assert_eq!(ev("i == 3 || 1 / 0"), Some(1)); // lhs true ⇒ rhs `1/0` skipped
        assert_eq!(ev("0 && nope"), Some(0)); // unknown name in the dead branch is fine
        assert_eq!(ev("1 || nope"), Some(1));
        // The live branch still propagates errors.
        assert_eq!(ev("i == 3 && 1 / 0"), None);
    }

    // A resolver with a float and an int variable (no places), for the float-promotion tests.
    struct Vars;
    impl Resolver for Vars {
        fn ident(&mut self, name: &str) -> Option<Value> {
            match name {
                "f" => Some(Value::Float(2.5)),
                "i" => Some(Value::Int(3)),
                _ => None,
            }
        }
        fn member(&mut self, _: &Value, _: &str) -> Option<Value> {
            None
        }
        fn index(&mut self, _: &Value, _: i64) -> Option<Value> {
            None
        }
        fn deref(&mut self, _: &Value) -> Option<Value> {
            None
        }
        fn load(&mut self, v: &Value) -> Option<Value> {
            match v {
                Value::Int(_) | Value::Float(_) => Some(*v),
                Value::Place { .. } => None,
            }
        }
    }

    fn evf(e: &str) -> Option<Value> {
        eval(e, &mut Vars)
    }

    #[test]
    fn float_literals_and_promotion() {
        assert!(matches!(evf("1.5 + 2"), Some(Value::Float(x)) if x == 3.5));
        assert!(matches!(evf("f * 2"), Some(Value::Float(x)) if x == 5.0)); // int promotes to float
        assert!(matches!(evf("f + i"), Some(Value::Float(x)) if x == 5.5));
        assert_eq!(evf("f > i"), Some(Value::Int(0))); // 2.5 > 3 ⇒ 0
        assert_eq!(evf("f < i"), Some(Value::Int(1)));
        assert!(matches!(evf("2e3"), Some(Value::Float(x)) if x == 2000.0));
        // Bitwise/`~` on a float is rejected (integer-only operators).
        assert_eq!(evf("f & 1"), None);
        assert_eq!(evf("~f"), None);
        // Integer arithmetic stays integer.
        assert_eq!(evf("i + 1"), Some(Value::Int(4)));
    }
}
