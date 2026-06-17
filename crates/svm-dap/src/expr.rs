//! A tiny **C-expression evaluator** for DAP `evaluate` and conditional breakpoints
//! (DEBUGGING.md W5). It supports integer literals (decimal / `0x` hex), variable names, parentheses,
//! unary `- ! ~`, the usual C arithmetic / bitwise / comparison / logical binary operators with C
//! precedence, and **postfix member / index / arrow access** (`a.b`, `arr[i]`, `p->x`) over the
//! §6 structured `TypeRef` layout. Comparisons and logical ops yield `0`/`1`.
//!
//! The evaluator is frontend-agnostic: it knows only [`Value`] (an integer, or an opaque typed
//! [`Place`](Value::Place) = window address + `TypeId`) and calls back into a [`Resolver`] for the
//! semantics (name lookup, member/index/deref, integer coercion). The caller (`svm-dap`) supplies
//! the resolver that reads the interpreter's window through the neutral debug info — so this module
//! has no dependency on `svm-ir`/`svm-interp` or any frontend. Floats and short-circuit `&&`/`||`
//! are still follow-ups (short-circuit is moot here: operands are side-effect-free).

/// An evaluated value: a plain integer, or a typed location (`Place`) the resolver can navigate /
/// read. `type_id` indexes the caller's structured type table (opaque to this module).
#[derive(Clone, Copy, PartialEq)]
pub enum Value {
    Int(i64),
    Place { addr: u64, type_id: u32 },
}

/// The semantic callbacks the evaluator needs — implemented by the caller against its debug info +
/// memory. Navigation returns `None` for a type mismatch (e.g. `.x` on a non-struct) or a missing
/// member, which fails the whole expression cleanly.
pub trait Resolver {
    /// A bare identifier → its value (a `Place` for a window var, an `Int` for a promoted scalar).
    fn ident(&mut self, name: &str) -> Option<Value>;
    /// `base.name` — a struct/union member.
    fn member(&mut self, base: &Value, name: &str) -> Option<Value>;
    /// `base[index]` — an array element or pointer offset.
    fn index(&mut self, base: &Value, index: i64) -> Option<Value>;
    /// `*base` — dereference a pointer (used to desugar `base->name`).
    fn deref(&mut self, base: &Value) -> Option<Value>;
    /// Coerce a value to the integer the arithmetic operators work on (reading a scalar `Place`).
    fn coerce_int(&mut self, v: &Value) -> Option<i64>;
}

/// Evaluate `expr` to a [`Value`] against `resolver`. `None` on a parse error, an unknown name, a
/// type mismatch, or division by zero.
pub fn eval(expr: &str, resolver: &mut dyn Resolver) -> Option<Value> {
    let tokens = tokenize(expr)?;
    let mut p = Parser {
        tokens: &tokens,
        pos: 0,
        resolver,
    };
    let v = p.parse(0)?;
    (p.pos == p.tokens.len()).then_some(v)
}

/// Evaluate `expr` to an `i64`, resolving identifiers through `resolve` — the scalar-only path for
/// conditional breakpoints (and existing callers). Member/index access is unavailable here (no type
/// info); use [`eval`] with a full [`Resolver`] for that.
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
        fn coerce_int(&mut self, v: &Value) -> Option<i64> {
            match v {
                Value::Int(n) => Some(*n),
                Value::Place { .. } => None,
            }
        }
    }
    match eval(expr, &mut Simple(resolve))? {
        Value::Int(n) => Some(n),
        Value::Place { .. } => None,
    }
}

#[derive(PartialEq)]
enum Tok {
    Num(i64),
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
                out.push(Tok::Num(
                    std::str::from_utf8(&b[start..i]).ok()?.parse().ok()?,
                ));
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

    /// Coerce a value to the integer the operators work on (reads a scalar `Place`).
    fn int_of(&mut self, v: &Value) -> Option<i64> {
        self.resolver.coerce_int(v)
    }

    /// Precedence-climbing: parse a (sub)expression whose binary operators bind at least `min_prec`.
    /// Operands flow as [`Value`]s (so a postfix chain can stay a `Place`); each binary op coerces
    /// its sides to integers.
    fn parse(&mut self, min_prec: u8) -> Option<Value> {
        let mut lhs = self.unary()?;
        while let Some(Tok::Op(op)) = self.peek() {
            let Some(prec) = prec_of(op) else { break };
            if prec < min_prec {
                break;
            }
            let op = *op;
            self.pos += 1;
            let rhs = self.parse(prec + 1)?; // left-associative
            let l = self.int_of(&lhs)?;
            let r = self.int_of(&rhs)?;
            lhs = Value::Int(apply(op, l, r)?);
        }
        Some(lhs)
    }

    fn unary(&mut self) -> Option<Value> {
        if let Some(Tok::Op(op)) = self.peek() {
            if matches!(*op, "-" | "!" | "~") {
                let op = *op;
                self.pos += 1;
                let v = self.unary()?;
                let v = self.int_of(&v)?;
                return Some(Value::Int(match op {
                    "-" => v.wrapping_neg(),
                    "!" => (v == 0) as i64,
                    _ => !v,
                }));
            }
        }
        self.postfix()
    }

    /// A primary followed by zero or more postfix accessors: `.field`, `->field`, `[index]`.
    fn postfix(&mut self) -> Option<Value> {
        let mut v = self.primary()?;
        loop {
            match self.peek() {
                Some(Tok::Dot) => {
                    self.pos += 1;
                    let name = self.ident_tok()?;
                    v = self.resolver.member(&v, &name)?;
                }
                Some(Tok::Arrow) => {
                    self.pos += 1;
                    let name = self.ident_tok()?;
                    let target = self.resolver.deref(&v)?; // base->name == (*base).name
                    v = self.resolver.member(&target, &name)?;
                }
                Some(Tok::LBracket) => {
                    self.pos += 1;
                    let idx = self.parse(0)?;
                    let idx = self.int_of(&idx)?;
                    matches!(self.peek(), Some(Tok::RBracket)).then_some(())?;
                    self.pos += 1;
                    v = self.resolver.index(&v, idx)?;
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

    fn primary(&mut self) -> Option<Value> {
        match self.peek()? {
            Tok::Num(n) => {
                let n = *n;
                self.pos += 1;
                Some(Value::Int(n))
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                self.resolver.ident(&name)
            }
            Tok::LParen => {
                self.pos += 1;
                let v = self.parse(0)?;
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

fn apply(op: &str, l: i64, r: i64) -> Option<i64> {
    Some(match op {
        "+" => l.wrapping_add(r),
        "-" => l.wrapping_sub(r),
        "*" => l.wrapping_mul(r),
        "/" => (r != 0).then(|| l.wrapping_div(r))?,
        "%" => (r != 0).then(|| l.wrapping_rem(r))?,
        "<<" => l.wrapping_shl(r as u32),
        ">>" => l.wrapping_shr(r as u32),
        "&" => l & r,
        "|" => l | r,
        "^" => l ^ r,
        "==" => (l == r) as i64,
        "!=" => (l != r) as i64,
        "<" => (l < r) as i64,
        "<=" => (l <= r) as i64,
        ">" => (l > r) as i64,
        ">=" => (l >= r) as i64,
        "&&" => (l != 0 && r != 0) as i64,
        "||" => (l != 0 || r != 0) as i64,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::eval_int;

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
}
