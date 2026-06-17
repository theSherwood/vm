//! A tiny **integer-expression evaluator** for DAP `evaluate` and conditional breakpoints
//! (DEBUGGING.md W5). It supports integer literals (decimal / `0x` hex), variable names (resolved by
//! the caller against the stopped frame), parentheses, unary `- ! ~`, and the usual C
//! arithmetic / bitwise / comparison / logical binary operators with C precedence. Values are `i64`;
//! comparisons and logical ops yield `0`/`1`.
//!
//! Deliberately scalar-only: member / index access (`a.b`, `arr[i]`) needs **structured type
//! layout** (field offsets, element strides) that the debug-info ABI does not carry yet (the W4
//! "structured `TypeRef`" gap, DEBUGGING.md §6) — a later slice. Floats and short-circuit `&&`/`||`
//! are also follow-ups (short-circuit is moot here: operands are side-effect-free).

/// Evaluate `expr` to an `i64`, resolving identifiers through `resolve`. `None` on a parse error, an
/// unknown/unresolvable name, or division by zero.
pub fn eval_int(expr: &str, resolve: &dyn Fn(&str) -> Option<i64>) -> Option<i64> {
    let tokens = tokenize(expr)?;
    let mut p = Parser {
        tokens: &tokens,
        pos: 0,
        resolve,
    };
    let v = p.parse(0)?;
    (p.pos == p.tokens.len()).then_some(v)
}

#[derive(PartialEq)]
enum Tok {
    Num(i64),
    Ident(String),
    Op(&'static str),
    LParen,
    RParen,
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
    resolve: &'a dyn Fn(&str) -> Option<i64>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    /// Precedence-climbing: parse a (sub)expression whose binary operators bind at least `min_prec`.
    fn parse(&mut self, min_prec: u8) -> Option<i64> {
        let mut lhs = self.unary()?;
        while let Some(Tok::Op(op)) = self.peek() {
            let Some(prec) = prec_of(op) else { break };
            if prec < min_prec {
                break;
            }
            let op = *op;
            self.pos += 1;
            let rhs = self.parse(prec + 1)?; // left-associative
            lhs = apply(op, lhs, rhs)?;
        }
        Some(lhs)
    }

    fn unary(&mut self) -> Option<i64> {
        if let Some(Tok::Op(op)) = self.peek() {
            if matches!(*op, "-" | "!" | "~") {
                let op = *op;
                self.pos += 1;
                let v = self.unary()?;
                return Some(match op {
                    "-" => v.wrapping_neg(),
                    "!" => (v == 0) as i64,
                    _ => !v,
                });
            }
        }
        self.primary()
    }

    fn primary(&mut self) -> Option<i64> {
        match self.peek()? {
            Tok::Num(n) => {
                let n = *n;
                self.pos += 1;
                Some(n)
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                (self.resolve)(&name)
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
