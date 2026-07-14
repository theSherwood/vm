// Recursive-descent arithmetic evaluator — the shape of expression/formula/query parsers (calculators,
// spreadsheets, filters). Parses+evaluates a fixed expression `n` times: recursion, branchy byte
// scanning, precedence climbing. No heap (the parse is the work).
static EXPR: &[u8] = b"((1+2)*3-4/2+5*(6-1))*2+7*8-9+((10+11)*2-3)*4/2+((100-50)/5+3)*2";

struct P {
    i: usize,
}
impl P {
    #[inline(always)]
    fn peek(&self) -> u8 {
        if self.i < EXPR.len() {
            EXPR[self.i]
        } else {
            0
        }
    }
    fn factor(&mut self) -> i64 {
        if self.peek() == b'(' {
            self.i += 1;
            let v = self.expr();
            self.i += 1; // ')'
            return v;
        }
        let mut v = 0i64;
        while self.peek().is_ascii_digit() {
            v = v * 10 + (self.peek() - b'0') as i64;
            self.i += 1;
        }
        v
    }
    fn term(&mut self) -> i64 {
        let mut v = self.factor();
        loop {
            match self.peek() {
                b'*' => {
                    self.i += 1;
                    v = v.wrapping_mul(self.factor());
                }
                b'/' => {
                    self.i += 1;
                    let d = self.factor();
                    v /= d;
                }
                _ => return v,
            }
        }
    }
    fn expr(&mut self) -> i64 {
        let mut v = self.term();
        loop {
            match self.peek() {
                b'+' => {
                    self.i += 1;
                    v = v.wrapping_add(self.term());
                }
                b'-' => {
                    self.i += 1;
                    v = v.wrapping_sub(self.term());
                }
                _ => return v,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut h = 0i64;
    for _ in 0..n {
        let mut p = P { i: 0 };
        h = h.wrapping_add(p.expr());
    }
    h
}
