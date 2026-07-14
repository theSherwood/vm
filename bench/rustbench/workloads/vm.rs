// A tiny stack-based bytecode interpreter — the "real program" archetype (dispatch loop, operand
// stack, locals), the same shape as a scripting-language VM. `run(n)` executes a fixed bytecode
// program whose inner loop runs `n` times (summing `(i*i) % 97`), so the subtraction isolates the
// per-iteration interpreter dispatch cost. Branchy and unpredictable — exactly what JITs can't
// vectorize away.
#[derive(Clone, Copy)]
enum Op {
    PushC(i64),
    PushL(usize),
    StoreL(usize),
    Add,
    Mul,
    Mod,
    Lt,
    Jz(usize),
    Jmp(usize),
    Ret,
}

#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    use Op::*;
    // locals: [0]=n, [1]=i, [2]=s ; program: s=0; i=0; while i<n { s += (i*i)%97; i+=1 } ret s
    let prog = [
        PushL(1), PushL(0), Lt, Jz(17), // 0..3   while i<n
        PushL(1), PushL(1), Mul, PushC(97), Mod, // 4..8  (i*i)%97
        PushL(2), Add, StoreL(2), // 9..11  s += ...
        PushL(1), PushC(1), Add, StoreL(1), // 12..15 i += 1
        Jmp(0),   // 16
        PushL(2), // 17 push s
        Ret,      // 18
    ];
    let mut locals = [n, 0i64, 0i64];
    let mut stack: Vec<i64> = Vec::with_capacity(8);
    let mut pc = 0usize;
    loop {
        match prog[pc] {
            PushC(c) => stack.push(c),
            PushL(i) => stack.push(locals[i]),
            StoreL(i) => locals[i] = stack.pop().unwrap(),
            Add => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_add(b));
            }
            Mul => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a.wrapping_mul(b));
            }
            Mod => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push(a % b);
            }
            Lt => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                stack.push((a < b) as i64);
            }
            Jz(t) => {
                if stack.pop().unwrap() == 0 {
                    pc = t;
                    continue;
                }
            }
            Jmp(t) => {
                pc = t;
                continue;
            }
            Ret => return stack.pop().unwrap(),
        }
        pc += 1;
    }
}
