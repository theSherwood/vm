// Arrays of small structs + enums: nested field GEPs, discriminant loads, memcpy of aggregates, and
// sorting by key (moves/swaps of 24-byte structs). Enum-with-data `match` is the discriminant path.
#[derive(Clone, Copy)]
struct P {
    x: i32,
    y: i32,
    tag: u8,
}
enum Op {
    Add(i64),
    Scale(i32, i32),
    Neg,
}
fn apply(acc: i64, op: &Op) -> i64 {
    match op {
        Op::Add(v) => acc.wrapping_add(*v),
        Op::Scale(a, b) => acc.wrapping_mul(*a as i64).wrapping_add(*b as i64),
        Op::Neg => acc.wrapping_neg(),
    }
}
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let len = (n as usize) * 2 + 5;
    let mut s = 0xfeed_face_cafe_babeu64;
    let mut ps: Vec<P> = Vec::with_capacity(len);
    let mut ops: Vec<Op> = Vec::with_capacity(len);
    for i in 0..len {
        let r = xs(&mut s);
        ps.push(P { x: (r as i32) % 500 - 250, y: ((r >> 20) as i32) % 500 - 250, tag: (r & 3) as u8 });
        ops.push(match r % 3 {
            0 => Op::Add((r as i32 % 100) as i64),
            1 => Op::Scale((i as i32 % 7) + 1, (r >> 8) as i32 % 50),
            _ => Op::Neg,
        });
    }
    // manual insertion sort (the library sort pulls an undefined `panic_on_ord_violation` extern in
    // no_std) — still exercises 12-byte struct moves/copies and nested field GEPs
    for i in 1..ps.len() {
        let mut j = i;
        while j > 0 && (ps[j - 1].x, ps[j - 1].y, ps[j - 1].tag) > (ps[j].x, ps[j].y, ps[j].tag) {
            ps.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut acc = 0i64;
    for p in &ps {
        acc = acc.wrapping_add(p.x as i64).wrapping_mul(3).wrapping_add(p.y as i64) ^ (p.tag as i64);
    }
    for op in &ops {
        acc = apply(acc, op);
    }
    acc
}
