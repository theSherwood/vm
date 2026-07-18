// Overflow-checked / saturating / wrapping arithmetic — emits llvm.{s,u}{add,sub,mul}.with.overflow
// and llvm.{s,u}{add,sub}.sat intrinsics that clang C rarely produces but Rust does routinely.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut acc = 0i64;
    let mut s = 0xdead_beef_0000_0001u64;
    for i in 0..(n as u64 * 8 + 5) {
        let a = xs(&mut s) as i32;
        let b = (i as i32).wrapping_mul(1103515245).wrapping_add(12345);
        acc = acc.wrapping_add(a.checked_add(b).unwrap_or(-1) as i64);
        acc = acc.wrapping_add(a.saturating_mul(b) as i64);
        acc = acc.wrapping_add(a.saturating_sub(b) as i64);
        acc = acc.wrapping_add((a as u32).saturating_add(b as u32) as i64);
        let (w, o) = a.overflowing_add(b);
        acc = acc.wrapping_add(w as i64).wrapping_add(o as i64);
        acc ^= (a as i64).rotate_left((i & 63) as u32);
    }
    acc
}
