// Bit intrinsics: count_ones/leading_zeros/trailing_zeros (llvm.ctpop/ctlz/cttz),
// rotate_left/right (llvm.fsh{l,r}), swap_bytes (llvm.bswap), reverse_bits (llvm.bitreverse).
// Rust emits bitreverse where clang almost never does.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut acc = 0i64;
    let mut s = 0x0123_4567_89ab_cdefu64;
    for _ in 0..(n as u64 * 6 + 4) {
        let x = xs(&mut s);
        acc = acc.wrapping_add(x.count_ones() as i64);
        acc = acc.wrapping_add(x.leading_zeros() as i64);
        acc = acc.wrapping_add(x.trailing_zeros() as i64);
        acc ^= x.rotate_left(13).wrapping_mul(0x9e37_79b9_7f4a_7c15) as i64;
        acc ^= x.rotate_right(29) as i64;
        acc ^= x.swap_bytes() as i64;
        acc ^= x.reverse_bits() as i64;
        acc = acc.wrapping_add((x as u32).count_ones() as i64);
        acc ^= (x as u16).reverse_bits() as i64;
    }
    acc
}
