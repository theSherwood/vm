// Auto-vectorized slice reductions with negatives — the I23 #2 class (a `<2 x i32>` / wider min/max
// the vectorizer emits from `if d > 0 { .. }` clamps and `.min()`/`.max()` folds). `n` sets the slice
// length so both the vectorized body and the scalar tail run.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let len = (n as usize) * 4 + 3;
    let mut v: Vec<i32> = Vec::with_capacity(len);
    let mut s = 0x1234_5678_9abc_def0u64;
    for _ in 0..len {
        // signed values straddling zero, so smax/smin clamps actually branch
        v.push((xs(&mut s) as i32) % 2000 - 1000);
    }
    let sum_pos: i64 = v.iter().map(|&d| if d > 0 { d as i64 } else { 0 }).sum();
    let clamped: i64 = v.iter().map(|&d| d.max(-5).min(5) as i64).sum();
    let mx = *v.iter().max().unwrap() as i64;
    let mn = *v.iter().min().unwrap() as i64;
    let absum: i64 = v.iter().map(|&d| (d as i64).abs()).sum();
    sum_pos
        .wrapping_mul(31)
        .wrapping_add(clamped.wrapping_mul(7))
        .wrapping_add(mx.wrapping_mul(13))
        .wrapping_add(mn.wrapping_mul(17))
        .wrapping_add(absum)
}
