// A long straight arithmetic loop over Vec<i32>/Vec<i64> that LLVM auto-vectorizes to WIDE vectors
// (<8 x i32>, <4 x i64>) with a scalar remainder — stresses per-lane vector int ops, wide shifts, and
// the mask/movemask idioms on rustc-shaped IR.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let len = (n as usize) * 16 + 13;
    let mut a: Vec<i32> = (0..len as i32).map(|i| i.wrapping_mul(2654435761u32 as i32)).collect();
    let b: Vec<i32> = (0..len as i32).map(|i| (i ^ 0x5a5a).wrapping_sub(len as i32)).collect();
    // elementwise fused ops (vectorized): a = (a*3 + b) ^ (b >> 2), then mask-count of a>0
    for i in 0..len {
        a[i] = a[i].wrapping_mul(3).wrapping_add(b[i]) ^ (b[i] >> 2);
    }
    let pos = a.iter().filter(|&&x| x > 0).count() as i64;
    let sum: i64 = a.iter().map(|&x| x as i64).sum();
    let xored = a.iter().fold(0i32, |h, &x| h ^ x) as i64;
    let shifted: i64 = a.iter().zip(&b).map(|(&x, &y)| ((x as i64) << (y as u32 & 31)) as i32 as i64).sum();
    sum.wrapping_mul(31)
        .wrapping_add(pos.wrapping_mul(1009))
        .wrapping_add(xored)
        .wrapping_add(shifted)
}
