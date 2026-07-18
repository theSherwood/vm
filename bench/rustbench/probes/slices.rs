// Slice iterator adapters — windows/chunks/enumerate/zip/rev/step_by — the opaque-pointer GEP
// machinery (the I23 #1 class: a source-element-type stride the on-ramp must honor). Also copy_within
// and rotate_left (memmove).
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let len = (n as usize) * 3 + 8;
    let mut v: Vec<i64> = (0..len as i64).map(|i| i * i - 7 * i + 1).collect();
    let mut acc = 0i64;
    for w in v.windows(3) {
        acc = acc.wrapping_add(w[0].wrapping_mul(w[1]).wrapping_sub(w[2]));
    }
    for (i, c) in v.chunks(4).enumerate() {
        acc ^= (i as i64).wrapping_add(c.iter().copied().sum::<i64>());
    }
    for (a, b) in v.iter().zip(v.iter().rev()) {
        acc = acc.wrapping_add(a.wrapping_mul(*b));
    }
    for x in v.iter().step_by(3) {
        acc ^= *x;
    }
    if v.len() >= 6 {
        v.copy_within(0..4, 2);
        v.rotate_left(3);
        acc = acc.wrapping_add(v.iter().sum::<i64>());
    }
    acc
}
