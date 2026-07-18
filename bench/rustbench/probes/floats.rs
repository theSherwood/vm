// Floating point without libm (no_std): arithmetic, comparison-based min/max (llvm.{min,max}num via
// f64::max/min are libm-free), int<->float conversions (sitofp/fptosi, and Rust's saturating
// fptosi.sat for `as`), and f32<->f64 (fpext/fptrunc). Returns a bit-hash so formatting is out of it.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut acc = 0u64;
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let cnt = (n as u64) * 5 + 4;
    for i in 0..cnt {
        let x = ((xs(&mut s) >> 11) as f64) / (1u64 << 53) as f64 * 200.0 - 100.0;
        let y = (i as f64) * 0.5 - 3.0;
        let r = x * y + x / (y + 128.0) - x;
        let m = x.max(y) - x.min(y); // fmax/fmin, no libm
        acc ^= r.to_bits();
        acc = acc.wrapping_add(m.to_bits());
        // int<->float round trips, incl. out-of-range `as` (saturating in Rust: fptosi.sat)
        acc = acc.wrapping_add((x as i64) as u64);
        acc ^= ((y * 1e18) as i32) as u64;
        acc = acc.wrapping_add((x as u32) as u64);
        acc = acc.wrapping_add(((x as f32) + (i as f32)).to_bits() as u64);
        acc ^= ((x as f32) as f64).to_bits();
    }
    acc as i64
}
