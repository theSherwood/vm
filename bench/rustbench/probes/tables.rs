// Static/const lookup tables + byte-string iteration — the constexpr-GEP class (I23 #1: a
// getelementptr into a `static` whose source element type the on-ramp must stride by, not the
// global's pointee type). Also `iter().position()`, `find`, and `&[u8]` indexing.
static LUT: [i32; 16] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
static MSG: &[u8] = b"the quick brown fox jumps over the lazy dog 0123456789";
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut acc = 0i64;
    let mut s = 0x1357_9bdf_2468_ace0u64;
    for _ in 0..(n as u64 * 4 + 6) {
        let r = xs(&mut s);
        let idx = (r & 15) as usize;
        acc = acc.wrapping_add(LUT[idx] as i64);
        acc ^= LUT[(idx + 7) & 15] as i64 * LUT[(idx * 3) & 15] as i64;
        let bi = (r as usize) % MSG.len();
        acc = acc.wrapping_add(MSG[bi] as i64);
    }
    // byte-string scans: computed offsets over a static slice
    acc = acc.wrapping_add(MSG.iter().filter(|&&b| b == b'o').count() as i64 * 100);
    acc = acc.wrapping_add(MSG.iter().position(|&b| b == b'z').map_or(-1, |p| p as i64));
    let digits: i64 = MSG.iter().filter(|b| b.is_ascii_digit()).map(|&b| (b - b'0') as i64).sum();
    acc = acc.wrapping_add(digits.wrapping_mul(31));
    // a rolling hash over the message, indexed both forward and reverse
    for (i, &b) in MSG.iter().enumerate() {
        acc = acc.wrapping_mul(131).wrapping_add(b as i64) ^ (i as i64);
    }
    acc
}
