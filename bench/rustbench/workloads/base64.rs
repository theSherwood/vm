// Base64 encode — byte/bit-manipulation over a buffer (data URIs, tokens, MIME): shifts, masks, table
// lookups. Encodes a fixed 96-byte buffer `n` times into a reused output and checksums it.
static TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    let mut input = vec![0u8; 96];
    let mut st: u64 = 0xcafef00dd15ea5e5;
    for (i, b) in input.iter_mut().enumerate() {
        *b = (xs(&mut st) as u8).wrapping_add(i as u8);
    }
    let mut out = vec![0u8; (input.len() / 3) * 4];
    let mut h = 0i64;
    for _ in 0..n {
        let mut o = 0;
        let mut k = 0;
        while k + 3 <= input.len() {
            let x = (input[k] as u32) << 16 | (input[k + 1] as u32) << 8 | input[k + 2] as u32;
            out[o] = TBL[(x >> 18 & 63) as usize];
            out[o + 1] = TBL[(x >> 12 & 63) as usize];
            out[o + 2] = TBL[(x >> 6 & 63) as usize];
            out[o + 3] = TBL[(x & 63) as usize];
            o += 4;
            k += 3;
        }
        for &c in out.iter() {
            h = h.wrapping_add(c as i64);
        }
    }
    h
}
