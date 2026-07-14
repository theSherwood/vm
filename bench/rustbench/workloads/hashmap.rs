// Open-addressing hash-table churn — the shape of a real dedup/counting/cache hot loop: PRNG-driven
// keys, linear-probe insert-or-increment, then a lookup, accumulating a checksum. Control-flow and
// memory heavy, not vectorizable. Key space (4000) is kept well under the table (8192 slots) so the
// probe never fills the table.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    const CAP: usize = 8192;
    let mut keys = vec![0i64; CAP];
    let mut vals = vec![0i64; CAP];
    let mut occ = vec![0u8; CAP];
    let mut h: i64 = 0;
    let mut st: u64 = 0x9e3779b97f4a7c15;
    for _ in 0..n {
        let k = (xs(&mut st) % 4000) as i64 + 1;
        let home = (k as u64 % CAP as u64) as usize;
        let mut idx = home;
        loop {
            if occ[idx] == 0 {
                occ[idx] = 1;
                keys[idx] = k;
                vals[idx] = 1;
                break;
            }
            if keys[idx] == k {
                vals[idx] += 1;
                break;
            }
            idx = (idx + 1) % CAP;
        }
        h = h.wrapping_add(vals[home]);
    }
    h
}
