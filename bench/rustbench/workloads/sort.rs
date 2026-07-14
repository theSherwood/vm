// Batch insertion-sort — the shape of sorting many small buffers (log lines, records, keys), which
// real programs do constantly. Each of `n` iterations fills a fixed-size buffer with PRNG values,
// insertion-sorts it, and folds the sorted middle element into a checksum. Comparison- and
// memory-move-heavy, branch-unpredictable.
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    const K: usize = 48;
    let mut buf: Vec<i64> = vec![0i64; K];
    let mut h: i64 = 0;
    let mut st: u64 = 0x243f6a8885a308d3;
    for _ in 0..n {
        for slot in buf.iter_mut() {
            *slot = (xs(&mut st) % 1_000_000) as i64;
        }
        // insertion sort
        for i in 1..K {
            let v = buf[i];
            let mut j = i;
            while j > 0 && buf[j - 1] > v {
                buf[j] = buf[j - 1];
                j -= 1;
            }
            buf[j] = v;
        }
        h = h.wrapping_add(buf[K / 2]);
    }
    h
}
