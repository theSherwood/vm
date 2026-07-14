// Breadth-first search over a fixed grid graph — traversal, queue, pointer-chasing (the shape of
// pathfinding/reachability). Runs BFS from a corner `n` times over a fixed pseudo-random-walled grid,
// summing distances. The `dist`/queue buffers are allocated once and cleared per iteration (the bump
// allocator does not reclaim within a run).
use alloc::collections::VecDeque;

#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    reset_arena();
    const W: usize = 48;
    const N: usize = W * W;
    let mut wall = vec![false; N];
    let mut st: u64 = 0x1234567890abcdef;
    for w in wall.iter_mut() {
        *w = xs(&mut st) % 5 == 0;
    }
    wall[0] = false;
    let mut dist = vec![-1i32; N];
    let mut q: VecDeque<usize> = VecDeque::with_capacity(N);
    let mut h = 0i64;
    for _ in 0..n {
        for d in dist.iter_mut() {
            *d = -1;
        }
        q.clear();
        dist[0] = 0;
        q.push_back(0);
        while let Some(c) = q.pop_front() {
            let (x, y) = (c % W, c / W);
            let d = dist[c];
            let neigh = [
                if x + 1 < W { Some(y * W + x + 1) } else { None },
                if x > 0 { Some(y * W + x - 1) } else { None },
                if y + 1 < W { Some((y + 1) * W + x) } else { None },
                if y > 0 { Some((y - 1) * W + x) } else { None },
            ];
            for ni in neigh.into_iter().flatten() {
                if !wall[ni] && dist[ni] < 0 {
                    dist[ni] = d + 1;
                    q.push_back(ni);
                }
            }
        }
        for &d in dist.iter() {
            if d > 0 {
                h = h.wrapping_add(d as i64);
            }
        }
    }
    h
}
