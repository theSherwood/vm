//! Doom **headless frame-hash differential** (slice 3c) — the §18 oracle for the full renderer.
//!
//! Doom's renderer is pure fixed-point, so given the same WAD + a deterministic clock + input, the
//! guest (translated through the LLVM on-ramp) and a native `cc` build produce byte-identical
//! framebuffers. `doom_diff.c` makes that observable: its `DG_DrawFrame` prints an FNV hash of each
//! `DG_ScreenBuffer`, so the whole run's stdout is a frame-hash stream. This test runs the translated
//! guest over an in-memory WAD and asserts its frame hashes equal the native build's, byte-for-byte.
//!
//! `#[ignore]`d — it needs build artifacts produced by `crates/svm-run/demos/doom/{fetch,build,diff}.sh`
//! (the guest `.svmb`, the native oracle's frame list, and the shareware WAD). Paths are overridable
//! via `DOOM_SVMB` / `DOOM_WAD` / `DOOM_NATIVE_FRAMES`; the defaults match the demo scripts' cache.
//! Run:  `cargo test -p svm-run --test doom_diff -- --ignored --nocapture`

use svm_interp::{bytecode, iface, Host, StreamRole, Value};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Lower the on-ramp's §7 libc→capability imports (`write`/`read`/`exit`/`vm_*`) to concrete caps.
fn onramp_resolver(name: &str) -> Option<svm_ir::ResolvedCap> {
    let (type_id, op) = match name {
        "write" => (iface::STREAM, 1),
        "read" => (iface::STREAM, 0),
        "exit" => (iface::EXIT, 0),
        "vm_map" => (iface::MEMORY, 0),
        "vm_unmap" => (iface::MEMORY, 1),
        "vm_protect" => (iface::MEMORY, 2),
        "vm_page_size" => (iface::MEMORY, 3),
        _ => return None,
    };
    Some(svm_ir::ResolvedCap { type_id, op })
}

#[test]
#[ignore = "needs demos/doom/{fetch,build,diff}.sh artifacts; slow"]
fn doom_frame_hashes_match_native() {
    let svmb = std::fs::read(env_or(
        "DOOM_SVMB",
        "/tmp/doomgeneric_cache/bcdiff/doom_diff.svmb",
    ))
    .expect("guest doom_diff.svmb — run demos/doom/diff.sh first");
    let wad = std::fs::read(env_or("DOOM_WAD", "/tmp/doomgeneric_cache/doom1.wad"))
        .expect("shareware doom1.wad");
    let native = std::fs::read_to_string(env_or(
        "DOOM_NATIVE_FRAMES",
        "/tmp/doomgeneric_cache/native/native_frames.txt",
    ))
    .expect("native frame list — run demos/doom/diff.sh first");
    assert!(&wad[..4] == b"IWAD", "shareware IWAD");

    let m = svm_ir::resolve_imports(&svm_encode::decode_module(&svmb).unwrap(), onramp_resolver)
        .expect("resolve imports");
    let arity = m.funcs.first().map_or(0, |f| f.params.len());

    // Powerbox prefix (stdout/stdin/exit/memory) by the entry's arity.
    let mut host = Host::new();
    let mut slots = Vec::new();
    if arity >= 1 {
        slots.push(Value::I32(host.grant_stream(StreamRole::Out)));
    }
    if arity >= 2 {
        slots.push(Value::I32(host.grant_stream(StreamRole::In)));
    }
    if arity >= 3 {
        slots.push(Value::I32(host.grant_exit()));
    }
    if arity >= 4 {
        slots.push(Value::I32(host.grant_memory()));
    }
    for (name, s) in ["stdout", "stdin", "exit", "memory"].iter().zip(&slots) {
        if let Value::I32(h) = s {
            host.register_cap_name(name, *h);
        }
    }

    // A read-only in-memory WAD over the `fs` capability (op protocol per lua_files_stdio.c):
    // 0 open(name,len,flags)->fd; 1 read(fd,buf,len)->n; 3 seek(fd,whence,off)->pos; 4 close.
    let mut cursors: Vec<u64> = Vec::new();
    let h = host.grant_host_fn(Box::new(move |op, a, mem| match op {
        0 => {
            let name = mem
                .and_then(|m| m.read_bytes(a[0] as u64, a[1] as u64))
                .unwrap_or_default();
            if String::from_utf8_lossy(&name).contains(".wad") {
                cursors.push(0);
                Ok(vec![(cursors.len() - 1) as i64])
            } else {
                Ok(vec![-2]) // ENOENT → fopen NULL → Doom uses defaults
            }
        }
        1 => {
            let (fd, buf, len) = (a[0] as usize, a[1] as u64, a[2] as u64);
            let (cur, end) = (cursors[fd], (cursors[fd] + len).min(wad.len() as u64));
            if end > cur {
                mem.expect("mem")
                    .write_bytes(buf, &wad[cur as usize..end as usize])
                    .unwrap();
            }
            cursors[fd] = end;
            Ok(vec![(end - cur) as i64])
        }
        3 => {
            let (fd, whence, off) = (a[0] as usize, a[1], a[2]);
            let base = match whence {
                1 => cursors[fd] as i64,
                2 => wad.len() as i64,
                _ => 0,
            };
            cursors[fd] = (base + off).max(0) as u64;
            Ok(vec![cursors[fd] as i64])
        }
        2 => Ok(vec![a[2]]), // write: discard-accept (config/savegame), non-fatal
        _ => Ok(vec![0]),
    }));
    host.register_cap_name("fs", h);

    // func 0 = _start → main → doomgeneric_Create + the N-frame loop, printing a hash per frame.
    let mut fuel = u64::MAX;
    bytecode::compile_and_run_with_host(&m, 0, &slots, &mut fuel, &mut host)
        .expect("engine supports the module")
        .expect("guest runs without trapping");

    let guest = String::from_utf8_lossy(&host.stdout);
    let guest_frames: String = guest
        .lines()
        .filter(|l| l.starts_with("frame "))
        .map(|l| format!("{l}\n"))
        .collect();
    let unique = guest_frames
        .lines()
        .map(|l| l.split(' ').nth(2).unwrap_or(""))
        .collect::<std::collections::HashSet<_>>()
        .len();
    eprintln!(
        "guest emitted {} frame lines ({unique} unique hashes)",
        guest_frames.lines().count()
    );
    assert!(!guest_frames.is_empty(), "guest produced frame hashes");
    assert_eq!(
        guest_frames, native,
        "guest frame hashes must match the native `cc` build byte-for-byte",
    );
}
