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

use svm_interp::{bytecode, iface, Host, StreamRole};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// The on-ramp's §7 libc→capability import names (`write`/`read`/`exit`/`vm_*`) and the
/// `(type_id, op)` each manifest slot binds to at instantiation.
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

    // The fixture is a manifest module (IMPORTS.md phase 3+): a paramless `_start` (func 0) whose
    // named imports bind to capability slots at instantiation — no rewrite, no positional handles.
    let m = svm_encode::decode_module(&svmb).unwrap();
    assert!(
        m.funcs.first().is_some_and(|f| f.params.is_empty()),
        "the committed fixture must carry the paramless manifest `_start` (rebuild via \
         demos/doom/diff.sh if this is a stale positional-entry blob)"
    );

    // Grant the powerbox prefix (stdout/stdin/exit/memory) and bind each manifest import's slot to
    // its `(type_id, op)` + granted handle — the same binding `svm_run::instantiate` performs.
    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let stdin = host.grant_stream(StreamRole::In);
    let exit = host.grant_exit();
    let memory = host.grant_memory();
    for (name, h) in [
        ("stdout", stdout),
        ("stdin", stdin),
        ("exit", exit),
        ("memory", memory),
    ] {
        host.register_cap_name(name, h);
    }
    let bindings = m
        .imports
        .iter()
        .map(|im| match onramp_resolver(&im.name) {
            Some(cap) => {
                let handle = match (cap.type_id, cap.op) {
                    (iface::STREAM, 1) => stdout,
                    (iface::STREAM, _) => stdin,
                    (iface::EXIT, _) => exit,
                    _ => memory,
                };
                svm_interp::BoundImport::required(cap.type_id, cap.op, handle)
            }
            // Unknown name: declared but unbound — a dispatch through it is a fail-closed CapFault.
            None => svm_interp::BoundImport::rebindable(0, 0, None),
        })
        .collect();
    host.set_import_bindings(bindings);

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
    bytecode::compile_and_run_with_host(&m, 0, &[], &mut fuel, &mut host)
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
