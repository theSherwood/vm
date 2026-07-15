//! **wasm-JIT reactor differential** (BROWSER.md § "wasm-JIT tier", slice 5c). Doom's whole `tick`
//! runs on **emitted wasm** each frame instead of the interpreter, over a window in the host's linear
//! memory; non-emitted (cross-tier) helpers bounce back to the interpreter over that same window with
//! the powerbox. This test plays the browser's JS host with `wasmi`: it instantiates the emitted `tick`
//! against a `wasmi` `Memory`, opens a [`JitOnrampReactor`] over that same memory (so `_start` and the
//! cross-tier callees seed/read the bytes the emitted code touches), drives frames by calling the
//! emitted `f{tick}(win, env, sp)`, and services `env.call_interp` with [`JitOnrampReactor::run_cross_tier`].
//! It asserts the emitted-`tick` frames are **byte-identical** to the interpreter reactor's frames (the
//! [`SharedOnrampReactor`] oracle, itself proven ≡ `OnrampReactor` in slice 5b) — the JIT correctness
//! gate. Requires the Doom assets (`doom.svmb` + `doom1.wad`); `#[ignore]`d without them.

use svm_browser::{Frame, JitOnrampReactor, SharedOnrampReactor, STATUS_OK};
use svm_interp::Value;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const DOOM_SVMB: &str = "/tmp/doomgeneric_cache/bc/doom.svmb";
const WAD: &str = "web/assets/doom1.wad";

// The emitted `tick`'s guest window is 2^24 = 16 MiB — covers Doom's zone heap (peaks ~11 MiB), so the
// emitter's static-`mapped` mask covers every access and no heap growth escapes the window.
const WIN_LOG2: u8 = 24;
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // the window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;
const FRAMES: usize = 3;

fn assets() -> Option<(svm_ir::Module, Vec<u8>)> {
    let svmb = std::fs::read(DOOM_SVMB).ok()?;
    let wad = std::fs::read(WAD).ok()?;
    Some((svm_encode::decode_module(&svmb).ok()?, wad))
}

/// A stable hash of a presented frame (dims + RGBA), so a per-frame equality check is cheap to compare.
fn frame_hash(f: &Frame) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    f.width.hash(&mut h);
    f.height.hash(&mut h);
    f.rgba.hash(&mut h);
    h.finish()
}

/// The interpreter-reactor oracle: `SharedOnrampReactor` (≡ `OnrampReactor`, slice 5b) over Doom, N
/// frames, returning each frame's `(hash, non_blank)`.
fn interp_frames(m: &svm_ir::Module, wad: &[u8]) -> Vec<(u64, bool)> {
    // 2^25 backing gives the size_log2=22 guest room to `vm_map`-grow its heap (as the browser interp
    // reactor does); the frames are what the JIT run must reproduce.
    let mut r = SharedOnrampReactor::open_owned_with_fs(m, 25, "doom1.wad".into(), wad.to_vec())
        .expect("open interp Doom reactor");
    (0..FRAMES)
        .map(|i| {
            let (s, _) = r.frame();
            assert_eq!(s, STATUS_OK, "interp frame {i} status");
            let f = r.take_frame().expect("interp frame presented");
            (frame_hash(&f), f.rgba.iter().any(|&b| b != 0))
        })
        .collect()
}

/// The wasm-JIT run: emitted `tick` on `wasmi`, cross-tier helpers on the interpreter over the shared
/// window. Returns each frame's `(hash, non_blank)`.
fn jit_frames(m: &svm_ir::Module, wad: &[u8]) -> Vec<(u64, bool)> {
    let engine = Engine::default();

    // One non-growable linear memory holding the env cell (below WIN_BASE) and the 16 MiB window. The
    // reactor lives in the `Store` data so the `env.call_interp` host closure reaches it through the
    // `Caller` (capturing only the Copy `memory` handle — the closure must be `Sync`).
    let total_bytes = WIN_BASE as u64 + WIN_SIZE;
    let pages = (total_bytes / (64 * 1024)) as u32;
    let mut store: Store<Option<JitOnrampReactor>> = Store::new(&engine, None);
    let memory =
        Memory::new(&mut store, MemoryType::new(pages, Some(pages))).expect("wasmi memory");

    // Open the reactor over *this* wasmi memory's window (a Region::shared at WIN_BASE): `_start` runs
    // on the interpreter and seeds the window in-place; then we compile the emitted `tick`.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    // SAFETY: `memory` is fixed-size (min==max), so its data pointer is stable for the run; the window
    // `[win_ptr, WIN_SIZE)` lives inside it and is used solely as this reactor's window.
    let reactor = unsafe {
        JitOnrampReactor::open_shared_jit(
            m,
            win_ptr,
            WIN_SIZE,
            WIN_LOG2,
            false,
            Some(("doom1.wad".into(), wad.to_vec())),
        )
    }
    .expect("open JIT Doom reactor");

    let emitted_wasm = reactor.emitted_wasm().to_vec();
    let entry_sp = reactor.entry_sp();
    let tick = reactor.tick();
    let module = WModule::new(&engine, &emitted_wasm).expect("emitted Doom tick validates");
    *store.data_mut() = Some(reactor);

    let mut linker: Linker<Option<JitOnrampReactor>> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_caller: Caller<'_, _>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Option<JitOnrampReactor>>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                // Clone the callee's declared arg/result types (release the reactor borrow before the run).
                let (params, results) = {
                    let r = caller.data().as_ref().unwrap();
                    let (p, rs) = r.func_sig(func as u32);
                    (p.to_vec(), rs.to_vec())
                };
                // Read the marshalled i64 arg slots (widen/narrow per the callee's declared types).
                let args: Vec<Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                svm_ir::ValType::I32 => Value::I32(raw as i32),
                                _ => Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                // Run the cross-tier callee on the interpreter over the shared window (no live wasmi
                // memory borrow held across the run — the interpreter writes through the same bytes).
                let outcome = caller
                    .data_mut()
                    .as_mut()
                    .unwrap()
                    .run_cross_tier(func as u32, &args);
                match outcome {
                    Ok(vals) => {
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            if i >= results.len() {
                                break;
                            }
                            let raw = match v {
                                Value::I32(x) => *x as u32 as u64,
                                Value::I64(x) => *x as u64,
                                _ => 0,
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Err(t) => {
                        caller
                            .data_mut()
                            .as_mut()
                            .unwrap()
                            .set_last_trap(format!("{t:?}"));
                        // Unwind the emitted `f{tick}` (the browser's JS import throwing) — `Exit` and
                        // real traps both surface as a caught RuntimeError at the top-level call.
                        Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ))
                    }
                }
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f_tick = instance
        .get_func(&store, &format!("f{tick}"))
        .expect("emitted f{tick} export");
    // `f{tick}`'s result arity (the SVM `tick` may return a value, e.g. an i32 status).
    let rtys: Vec<svm_ir::ValType> = store.data().as_ref().unwrap().func_sig(tick).1.to_vec();
    let mut results: Vec<Val> = rtys
        .iter()
        .map(|t| match t {
            svm_ir::ValType::I32 => Val::I32(0),
            svm_ir::ValType::I64 => Val::I64(0),
            svm_ir::ValType::F32 => Val::F32(0.0f32.into()),
            svm_ir::ValType::F64 => Val::F64(0.0f64.into()),
            _ => Val::I32(0),
        })
        .collect();

    (0..FRAMES)
        .map(|i| {
            // Refill fuel (the emitted code debits an i64 counter at env[0] and traps when it goes < 0).
            memory
                .write(&mut store, ENV_PTR as usize, &(1i64 << 52).to_le_bytes())
                .unwrap();
            let r = f_tick.call(
                &mut store,
                &[
                    Val::I32(WIN_BASE as i32),
                    Val::I32(ENV_PTR as i32),
                    Val::I64(entry_sp as i64),
                ],
                &mut results,
            );
            if let Err(e) = r {
                let why = store.data().as_ref().unwrap().last_trap().to_string();
                panic!("emitted tick frame {i} trapped: {e} ({why})");
            }
            let f = store
                .data()
                .as_ref()
                .unwrap()
                .take_frame()
                .expect("JIT frame presented");
            (frame_hash(&f), f.rgba.iter().any(|&b| b != 0))
        })
        .collect()
}

#[test]
fn doom_jit_tick_matches_interpreter() {
    let Some((m, wad)) = assets() else {
        eprintln!("skipping: Doom assets not present ({DOOM_SVMB} / {WAD})");
        return;
    };
    let interp = interp_frames(&m, &wad);
    let jit = jit_frames(&m, &wad);
    assert_eq!(interp.len(), FRAMES);
    for (i, (a, b)) in interp.iter().zip(&jit).enumerate() {
        assert!(a.1, "interp frame {i} should be non-blank");
        assert!(b.1, "JIT frame {i} should be non-blank");
        assert_eq!(
            a.0, b.0,
            "emitted-tick frame {i} differs from the interpreter frame"
        );
    }
}
