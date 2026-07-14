//! The persistent `bytecode::Reactor` — instantiate once, call exports many times, with the **whole**
//! guest window (globals/BSS **and** a `vm_map`-grown heap) persisting across calls. This is what the
//! snapshot reactors (`svm-run`'s `Session`, the browser `OnrampReactor`) could not do: they
//! round-trip only the low `SNAP_CAP` (256 KiB) prefix, so a guest's state above that — a grown heap —
//! was lost every call. Keeping the `Mem` live fixes it, and is what lets a heavy-heap guest (the
//! playground's Game of Life, eventually Doom) hold state frame to frame.

use svm_interp::{bytecode, Host, Value};
use svm_text::parse_module;

// A counter at a HIGH address (300000 ≈ 293 KiB — above the 256 KiB `SNAP_CAP` the snapshot reactors
// captured) is loaded, incremented, stored, and returned. Over repeated `call`s it must climb
// 1, 2, 3, … which holds only if the reactor persists memory beyond that 256 KiB prefix.
const SRC: &str = r#"
memory 19
func () -> (i64) {
block0():
  v0 = i64.const 300000
  v1 = i64.load v0
  v2 = i64.const 1
  v3 = i64.add v1 v2
  i64.store v0 v3
  return v3
}
"#;

#[test]
fn reactor_persists_high_memory_across_calls() {
    let m = parse_module(SRC).expect("parse the counter module");
    let mut r = bytecode::Reactor::open(&m).expect("open the reactor");
    let mut host = Host::new();
    for expect in 1..=5i64 {
        let mut fuel = u64::MAX;
        let out = r
            .call(0, &[], &mut fuel, &mut host)
            .expect("call the counter");
        assert_eq!(
            out,
            vec![Value::I64(expect)],
            "the counter at 293 KiB climbs — memory above the 256 KiB prefix persisted across calls",
        );
    }
}

// A fresh reactor starts from a zeroed window (persistence is per-instance, not global): the counter
// begins at 1 again, proving `open` seeds a clean window rather than leaking the previous instance's.
#[test]
fn a_fresh_reactor_starts_clean() {
    let m = parse_module(SRC).expect("parse");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let mut r1 = bytecode::Reactor::open(&m).expect("open r1");
    assert_eq!(
        r1.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(1)])
    );
    assert_eq!(
        r1.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(2)])
    );
    let mut r2 = bytecode::Reactor::open(&m).expect("open r2");
    assert_eq!(
        r2.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(1)]),
        "a new reactor's window is fresh, not inherited from r1",
    );
}
