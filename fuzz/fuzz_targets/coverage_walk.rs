//! libFuzzer target for the §3.5 **coverage walk** (`svm_interp::coverage_remap`) — the
//! grouped-import binding hinge. On arbitrary requirement/provider `(name, sig)` lists the
//! walk must never panic, and a returned remap must actually witness coverage:
//!
//! * one remap entry per required op, each **in range** of the provider's op list;
//! * the mapped provider op's signature **equals** the required signature;
//! * with provider names present, the mapped op's name equals the required name;
//! * with a name-less provider (legacy wire), the mapped op is the **first** sig match.
//!
//! A violated invariant here is a mis-bind: a consumer-local op silently dispatching to the
//! wrong provider op — the exact bug class coverage binding exists to prevent.
//!
//! Run: `cargo +nightly fuzz run coverage_walk`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_ir::{FuncType, ValType};

/// Decode a small `(names, sigs)` pool from raw fuzz bytes: names from a tiny alphabet (so
/// requirement/provider names collide often — the interesting case), signatures from a small
/// shape pool (ditto).
fn lists(data: &mut &[u8]) -> (Vec<String>, Vec<FuncType>) {
    let tys = [ValType::I32, ValType::I64, ValType::F32, ValType::F64];
    let mut names = Vec::new();
    let mut sigs = Vec::new();
    let n = data.first().copied().unwrap_or(0) % 9;
    *data = data.get(1..).unwrap_or(&[]);
    for _ in 0..n {
        let b = data.first().copied().unwrap_or(0);
        *data = data.get(1..).unwrap_or(&[]);
        names.push(format!("op{}", b % 5));
        let nparams = (b >> 3) % 4;
        let params = (0..nparams)
            .map(|i| tys[((b as usize) + i as usize) % tys.len()])
            .collect();
        let results = if b & 0x40 != 0 {
            vec![tys[(b as usize >> 1) % tys.len()]]
        } else {
            vec![]
        };
        sigs.push(FuncType { params, results });
    }
    (names, sigs)
}

fuzz_target!(|data: &[u8]| {
    let mut d = data;
    let (req_names, req_sigs) = lists(&mut d);
    let (prov_names, prov_sigs) = lists(&mut d);
    // Both name-keyed (names present) and legacy positional (names dropped) provider forms.
    for prov_names in [prov_names.as_slice(), &[]] {
        let Some(remap) = svm_interp::coverage_remap(&req_names, &req_sigs, prov_names, &prov_sigs)
        else {
            continue;
        };
        assert_eq!(remap.len(), req_sigs.len(), "one remap entry per required op");
        for (i, &p) in remap.iter().enumerate() {
            let p = p as usize;
            assert!(p < prov_sigs.len(), "remap entry in provider range");
            assert_eq!(prov_sigs[p], req_sigs[i], "mapped signature equals requirement");
            if prov_names.is_empty() {
                let first = prov_sigs.iter().position(|s| *s == req_sigs[i]);
                assert_eq!(first, Some(p), "legacy fallback maps to the first sig match");
            } else {
                assert_eq!(prov_names[p], req_names[i], "mapped name equals requirement");
            }
        }
    }
});
