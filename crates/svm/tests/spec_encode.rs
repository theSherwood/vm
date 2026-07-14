//! SPEC.md suite 3 — encoding conformance. For every op row in the executable spec,
//! (1) the single-op module round-trips `decode∘encode = id` (the directed, per-op
//! version of the `roundtrip` fuzz property), and (2) the instruction's **opcode byte
//! is pinned** against the spec's independently-restated byte map (`OpRow::encoding`),
//! so an `svm-encode` renumbering or family-base move is a red test, not a silent
//! format break.
//!
//! Opcode location needs no knowledge of the container layout: the wire format has
//! **counts, not byte-length prefixes** (see `encode_module`/`encode_func`), so a
//! baseline module differing *only* in the single instruction encodes byte-identically
//! up to the instruction's first byte — the first divergent byte IS the opcode.

use svm::encode::{decode_module, encode_module};
use svm_ir::Inst;
use svm_spec::{all_rows, module_for, vectors_for, Enc, Shape};

#[test]
fn spec_encoding_conformance() {
    for row in all_rows() {
        let m = match row.shape {
            Shape::Operands => module_for(&row, &[]),
            Shape::Immediate => {
                let sample = vectors_for(&row)
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| panic!("no vectors for {}", row.id));
                module_for(&row, &sample)
            }
        };

        // (1) Round-trip identity.
        let bytes = encode_module(&m);
        let back =
            decode_module(&bytes).unwrap_or_else(|e| panic!("decode failed for {}: {e:?}", row.id));
        assert_eq!(back, m, "decode∘encode changed the IR for {}", row.id);

        // (2) Opcode pin by first divergence against a const baseline (a different
        // const for the const rows themselves).
        let mut base = m.clone();
        base.funcs[0].blocks[0].insts[0] = if row.encoding == Enc::Byte(0x10) {
            Inst::ConstI64(0)
        } else {
            Inst::ConstI32(0)
        };
        let base_bytes = encode_module(&base);
        let i = bytes
            .iter()
            .zip(&base_bytes)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| panic!("no divergence vs const baseline for {}", row.id));
        match row.encoding {
            Enc::Byte(b) => assert_eq!(
                bytes[i], b,
                "opcode byte for {}: spec says {b:#04x}, encoder wrote {:#04x}",
                row.id, bytes[i]
            ),
            Enc::Prefixed(p, s) => {
                assert_eq!(
                    bytes[i], p,
                    "escape prefix for {}: spec says {p:#04x}, encoder wrote {:#04x}",
                    row.id, bytes[i]
                );
                assert_eq!(
                    bytes[i + 1],
                    s,
                    "sub-opcode for {}: spec says {s:#04x}, encoder wrote {:#04x}",
                    row.id,
                    bytes[i + 1]
                );
            }
        }
    }
}
