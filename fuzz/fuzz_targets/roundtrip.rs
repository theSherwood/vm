//! libFuzzer target for format round-trip identity (`DESIGN.md` §3a).
//!
//! For any decodable module, re-encoding and decoding again must reproduce it
//! exactly (`decode ∘ encode = id` on the IR). For any *verified* module, the text
//! form must round-trip identically too (`parse ∘ print = id`). A crash here is an
//! encoder/decoder or printer/parser bug.
//!
//! Run: `cargo +nightly fuzz run roundtrip`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_encode::{decode_module, encode_module};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

fuzz_target!(|data: &[u8]| {
    let Ok(m) = decode_module(data) else { return };

    // Binary round-trip holds for every decodable module (not just verified ones).
    let re = decode_module(&encode_module(&m)).expect("re-decode of encoded IR failed");
    assert_eq!(m, re, "binary round-trip changed the IR");

    // Text round-trip is only well-defined for verified modules (the text form can
    // only name backward value references, valid call indices, etc.).
    if verify_module(&m).is_ok() {
        let parsed = parse_module(&print_module(&m)).expect("re-parse of printed IR failed");
        assert_eq!(m, parsed, "text round-trip changed the IR");
    }
});
