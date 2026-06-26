//! Restore-time validation of `AddressSpace`/`Instantiator` handle bindings (the §14 sub-window
//! base/size). A frozen artifact is untrusted, persisted input and its (non-cryptographic) module
//! digest is not an adversary defense — so the window-containment invariant the grant path
//! guarantees (and the §14 JIT instantiator's `unsafe` window copy *assumes*) must be re-checked on
//! restore. Without it, a forged `base` smuggled into a binding drives an out-of-window host pointer
//! on the next `instantiate` — a host-memory escape. These pin that the codec rejects it.

use svm_interp::Host;
use svm_ir::Module;
use svm_snapshot::{freeze, restore, RestoreError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

fn module() -> Module {
    svm_text::parse_module(&format!(
        "memory {SIZE_LOG2}\n\
         func () -> (i64) {{\n\
         block0():\n\
         \x20 v0 = i64.const 0\n\
         \x20 return v0\n\
         }}\n"
    ))
    .expect("parse")
}

#[test]
fn restore_accepts_in_window_address_space_binding() {
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    // The legitimate whole-window root grant (`base = 0`, `size = window`).
    host.grant_address_space(0, WINDOW as u64);
    let art = freeze(&m, &window, &host).expect("freeze");
    let mut rhost = Host::new();
    restore(&art, &m, &mut rhost).expect("an in-window binding must restore");
}

#[test]
fn restore_rejects_out_of_window_instantiator_binding() {
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    // The grant path doesn't validate base/size (a documented caller contract), so freeze happily
    // encodes a forged, out-of-window Instantiator base. Restore must reject it: otherwise the §14
    // JIT instantiator would later compute `parent_mem_base + base + off` outside the window and
    // `from_raw_parts`/`copy_*` through it.
    host.grant_instantiator(WINDOW as u64, 4096);
    let art = freeze(&m, &window, &host).expect("freeze");
    let mut rhost = Host::new();
    assert_eq!(
        restore(&art, &m, &mut rhost),
        Err(RestoreError::BindingOutOfWindow),
        "an out-of-window Instantiator binding must be rejected on restore"
    );
}

#[test]
fn restore_rejects_overflowing_address_space_binding() {
    let m = module();
    let window = vec![0u8; WINDOW];
    let mut host = Host::new();
    // base + size wraps u64; the checked add must reject rather than alias back into the window.
    // The largest window-aligned base (`!(WINDOW-1)`) + WINDOW overflows u64.
    host.grant_address_space(!(WINDOW as u64 - 1), WINDOW as u64);
    let art = freeze(&m, &window, &host).expect("freeze");
    let mut rhost = Host::new();
    assert_eq!(
        restore(&art, &m, &mut rhost),
        Err(RestoreError::BindingOutOfWindow),
        "a base+size overflow must be rejected, not wrapped"
    );
}
