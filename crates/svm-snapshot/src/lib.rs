//! `svm-snapshot` — the durable-domain **snapshot artifact codec** (DURABILITY.md §12).
//!
//! A **tooling-tier, +0-TCB** crate: it (de)serializes a quiesced durable domain into the
//! backend-independent, recompile-survivable artifact of §12, and back. It pairs with
//! `svm-durable` (the IR→IR freeze/thaw transform) but does not depend on it — the transform
//! produces the in-window shadow state; this crate only moves bytes + authority.
//!
//! # What an artifact is (§12.0)
//!
//! A frozen single-vCPU domain is described almost entirely by its **window image** — the
//! shadow stack, spilled live values, and the state word are all guest-resident bytes, so
//! they ride along in the window for free. What lives *host-side* and is captured separately
//! is the **handle table** (authority, not the resources it names — §12.5 / D-scope). The
//! artifact binds the **instrumented-module digest** (R5 / D-hash): restore refuses on a
//! mismatch, which is the durability boundary from §1 (the shadow schema is a function of the
//! instrumented module's structure).
//!
//! # Container (§12.1)
//!
//! `b"SVMD"`, a `u16` format version, then ascending-tag TLV sections (`tag`/`len`/body, so a
//! reader can skip unknown tags). Encoding is **canonical** — minimal LEB128, fixed section
//! order, sparse entries ascending — so the §12.6 invariant "re-serialize a freshly-restored
//! domain at the same safepoint is byte-identical" is a plain `==`.
//!
//! # Scope (Phase-1 domain shape)
//!
//! Single vCPU, no fibers, no protected pages: the window image is sparse with **zero-page
//! elision**, every page `Rw`; control state is just the header's `vcpu_count = 1` /
//! `fiber_count = 0` (the state word is in the window image). §12.4 fiber/dispatch state and
//! `svm-mem` page-prot restore are later slices; the TLV container is forward-compatible.

#![forbid(unsafe_code)]

use svm_encode::encode_module;
use svm_interp::{DurableBinding, DurableHandle, Host, NonDurableHandle, StreamRole};
use svm_ir::Module;

/// Container magic (§12.2): "SVM-Durable".
const MAGIC: &[u8; 4] = b"SVMD";
/// Format version; bump on an incompatible change (§12.2).
const FORMAT_VERSION: u16 = 1;
/// Window-image page granularity (§12.3). The window length is a power of two `≥ PAGE`, so
/// every page is exactly `PAGE` bytes (no partial tail).
const PAGE: usize = 4096;

// ---- Section tags (ascending, §12.2-12.5). Tag 2 (control state) is unused in Phase 1. ----
const TAG_HEADER: u64 = 0;
const TAG_WINDOW: u64 = 1;
const TAG_HANDLES: u64 = 3;

// ---- Binding descriptors (§12.5). One tag byte + value-typed payload. ----
const B_STREAM: u8 = 0;
const B_EXIT: u8 = 1;
const B_CLOCK: u8 = 2;
const B_MEMORY: u8 = 3;
const B_YIELDER: u8 = 4;
const B_ADDRESS_SPACE: u8 = 5;
const B_INSTANTIATOR: u8 = 6;

const PROT_RW: u8 = 0;

/// Why a domain can't be frozen.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FreezeError {
    /// A live handle isn't re-grantable, so freeze refuses rather than dropping authority
    /// (§12.5). The domain must close/drain it first.
    NonDurableHandle(NonDurableHandle),
    /// The window length isn't a power of two `≥ PAGE` — it can't be a valid masked window.
    WindowGeometry(usize),
}

/// Why restoring an artifact failed. All are fail-closed: restore never yields partial state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RestoreError {
    /// Not an `SVMD` container.
    BadMagic,
    /// A format version this build doesn't understand.
    UnsupportedVersion(u16),
    /// The byte stream ended mid-field (a section length or varint ran off the end).
    Truncated,
    /// A section body was malformed (bad varint, unknown binding tag, leftover bytes, …).
    Malformed,
    /// Required section (header / window / handles) missing.
    MissingSection(u64),
    /// The artifact's instrumented-module digest doesn't match the module restore was handed
    /// (R5 / §12.6 invariant 2 — the durability boundary).
    ModuleMismatch,
    /// The artifact's window geometry doesn't match the module's declared memory.
    GeometryMismatch,
    /// A handle named a slot outside the table capacity.
    SlotOutOfRange(u32),
}

/// Serialize a quiesced durable domain into a §12 artifact: the `window` image (the shadow
/// state rides along) bound to `module`'s digest, plus `host`'s re-grantable handle table.
/// Refuses if a live handle isn't durable (§12.5).
pub fn freeze(module: &Module, window: &[u8], host: &Host) -> Result<Vec<u8>, FreezeError> {
    if !window.len().is_power_of_two() || window.len() < PAGE {
        return Err(FreezeError::WindowGeometry(window.len()));
    }
    let handles = host
        .capture_durable_handles()
        .map_err(FreezeError::NonDurableHandle)?;
    let digest = digest256(&encode_module(module));
    let reserved_log2 = window.len().trailing_zeros() as u8;

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

    // Section 0 — Header (§12.2).
    section(&mut out, TAG_HEADER, |b| {
        b.extend_from_slice(&digest);
        b.push(reserved_log2);
        write_uleb(b, window.len() as u64); // mapped
        write_uleb(b, PAGE as u64); // host page size at capture
        write_uleb(b, 1); // vcpu_count (single-vCPU Phase 1)
        write_uleb(b, 0); // fiber_count
    });

    // Section 1 — Window image (§12.3): sparse, zero-eliding, ascending page index.
    section(&mut out, TAG_WINDOW, |b| {
        let pages: Vec<(usize, &[u8])> = window
            .chunks_exact(PAGE)
            .enumerate()
            .filter(|(_, p)| p.iter().any(|&x| x != 0))
            .collect();
        write_uleb(b, pages.len() as u64);
        for (idx, page) in pages {
            write_uleb(b, idx as u64);
            b.push(PROT_RW);
            b.extend_from_slice(page);
        }
    });

    // Section 3 — Handle table (§12.5): ascending slot (capture already orders it).
    section(&mut out, TAG_HANDLES, |b| {
        write_uleb(b, handles.len() as u64);
        for h in &handles {
            write_uleb(b, h.slot as u64);
            write_uleb(b, h.generation as u64);
            write_uleb(b, h.type_id as u64);
            write_binding(b, &h.binding);
        }
    });

    Ok(out)
}

/// Restore a §12 artifact: validate it against `module` (R5 digest gate + geometry), re-grant
/// its handle table into `host` (a fresh table the embedder supplies behind its own resources,
/// D-scope), and return the reconstructed window image. The caller flips the state word to
/// `REWINDING` and re-enters to thaw.
pub fn restore(artifact: &[u8], module: &Module, host: &mut Host) -> Result<Vec<u8>, RestoreError> {
    let mut r = Reader::new(artifact);
    if r.take(4)? != MAGIC {
        return Err(RestoreError::BadMagic);
    }
    let version = u16::from_le_bytes([r.u8()?, r.u8()?]);
    if version != FORMAT_VERSION {
        return Err(RestoreError::UnsupportedVersion(version));
    }

    let (mut header, mut win_body, mut handles_body) = (None, None, None);
    while !r.at_end() {
        let tag = r.uleb()?;
        let len = r.uleb()? as usize;
        let body = r.take(len)?;
        match tag {
            TAG_HEADER => header = Some(body),
            TAG_WINDOW => win_body = Some(body),
            TAG_HANDLES => handles_body = Some(body),
            _ => {} // forward-compatible: skip unknown sections
        }
    }
    let header = header.ok_or(RestoreError::MissingSection(TAG_HEADER))?;
    let win_body = win_body.ok_or(RestoreError::MissingSection(TAG_WINDOW))?;
    let handles_body = handles_body.ok_or(RestoreError::MissingSection(TAG_HANDLES))?;

    // ---- Header: the R5 identity gate, then geometry. ----
    let mut h = Reader::new(header);
    let digest = h.take(32)?;
    let reserved_log2 = h.u8()?;
    let mapped = h.uleb()? as usize;
    let page_size = h.uleb()? as usize;
    let _vcpu_count = h.uleb()?;
    let _fiber_count = h.uleb()?;
    if digest != digest256(&encode_module(module)) {
        return Err(RestoreError::ModuleMismatch);
    }
    if let Some(mem) = &module.memory {
        if mem.size_log2 != reserved_log2 {
            return Err(RestoreError::GeometryMismatch);
        }
    }
    if page_size != PAGE || mapped != 1usize << reserved_log2 {
        return Err(RestoreError::GeometryMismatch);
    }

    // ---- Window image: zeroed window, splat the stored non-zero pages. ----
    let mut window = vec![0u8; mapped];
    let mut w = Reader::new(win_body);
    let n_pages = w.uleb()?;
    let mut last: Option<u64> = None;
    for _ in 0..n_pages {
        let idx = w.uleb()?;
        if last.is_some_and(|p| idx <= p) {
            return Err(RestoreError::Malformed); // non-canonical: pages must ascend
        }
        last = Some(idx);
        let _prot = w.u8()?;
        let bytes = w.take(PAGE)?;
        let start = (idx as usize)
            .checked_mul(PAGE)
            .filter(|&s| s + PAGE <= mapped)
            .ok_or(RestoreError::Malformed)?;
        window[start..start + PAGE].copy_from_slice(bytes);
    }
    if !w.at_end() {
        return Err(RestoreError::Malformed);
    }

    // ---- Handle table: decode, bounds-check, re-grant. ----
    let mut hr = Reader::new(handles_body);
    let n = hr.uleb()?;
    let mut handles = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let slot = u32::try_from(hr.uleb()?).map_err(|_| RestoreError::Malformed)?;
        if slot >= Host::handle_capacity() {
            return Err(RestoreError::SlotOutOfRange(slot));
        }
        let generation = u32::try_from(hr.uleb()?).map_err(|_| RestoreError::Malformed)?;
        let type_id = u32::try_from(hr.uleb()?).map_err(|_| RestoreError::Malformed)?;
        let binding = read_binding(&mut hr)?;
        handles.push(DurableHandle {
            slot,
            generation,
            type_id,
            binding,
        });
    }
    if !hr.at_end() {
        return Err(RestoreError::Malformed);
    }
    host.restore_durable_handles(&handles);

    Ok(window)
}

// ---- Binding (de)serialization (§12.5) ----

fn write_binding(b: &mut Vec<u8>, binding: &DurableBinding) {
    match *binding {
        DurableBinding::Stream(role) => {
            b.push(B_STREAM);
            b.push(match role {
                StreamRole::In => 0,
                StreamRole::Out => 1,
                StreamRole::Err => 2,
            });
        }
        DurableBinding::Exit => b.push(B_EXIT),
        DurableBinding::Clock => b.push(B_CLOCK),
        DurableBinding::Memory => b.push(B_MEMORY),
        DurableBinding::Yielder => b.push(B_YIELDER),
        DurableBinding::AddressSpace { base, size } => {
            b.push(B_ADDRESS_SPACE);
            write_uleb(b, base);
            write_uleb(b, size);
        }
        DurableBinding::Instantiator { base, size } => {
            b.push(B_INSTANTIATOR);
            write_uleb(b, base);
            write_uleb(b, size);
        }
    }
}

fn read_binding(r: &mut Reader) -> Result<DurableBinding, RestoreError> {
    Ok(match r.u8()? {
        B_STREAM => DurableBinding::Stream(match r.u8()? {
            0 => StreamRole::In,
            1 => StreamRole::Out,
            2 => StreamRole::Err,
            _ => return Err(RestoreError::Malformed),
        }),
        B_EXIT => DurableBinding::Exit,
        B_CLOCK => DurableBinding::Clock,
        B_MEMORY => DurableBinding::Memory,
        B_YIELDER => DurableBinding::Yielder,
        B_ADDRESS_SPACE => DurableBinding::AddressSpace {
            base: r.uleb()?,
            size: r.uleb()?,
        },
        B_INSTANTIATOR => DurableBinding::Instantiator {
            base: r.uleb()?,
            size: r.uleb()?,
        },
        _ => return Err(RestoreError::Malformed),
    })
}

// ---- Container helpers ----

/// Emit a TLV section: tag, then the body's length, then the body built by `f`.
fn section(out: &mut Vec<u8>, tag: u64, f: impl FnOnce(&mut Vec<u8>)) {
    let mut body = Vec::new();
    f(&mut body);
    write_uleb(out, tag);
    write_uleb(out, body.len() as u64);
    out.extend_from_slice(&body);
}

/// Minimal-length LEB128 (the canonical encoding §12.1 requires).
fn write_uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A bounds-checked forward cursor over the artifact bytes.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Reader<'a> {
        Reader { b, pos: 0 }
    }
    fn at_end(&self) -> bool {
        self.pos >= self.b.len()
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], RestoreError> {
        let end = self.pos.checked_add(n).ok_or(RestoreError::Truncated)?;
        let s = self.b.get(self.pos..end).ok_or(RestoreError::Truncated)?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, RestoreError> {
        Ok(self.take(1)?[0])
    }
    fn uleb(&mut self) -> Result<u64, RestoreError> {
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.u8()?;
            if shift >= 64 || (shift == 63 && byte > 1) {
                return Err(RestoreError::Malformed); // overflow / non-canonical
            }
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                if byte == 0 && shift != 0 {
                    return Err(RestoreError::Malformed); // non-minimal trailing zero byte
                }
                return Ok(v);
            }
            shift += 7;
        }
    }
}

/// A 256-bit **non-cryptographic** digest of `bytes` (D-hash): four independent multiplicative
/// lanes with distinct odd primes, length-mixed. Guards *accidental* restore-into-wrong-module
/// (§12.6 invariant 2), not an adversary (a guest can't forge past confinement — §3), so no
/// crypto-hash dependency is pulled into the toolchain.
fn digest256(bytes: &[u8]) -> [u8; 32] {
    const PRIMES: [u64; 4] = [
        0x0000_0100_0000_01b3, // FNV-1a 64-bit prime
        0xff51_afd7_ed55_8ccd, // murmur3 fmix constant (odd)
        0xc4ce_b9fe_1a85_ec53, // murmur3 fmix constant (odd)
        0x9e37_79b9_7f4a_7c15, // golden-ratio odd constant
    ];
    let mut lanes: [u64; 4] = [
        0xcbf2_9ce4_8422_2325,
        0x1234_5678_9abc_def0,
        0x0f1e_2d3c_4b5a_6978,
        0xa5a5_5a5a_3c3c_c3c3,
    ];
    for &byte in bytes {
        for (lane, prime) in lanes.iter_mut().zip(PRIMES) {
            *lane = (*lane ^ byte as u64).wrapping_mul(prime);
        }
    }
    for (lane, prime) in lanes.iter_mut().zip(PRIMES) {
        *lane = (*lane ^ bytes.len() as u64).wrapping_mul(prime);
    }
    let mut out = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}
