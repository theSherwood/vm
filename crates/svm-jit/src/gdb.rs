//! In-memory ELF wrapper + the **GDB JIT interface** for the W5 JIT/DWARF tier (DEBUGGING.md §7,
//! Stage 2c) — the registration half that makes the Stage 2a/2b DWARF (`.debug_line` /
//! `.debug_info` / `.debug_abbrev`) visible to a live gdb/lldb.
//!
//! gdb discovers JIT'd code through a fixed protocol it knows by symbol name: it plants a breakpoint
//! on [`__jit_debug_register_code`] and walks the [`__jit_debug_descriptor`] linked list, reading
//! each [`JitCodeEntry`]'s `symfile` as an in-memory ELF object. So we wrap the finalized machine
//! code (an `SHT_NOBITS` `.text` whose `sh_addr` is the *real* runtime address) plus the synthesized
//! DWARF sections in a minimal hand-rolled ELF64 (the doc's "lean minimal-hand-rolled" decision —
//! only `.text` + a few `.debug_*` + a symbol table are needed), and link/unlink it on the
//! descriptor list around a call to the register hook.
//!
//! Strippable host-side tooling, untrusted-for-escape (§2a): a malformed object mis-renders in the
//! debugger (or gdb ignores it), never affects the running guest. Hand-rolled (no `object`/`gimli`)
//! to match the rest of the waist's ethos, and because the ELF we need is a fixed, tiny shape.

use std::sync::Mutex;

// --- minimal ELF64 (little-endian) builder ----------------------------------------------------

const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EV_CURRENT: u8 = 1;
const ET_REL: u16 = 1;
// The architecture gdb expects in `e_machine`; the JIT only targets the host, so pick it by cfg.
#[cfg(target_arch = "x86_64")]
const EM_HOST: u16 = 62; // EM_X86_64
#[cfg(target_arch = "aarch64")]
const EM_HOST: u16 = 183; // EM_AARCH64
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const EM_HOST: u16 = 0; // EM_NONE — still well-formed; gdb just won't recognize the ISA

const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_NOBITS: u32 = 8;
const SHF_ALLOC: u64 = 0x2;
const SHF_EXECINSTR: u64 = 0x4;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;

const EHDR_LEN: usize = 64;
const SHDR_LEN: usize = 64;
const SYM_LEN: usize = 24;

/// Append one `Elf64_Shdr` (64 bytes, the field order gdb's BFD reader expects).
#[allow(clippy::too_many_arguments)]
fn push_shdr(
    out: &mut Vec<u8>,
    name: u32,
    sh_type: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
) {
    out.extend_from_slice(&name.to_le_bytes());
    out.extend_from_slice(&sh_type.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&addr.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(&link.to_le_bytes());
    out.extend_from_slice(&info.to_le_bytes());
    out.extend_from_slice(&addralign.to_le_bytes());
    out.extend_from_slice(&entsize.to_le_bytes());
}

/// Build the in-memory ELF gdb reads as a JIT `symfile`: an `SHT_NOBITS` `.text` whose `sh_addr` is
/// the live code address (gdb reads the actual bytes from the inferior), the three DWARF sections,
/// and a `.symtab`/`.strtab` naming one `STT_FUNC` per function at its real `[lo, hi)` so a stop
/// in JIT'd code resolves to a function even before the DWARF is consulted. `funcs` is `(func index,
/// lo, hi)` (absolute machine addresses); `[code_base, code_base + code_size)` spans them all.
#[allow(clippy::too_many_arguments)]
pub fn build_elf(
    code_base: u64,
    code_size: u64,
    funcs: &[(u32, u64, u64)],
    debug_info: &[u8],
    debug_abbrev: &[u8],
    debug_line: &[u8],
    debug_loc: &[u8],
    debug_frame: &[u8],
) -> Vec<u8> {
    // Section name string table (`.shstrtab`): a leading NUL, then each name NUL-terminated. Record
    // each name's offset for the headers.
    let names = [
        ".text",
        ".debug_abbrev",
        ".debug_info",
        ".debug_line",
        ".debug_loc",
        ".debug_frame",
        ".symtab",
        ".strtab",
        ".shstrtab",
    ];
    let mut shstrtab = vec![0u8];
    let mut name_off = [0u32; 9];
    for (i, n) in names.iter().enumerate() {
        name_off[i] = shstrtab.len() as u32;
        shstrtab.extend_from_slice(n.as_bytes());
        shstrtab.push(0);
    }

    // Symbol string table (`.strtab`) + symbol table (`.symtab`). Index 0 is the reserved null
    // symbol (all zero); then one global `STT_FUNC` per function.
    let mut strtab = vec![0u8];
    let mut symtab = vec![0u8; SYM_LEN]; // null symbol
    for &(func, lo, hi) in funcs {
        let st_name = strtab.len() as u32;
        strtab.extend_from_slice(format!("fn{func}").as_bytes());
        strtab.push(0);
        symtab.extend_from_slice(&st_name.to_le_bytes()); // st_name
        symtab.push((STB_GLOBAL << 4) | STT_FUNC); // st_info
        symtab.push(0); // st_other
        symtab.extend_from_slice(&1u16.to_le_bytes()); // st_shndx → .text (section 1)
        symtab.extend_from_slice(&lo.to_le_bytes()); // st_value
        symtab.extend_from_slice(&hi.saturating_sub(lo).to_le_bytes()); // st_size
    }

    // Lay the section *contents* out after the ELF header, 8-byte aligned, recording each one's file
    // offset. `.text` is NOBITS (no bytes in the file), so it gets a zero offset.
    let mut body = Vec::new();
    let off = |body: &mut Vec<u8>, data: &[u8]| -> (u64, u64) {
        while !(EHDR_LEN + body.len()).is_multiple_of(8) {
            body.push(0);
        }
        let o = (EHDR_LEN + body.len()) as u64;
        body.extend_from_slice(data);
        (o, data.len() as u64)
    };
    let (abbrev_off, abbrev_sz) = off(&mut body, debug_abbrev);
    let (info_off, info_sz) = off(&mut body, debug_info);
    let (line_off, line_sz) = off(&mut body, debug_line);
    let (loc_off, loc_sz) = off(&mut body, debug_loc);
    let (frame_off, frame_sz) = off(&mut body, debug_frame);
    let (sym_off, sym_sz) = off(&mut body, &symtab);
    let (str_off, str_sz) = off(&mut body, &strtab);
    let (shstr_off, shstr_sz) = off(&mut body, &shstrtab);

    // The section header table follows the body, 8-byte aligned.
    while !(EHDR_LEN + body.len()).is_multiple_of(8) {
        body.push(0);
    }
    let shoff = (EHDR_LEN + body.len()) as u64;
    let shnum: u16 = 10; // NULL + .text + 5×debug + symtab + strtab + shstrtab
    let shstrndx: u16 = 9;

    // --- ELF header ---
    let mut out = Vec::with_capacity(EHDR_LEN + body.len() + shnum as usize * SHDR_LEN);
    out.extend_from_slice(&[0x7f, b'E', b'L', b'F']);
    out.push(ELFCLASS64);
    out.push(ELFDATA2LSB);
    out.push(EV_CURRENT);
    out.extend_from_slice(&[0u8; 9]); // EI_OSABI .. EI_PAD
    out.extend_from_slice(&ET_REL.to_le_bytes()); // e_type
    out.extend_from_slice(&EM_HOST.to_le_bytes()); // e_machine
    out.extend_from_slice(&1u32.to_le_bytes()); // e_version
    out.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    out.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    out.extend_from_slice(&shoff.to_le_bytes()); // e_shoff
    out.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    out.extend_from_slice(&(EHDR_LEN as u16).to_le_bytes()); // e_ehsize
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    out.extend_from_slice(&(SHDR_LEN as u16).to_le_bytes()); // e_shentsize
    out.extend_from_slice(&shnum.to_le_bytes()); // e_shnum
    out.extend_from_slice(&shstrndx.to_le_bytes()); // e_shstrndx
    debug_assert_eq!(out.len(), EHDR_LEN);

    out.extend_from_slice(&body);

    // --- section header table ---
    push_shdr(&mut out, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // [0] NULL
    push_shdr(
        &mut out,
        name_off[0],
        SHT_NOBITS,
        SHF_ALLOC | SHF_EXECINSTR,
        code_base, // sh_addr: the live code address gdb maps the DWARF onto
        0,
        code_size,
        0,
        0,
        16,
        0,
    ); // [1] .text
    push_shdr(
        &mut out,
        name_off[1],
        SHT_PROGBITS,
        0,
        0,
        abbrev_off,
        abbrev_sz,
        0,
        0,
        1,
        0,
    ); // [2] .debug_abbrev
    push_shdr(
        &mut out,
        name_off[2],
        SHT_PROGBITS,
        0,
        0,
        info_off,
        info_sz,
        0,
        0,
        1,
        0,
    ); // [3] .debug_info
    push_shdr(
        &mut out,
        name_off[3],
        SHT_PROGBITS,
        0,
        0,
        line_off,
        line_sz,
        0,
        0,
        1,
        0,
    ); // [4] .debug_line
    push_shdr(
        &mut out,
        name_off[4],
        SHT_PROGBITS,
        0,
        0,
        loc_off,
        loc_sz,
        0,
        0,
        1,
        0,
    ); // [5] .debug_loc
    push_shdr(
        &mut out,
        name_off[5],
        SHT_PROGBITS,
        0,
        0,
        frame_off,
        frame_sz,
        0,
        0,
        1,
        0,
    ); // [6] .debug_frame
    push_shdr(
        &mut out,
        name_off[6],
        SHT_SYMTAB,
        0,
        0,
        sym_off,
        sym_sz,
        8,                                                     // sh_link → .strtab (section 8)
        (sym_sz / SYM_LEN as u64) as u32 - funcs.len() as u32, // sh_info: first global = local count (just the null symbol)
        8,
        SYM_LEN as u64,
    ); // [7] .symtab
    push_shdr(
        &mut out,
        name_off[7],
        SHT_STRTAB,
        0,
        0,
        str_off,
        str_sz,
        0,
        0,
        1,
        0,
    ); // [8] .strtab
    push_shdr(
        &mut out,
        name_off[8],
        SHT_STRTAB,
        0,
        0,
        shstr_off,
        shstr_sz,
        0,
        0,
        1,
        0,
    ); // [9] .shstrtab

    out
}

/// Extract a named section's bytes from an ELF64 object (the inverse of [`build_elf`]'s layout) —
/// for round-tripping the embedded DWARF back through the readers in tests, and any host tool that
/// wants the sections out of the `symfile`. `None` if the input isn't a parseable ELF64 or the
/// section is absent.
pub fn elf_section<'a>(elf: &'a [u8], name: &str) -> Option<&'a [u8]> {
    if elf.len() < EHDR_LEN || elf[0..4] != [0x7f, b'E', b'L', b'F'] {
        return None;
    }
    let u64at =
        |o: usize| -> Option<u64> { Some(u64::from_le_bytes(elf.get(o..o + 8)?.try_into().ok()?)) };
    let u32at =
        |o: usize| -> Option<u32> { Some(u32::from_le_bytes(elf.get(o..o + 4)?.try_into().ok()?)) };
    let u16at =
        |o: usize| -> Option<u16> { Some(u16::from_le_bytes(elf.get(o..o + 2)?.try_into().ok()?)) };

    let shoff = u64at(40)? as usize;
    let shentsize = u16at(58)? as usize;
    let shnum = u16at(60)? as usize;
    let shstrndx = u16at(62)? as usize;
    let shdr = |i: usize| shoff + i * shentsize;

    // The section-name string table.
    let str_off = u64at(shdr(shstrndx) + 24)? as usize;
    let str_size = u64at(shdr(shstrndx) + 32)? as usize;
    let shstr = elf.get(str_off..str_off + str_size)?;

    for i in 0..shnum {
        let nameoff = u32at(shdr(i))? as usize;
        let end = shstr[nameoff..].iter().position(|&b| b == 0)? + nameoff;
        if &shstr[nameoff..end] == name.as_bytes() {
            let off = u64at(shdr(i) + 24)? as usize;
            let size = u64at(shdr(i) + 32)? as usize;
            return elf.get(off..off + size);
        }
    }
    None
}

// --- the GDB JIT interface --------------------------------------------------------------------

const JIT_NOACTION: u32 = 0;
const JIT_REGISTER_FN: u32 = 1;
const JIT_UNREGISTER_FN: u32 = 2;

/// One node of the JIT-object linked list gdb walks. Layout is fixed by the GDB JIT ABI.
#[repr(C)]
pub struct JitCodeEntry {
    next: *mut JitCodeEntry,
    prev: *mut JitCodeEntry,
    symfile_addr: *const u8,
    symfile_size: u64,
}

/// The process-global descriptor gdb reads (it knows this symbol by name). Layout fixed by the ABI.
#[repr(C)]
pub struct JitDescriptor {
    version: u32,
    /// One of `JIT_{NOACTION,REGISTER_FN,UNREGISTER_FN}` — what the pending [`relevant_entry`] op is.
    action_flag: u32,
    relevant_entry: *mut JitCodeEntry,
    first_entry: *mut JitCodeEntry,
}

// SAFETY: the descriptor is only ever mutated under `GDB_LOCK`; the raw pointers are never sent
// across threads except through that serialized protocol.
unsafe impl Sync for JitDescriptor {}

/// The hook gdb plants a breakpoint on. It **must not** be optimized away and must survive as a
/// distinct, named symbol — gdb finds it by name. `#[no_mangle]` + `#[inline(never)]` + a volatile
/// touch of the descriptor keep it real.
#[no_mangle]
#[inline(never)]
pub extern "C" fn __jit_debug_register_code() {
    // SAFETY: a relaxed volatile *read* of a plain `u32` field; `addr_of!` never forms a reference
    // to the `static mut`.
    unsafe {
        std::ptr::read_volatile(std::ptr::addr_of!(__jit_debug_descriptor.version));
    }
}

/// The descriptor gdb reads. `version` must be 1.
#[no_mangle]
pub static mut __jit_debug_descriptor: JitDescriptor = JitDescriptor {
    version: 1,
    action_flag: JIT_NOACTION,
    relevant_entry: std::ptr::null_mut(),
    first_entry: std::ptr::null_mut(),
};

// The GDB JIT protocol is process-global and single-threaded by construction; serialize every
// descriptor mutation so concurrent (un)registrations can't corrupt the list.
static GDB_LOCK: Mutex<()> = Mutex::new(());

/// A live GDB JIT registration: owns the `symfile` bytes (kept on the heap so gdb's stored pointer
/// stays valid) and its [`JitCodeEntry`], and **unregisters on drop** — an RAII handle. Built by
/// [`crate::CompiledModule::register_with_gdb`]. Hold it as long as the JIT'd code is debuggable.
pub struct GdbRegistration {
    entry: *mut JitCodeEntry,
    // Kept alive (and never moved — it's a heap allocation) because `entry.symfile_addr` points
    // into it for gdb's lifetime.
    _symfile: Box<[u8]>,
}

impl GdbRegistration {
    /// Wrap `symfile` (an in-memory ELF from [`build_elf`]) in a code entry, link it onto the
    /// descriptor list, and notify gdb.
    pub fn register(symfile: Vec<u8>) -> GdbRegistration {
        let symfile = symfile.into_boxed_slice();
        let entry = Box::into_raw(Box::new(JitCodeEntry {
            next: std::ptr::null_mut(),
            prev: std::ptr::null_mut(),
            symfile_addr: symfile.as_ptr(),
            symfile_size: symfile.len() as u64,
        }));

        let _g = GDB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: under `GDB_LOCK`; `desc`/`entry` are valid raw pointers and all writes go through
        // them (no reference to the `static mut` is formed).
        unsafe {
            let desc = std::ptr::addr_of_mut!(__jit_debug_descriptor);
            let first = (*desc).first_entry;
            (*entry).next = first;
            if !first.is_null() {
                (*first).prev = entry;
            }
            (*desc).first_entry = entry;
            (*desc).relevant_entry = entry;
            (*desc).action_flag = JIT_REGISTER_FN;
        }
        __jit_debug_register_code();
        GdbRegistration {
            entry,
            _symfile: symfile,
        }
    }
}

impl Drop for GdbRegistration {
    fn drop(&mut self) {
        let _g = GDB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: under `GDB_LOCK`; `entry` was produced by `Box::into_raw` in `register` and is
        // unlinked exactly once here before being reclaimed.
        unsafe {
            let desc = std::ptr::addr_of_mut!(__jit_debug_descriptor);
            let entry = self.entry;
            let prev = (*entry).prev;
            let next = (*entry).next;
            if prev.is_null() {
                (*desc).first_entry = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
            (*desc).relevant_entry = entry;
            (*desc).action_flag = JIT_UNREGISTER_FN;
            __jit_debug_register_code();
            drop(Box::from_raw(entry));
        }
    }
}

/// A snapshot of the GDB JIT descriptor for tests/tooling: `(action_flag, number_of_entries)` —
/// the last action gdb was signaled and how many objects are currently registered.
pub fn descriptor_state() -> (u32, usize) {
    let _g = GDB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: under `GDB_LOCK`; only reads through the raw pointer, forming no reference.
    unsafe {
        let desc = std::ptr::addr_of!(__jit_debug_descriptor);
        let action = (*desc).action_flag;
        let mut n = 0;
        let mut e = (*desc).first_entry;
        while !e.is_null() {
            n += 1;
            e = (*e).next;
        }
        (action, n)
    }
}
