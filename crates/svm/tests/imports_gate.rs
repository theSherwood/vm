//! The IMPORTS.md §2.5 **completion gate for phase 4** — grep-clean checks, so "done" is checked,
//! not asserted. The legacy import-binding conventions (the instantiation-time rewrite, the
//! positional/stash bootstraps, the `CapBound` general-form resolvers) were deleted in phase 4;
//! this test scans the tree's source files and fails if any of their symbols reappear, and pins
//! `resolve_imports_with` — the one surviving rewrite pass — to its linker-only call sites.
//!
//! Scope: `.rs`/`.c`/`.h` sources under `crates/`, `browser/`, and `frontend/chibicc` (skipping
//! `target/` build output and this file, which necessarily names the banned symbols).

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Collect every scannable source file under `dir`, recursively, skipping build output.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            if name == "target" || name == "node_modules" || name.starts_with('.') {
                continue;
            }
            sources(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("rs" | "c" | "h")
        ) && name != "imports_gate.rs"
        {
            out.push(path);
        }
    }
}

fn scan(check: impl Fn(&Path, &str, &mut Vec<String>)) -> Vec<String> {
    let root = repo_root();
    let mut files = Vec::new();
    for top in ["crates", "browser", "frontend/chibicc"] {
        let dir = root.join(top);
        if dir.is_dir() {
            sources(&dir, &mut files);
        }
    }
    assert!(
        files.len() > 100,
        "gate scanned only {} files — the walk is broken, not the tree clean",
        files.len()
    );
    let mut violations = Vec::new();
    for f in &files {
        // Non-UTF-8 sources (none today) would be a scan gap; fail loudly rather than skip.
        let text = std::fs::read_to_string(f)
            .unwrap_or_else(|e| panic!("gate could not read {}: {e}", f.display()));
        let rel = f.strip_prefix(&root).unwrap();
        check(rel, &text, &mut violations);
    }
    violations
}

/// §2.5: the deleted conventions' symbols must not reappear anywhere in the tree's sources.
#[test]
fn deleted_import_machinery_stays_deleted() {
    // Split token construction keeps this file's own strings from ever matching a plain grep for
    // the banned names (the scan itself skips this file; greps by humans should stay clean too).
    let banned: Vec<String> = [
        ("Cap", "Bound"),
        ("resolve_", "bound"),
        ("powerbox_", "resolver"),
        ("synth_powerbox_", "start"),
        ("resolve_capability_", "imports"),
        ("NAMED_", "IMPORT"),
        ("handle_", "modules"),
        ("stash_", "base"),
    ]
    .iter()
    .map(|(a, b)| format!("{a}{b}"))
    .collect();
    let violations = scan(|rel, text, out| {
        for tok in &banned {
            if text.contains(tok.as_str()) {
                out.push(format!("{}: contains `{tok}`", rel.display()));
            }
        }
    });
    assert!(
        violations.is_empty(),
        "IMPORTS.md §2.5 gate: deleted symbols reappeared:\n{}",
        violations.join("\n")
    );
}

/// §2.5: `resolve_imports*` survives **in the linker only** — `Resolved::Func`/`Slot` (and the
/// `compile_linked` guest symbol table's `Cap`) are link-time symbol resolution, which
/// legitimately produces new module bytes. Instantiation never rewrites, so no other file may
/// call it. `patch_placeholder`/`SlotHandleNotConst` are the `Slot` rewrite's internals and stay
/// inside the linker (plus the dynlink linker tests).
#[test]
fn resolve_imports_calls_are_linker_only() {
    // Call sites of the surviving linker pass (an open paren distinguishes a call/definition from
    // prose in a doc comment).
    let call = "resolve_imports_with(";
    let callers: &[&str] = &[
        "crates/svm-ir/src/lib.rs",    // the pass itself + its linker tests
        "crates/svm-run/src/lib.rs",   // the guest-JIT `compile_linked` symbol-table path
        "browser/src/lib.rs",          // browser_jit_validator — the same `compile_linked` path
        "crates/svm-text/src/lib.rs",  // parse → link integration test
        "crates/svm/tests/dynlink.rs", // linker tests (Func/Slot)
        "crates/svm/tests/dynlink_runtime.rs",
        "crates/svm/tests/dynlink_resolve.rs",
        "crates/svm/tests/dynlink_repl.rs",
        "crates/svm/tests/c_shell.rs", // link a compiled C object's symbols (link-time, no handles)
        "crates/svm/tests/c_shell_exec.rs",
        "crates/svm/tests/stage1_posix_spawn.rs",
    ];
    let internals: &[(&str, &[&str])] = &[
        ("patch_placeholder", &["crates/svm-ir/src/lib.rs"]),
        (
            "SlotHandleNotConst",
            &["crates/svm-ir/src/lib.rs", "crates/svm/tests/dynlink.rs"],
        ),
    ];
    let violations = scan(|rel, text, out| {
        let rel_s = rel.to_string_lossy().replace('\\', "/");
        if text.contains(call) && !callers.contains(&rel_s.as_str()) {
            out.push(format!("{rel_s}: non-linker `{call}` call site"));
        }
        for (tok, allowed) in internals {
            if text.contains(tok) && !allowed.contains(&rel_s.as_str()) {
                out.push(format!("{rel_s}: `{tok}` outside the linker"));
            }
        }
    });
    assert!(
        violations.is_empty(),
        "IMPORTS.md §2.5 gate: the rewrite leaked outside the linker:\n{}",
        violations.join("\n")
    );
}
