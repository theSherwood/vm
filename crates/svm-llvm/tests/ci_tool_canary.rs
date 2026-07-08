//! CI canary against silent tool-skip rot (ISSUES.md, "Platform-coverage skips & caps" inventory).
//!
//! ~30 tests in this crate **auto-skip** when a pipeline tool is absent (`clang`/`cc` for the C/C++
//! corpus, `llvm-dis` for the textual reader, `llvm-as[-18]` for hand-written `.ll`, and
//! `rustc +1.81.0` / `llvm-link-18` / `opt-18` for the `peval_*` probes). That keeps contributor
//! machines unburdened — but it means that if a CI setup step ever rots (an apt package rename, a
//! rustup failure, a PATH change), the whole `svm-llvm` lane goes green while testing nothing.
//! That failure shape is not hypothetical: the TSan and ASan lanes ran *nothing* for two weeks in
//! June before anyone noticed (ISSUES.md I19/I20), because a lane that fails during setup looks
//! like a lane that passes its (never-run) tests.
//!
//! So: on CI (GitHub Actions sets `CI=true` on every runner) on Linux — the only platform whose CI
//! job installs this toolchain — every tool the auto-skips probe for must actually be runnable.
//! Anywhere else this test is a no-op.

use std::process::Command;

/// Same probe shape as `tests/common/mod.rs::tool_ok` — the canary must agree with the skips.
fn runnable(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn ci_has_every_tool_the_auto_skips_probe_for() {
    if std::env::var_os("CI").is_none() || !cfg!(target_os = "linux") {
        eprintln!(
            "note: not Linux CI — tool canary is a no-op (auto-skips stay permissive locally)"
        );
        return;
    }
    // Everything the in-crate skips (`have()`s / `toolchain_present()` / `llvm-dis` in the reader)
    // probe for, minus network fetchers (`curl`/`unzip`) and `make` (only `#[ignore]`d benches).
    let tools: &[(&str, &[&str])] = &[
        ("clang", &["--version"]),
        ("clang++", &["--version"]),
        ("cc", &["--version"]),
        ("llvm-dis", &["--version"]),
        ("llvm-as", &["--version"]),
        ("llvm-as-18", &["--version"]),
        ("llvm-link-18", &["--version"]),
        ("opt-18", &["--version"]),
        ("rustc", &["+1.81.0", "--version"]),
    ];
    let missing: Vec<&str> = tools
        .iter()
        .filter(|(cmd, args)| !runnable(cmd, args))
        .map(|(cmd, _)| *cmd)
        .collect();
    assert!(
        missing.is_empty(),
        "CI runner is missing {missing:?} — the svm-llvm tests that need these would silently \
         auto-skip, so this lane would be green while testing nothing. Fix the CI setup step \
         (ci.yml `svm-llvm` job: apt llvm-18/clang-18 + PATH, rustup 1.81.0) — do not delete this \
         canary."
    );
}
