# tiny-regex-c demo (backtracking recursion)

`re.c` / `re.h` are [tiny-regex-c](https://github.com/kokke/tiny-regex-c) by kokke
(**public domain / Unlicense**) — a small Rob-Pike-style regular-expression matcher (`.`, `^`,
`$`, `*`, `+`, `?`, char classes/ranges, `\d \w \s` and negations).

The files are vendored with one minimal change: the two libc includes (`<stdio.h>`, `<ctype.h>`)
and the printf-only `re_print` debug helper (not part of `re.h`'s API) are guarded behind
`#ifndef RE_FREESTANDING`, so the matcher builds for the libc-free sandbox. `regex_demo.c` defines
`RE_FREESTANDING`, supplies the three ctype predicates the library uses (`isdigit`/`isalpha`/
`isspace`), and runs a table of (pattern, text) cases, printing each match's index and length.

```sh
cargo run -p svm-run -- crates/svm-run/demos/regex/regex_demo.c
```

A new control-flow shape for the shakedown series: `re_match` recurses through
`matchpattern` → `matchstar`/`matchplus`/`matchquestion` → `matchpattern`, backtracking on
failure. That recursion-with-backtracking exercises the data-stack threading and the general
goto/branch lowering differently from the earlier integer/struct/float libraries. Output matches a
native `cc` build byte-for-byte, with no new fixes.
