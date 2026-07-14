# Doom — the "wow" milestone (LLVM.md §7, slices toward Doom)

Running **shareware Doom** (via [doomgeneric](https://github.com/ozkl/doomgeneric)) through the
LLVM→SVM-IR on-ramp, in the browser playground, driven by the reactor run model. This directory holds
the **feasibility spike (slice 3b)** — the platform layer, the fetch tooling, and the precisely
quantified gap list — established before the full bring-up.

doomgeneric is deliberately portable: all platform code is six `DG_*` hooks plus the `DG_ScreenBuffer`
it draws into, and `w_file_stdc.c` reads the WAD through stock C file I/O. That maps cleanly onto the
capabilities the playground already exposes (slices 1–3a): `display` out, `keyboard` in, the `fs`
capability for the WAD, and a persistent multi-MB heap for Doom's zone allocator.

## What's here

- **`doomgeneric_svm.c`** — the platform layer (real, compiles against doomgeneric's headers). `DG_*`
  onto the caps: `DG_DrawFrame` swizzles the XRGB `DG_ScreenBuffer` to RGBA and presents it through
  `display`; `DG_GetKey` polls `keyboard` (browser keyCodes → Doom key codes); `DG_GetTicksMs` is a
  **deterministic** frame clock (fixed ms/frame — no wall-clock, so the differential is reproducible).
- **`main.c`** — the reactor entry: `main()` runs `doomgeneric_Create(-iwad doom1.wad)` once at
  `_start` (reading the WAD via `fs`); the exported `tick()` calls `doomgeneric_Tick()` once per frame
  over the persistent window (slice 3a keeps the zone heap alive between frames).
- **`fetch.sh`** — fetch-and-cache doomgeneric's sources (not vendored — id Software's Doom source
  under the Doom Source License; ~1 MB). CI uses the GitHub archive; a `raw.githubusercontent.com`
  per-file fallback works where the archive host is unavailable.

## Spike findings — Doom is a libc shim + ONE translator feature away

Reproduced with the fetched sources + the platform here (`fetch.sh` documents the exact commands):

1. **Compiles clean.** All 79 Doom translation units build to LLVM-18 bitcode with stock
   `clang -O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize -DNORMALUNIX -DLINUX` — 79/79, zero
   compile errors. (The X11 platform `doomgeneric_xlib.c` is replaced by `doomgeneric_svm.c`.)
2. **Links** into one ~900 KB module (`llvm-link`).
3. **Translates the whole program** under `svm-llvm-translate --stub-externs` (libc stubbed) except
   for **one** unsupported IR construct — no SIMD / `i128` / inline-asm / vector-memory walls like the
   Postgres spike hit:
   - **Indirect calls through an unprototyped (`void (...)`, K&R-style) function pointer.** Doom
     declares several callback pointers with empty parens — `d_think.h`'s `actionf_v` (`typedef void
     (*actionf_v)()`), `d_loop.c`'s `loop_interface_t`, and `m_menu.c`'s menu `routine`s — which LLVM
     types as `void (...)`. They are called with **concrete** args (0 here), never true varargs; the
     on-ramp currently rejects them (`Unsupported("indirect varargs call")`, first seen in
     `@NetUpdate`/`@TryRunTics`/`@M_Drawer`).
4. **Memory.** The framebuffer is 640×400×4 ≈ 1 MB and Doom's zone allocator takes several MB — all in
   the `malloc` heap **above** the mapped window, which the slice-3a persistent reactor now keeps live
   across frames. This is exactly why slice 3a came first.

## Remaining work (the next sub-slices)

- **Translator feature** *(the one IR gap)*: lower an indirect call through a `void (...)` /
  unprototyped function-pointer type using the **call site's concrete signature** (these are never
  true varargs in Doom). Well-scoped, and it generally unblocks K&R-style callbacks. This is the
  single blocker to translating Doom with a stubbed libc.
- **libc shim** (~35 functions, modeled on the Lua/SQLite guest shims): the string/ctype set
  (`strlen`/`strcmp`/`strncmp`/`strncpy`/`strrchr`/`strstr`/`memchr`/`bcmp`/`strcasecmp`/
  `strncasecmp`/`strdup`/`__ctype_toupper_loc`); `stdlib` (`exit`/`strtol`/`strtod`/`atoi`); the
  `printf` family (`printf`/`fprintf`/`snprintf`/`vsnprintf`/`vfprintf` → the `lua_fmt_snprintf.c`
  format-engine model + stdout via `Stream`); file I/O (`fopen`/`fread`/`fseek`/`ftell`/`fclose`/
  `fwrite` → the `fs` capability, the `lua_files_stdio.c` model, for the WAD); `sscanf` (config
  parsing); and two netgame stubs (`drone`, `net_client_connected`). `malloc`/`calloc`/`realloc`/
  `free`/`memcpy`/`memset` are synthesized by the on-ramp already.
- **The WAD** (`doom1.wad`, freely distributable shareware) served through the `fs` capability, and a
  **headless differential**: run N frames of the guest and a native `cc` build with the same
  deterministic clock + input script, and assert the per-frame framebuffer hashes match byte-for-byte
  (the interpreter-is-the-oracle contract, §18).
- **Playground wiring** (slice 4): the `.svmb` + WAD as assets, the reactor loop + keyboard already in
  `play.js`.
