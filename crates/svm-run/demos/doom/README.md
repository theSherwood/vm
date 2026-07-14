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
- **`doom_libc.c`** — the freestanding libc the on-ramp doesn't synthesize and the reused Lua shims
  don't already cover: the string/ctype set, `stdlib` (`atoi`/`strtol`/`strtod`/`abs`/`system`/
  `mkdir`), `printf`/`puts`/`vfprintf`, a single-integer `sscanf` (Doom's config only uses `%d`/`%i`/
  `%x`/`%o`), and the two netgame stubs (`drone`, `net_client_connected`).
- **`build.sh`** — compile the fetched sources + platform + shim, link, and translate to `doom.svmb`.
- **`fetch.sh`** — fetch-and-cache doomgeneric's sources (not vendored — id Software's Doom source
  under the Doom Source License; ~1 MB). CI uses the GitHub archive; a `raw.githubusercontent.com`
  per-file fallback works where the archive host is unavailable.

The reused Lua guest shims (`crates/svm-llvm/tests/fixtures/lua/`) do the heavy lifting:
`lua_files_stdio.c` is the `FILE`-over-`fs`-capability layer (fopen/fread/fseek/ftell/fclose/fprintf/
errno/std streams — the WAD read path), and `lua_fmt_snprintf.c` is the printf format engine
(snprintf/vsnprintf, with the on-ramp's `__vm_fmt_*` Dragon4 float formatters).

## Status — Doom translates and BOOTS in the sandbox

Reproduced with `sh fetch.sh && sh build.sh` (the on-ramp translator built first):

1. **Compiles clean.** All 79 Doom TUs build to LLVM-18 bitcode with stock `clang -O2 -emit-llvm`
   (79/79, zero errors; the X11 platform is replaced by `doomgeneric_svm.c`).
2. **Translates whole-program — no IR gaps.** `llvm-link` → one ~900 KB module → `svm-llvm-translate`
   produces a **797 KB `doom.svmb`** with `main`/`tick` exported. There are **no unsupported IR
   constructs** — no SIMD / `i128` / inline-asm / vector-memory walls (unlike the Postgres spike), and
   the on-ramp already handles indirect calls through unprototyped (`void (...)`, K&R) function
   pointers (Doom's `actionf_v`/`loop_interface_t`/menu callbacks). *(An earlier spike using a stale
   translator binary reported that as a gap; a current build translates it.)* Every remaining
   unresolved symbol is on-ramp-provided (`malloc`/`calloc`/`realloc`/`free`, `read`/`write`/`exit`,
   `__vm_*`).
3. **Boots.** Driven through a reactor (the slice-3a persistent instance) with the powerbox + a
   `display` cap + a `keyboard` cap + an `fs` cap serving the shareware `doom1.wad`, `_start` →
   `doomgeneric_Create` runs Doom's **entire** initialization on the bytecode interpreter — the real
   startup log:

   ```
   Z_Init: Init zone memory allocation daemon.
   zone memory: 0x4fa050, 600000 allocated for zone
   W_Init: Init WADfiles.  /  adding doom1.wad
                               DOOM Shareware
   I_Init / M_Init / R_Init: Init DOOM refresh daemon - ...
   P_Init / S_Init / D_CheckNetGame / HU_Init / ST_Init
   ```

   The WAD is parsed (through the `fs` cap), the zone allocator runs in the persistent heap, the
   renderer builds its data (`R_Init`), and the game reaches the main loop — so the libc shim is
   correct end to end (`fread`/`fseek` on the WAD, `sscanf` config parsing, `printf`, the `malloc`
   zone, the string set). This proves the "Doom runs sandboxed through the LLVM on-ramp" thesis.

4. **Memory.** The 640×400×4 ≈ 1 MB framebuffer + the multi-MB zone live in the `malloc` heap **above**
   the mapped window, which the slice-3a persistent reactor keeps live across frames — why 3a came
   first.

## Remaining work (next sub-slices)

- **A rendered frame + the headless differential.** A full 640×400 software-rendered frame is billions
  of instructions; on the **bytecode interpreter** that's ~minutes/frame (`_start` alone burned 20 B
  instructions reaching the main loop). The frame render + a byte-exact differential vs a native `cc`
  build (same deterministic clock + input script, per-frame framebuffer hashes — the §18 oracle) want
  the **JIT** for speed. That's the substance of the next slice.
- **Playground wiring** (slice 4): the `.svmb` + WAD as assets; grant the `fs` cap (with the WAD) in
  the browser reactor; the reactor loop + `keyboard` are already in `play.js`.
