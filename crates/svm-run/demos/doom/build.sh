#!/bin/sh
# Build the Doom guest for the SVM sandbox: compile the fetched doomgeneric sources + this platform
# layer + the libc shim, link to one module, and translate through the LLVM on-ramp to a `.svmb`.
# Reproduces the slice-3b result — Doom translates with a complete libc shim and boots in the sandbox.
#
# Prereqs: `clang`/`llvm-link` on PATH, the on-ramp translator built
# (`cargo build --release --bin svm-llvm-translate` in crates/svm-llvm), and the sources fetched
# (`sh fetch.sh`). Usage:  sh build.sh [SRC_CACHE] [OUT_DIR]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
SRC="${1:-/tmp/doomgeneric_cache/dg}"
OUT="${2:-/tmp/doomgeneric_cache/bc}"
LUA="$REPO/crates/svm-llvm/tests/fixtures/lua"           # reused fs + printf guest shims
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
mkdir -p "$OUT"; rm -f "$OUT"/*.bc
FL="-O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize -DNORMALUNIX -DLINUX"

# All Doom TUs from doomgeneric's Makefile SRC_DOOM, MINUS the X11 platform (doomgeneric_xlib) which
# `doomgeneric_svm.c` replaces.
SRCS="am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale \
f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio \
m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy \
p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch \
p_telept p_tick p_user r_bsp r_data r_draw r_main r_plane r_segs r_sky r_things sha1 sounds statdump \
st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc \
i_input i_video doomgeneric"
for s in $SRCS; do clang $FL -I"$SRC" "$SRC/$s.c" -o "$OUT/$s.bc"; done

# Platform + reactor entry + the libc shim. The reused Lua shims provide the FILE-over-`fs` layer
# (fopen/fread/fseek/ftell/fclose/fprintf/errno/std streams) and the printf format engine
# (snprintf/vsnprintf); doom_libc.c adds string/ctype/stdlib/sscanf + the netgame stubs.
clang $FL -I"$SRC" "$HERE/doomgeneric_svm.c" -o "$OUT/_platform.bc"
clang $FL -I"$SRC" "$HERE/main.c"            -o "$OUT/_main.bc"
clang $FL          "$HERE/doom_libc.c"       -o "$OUT/_libc.bc"
clang $FL "$LUA/lua_fmt_snprintf.c"          -o "$OUT/_fmt.bc"
clang $FL "$LUA/lua_files_stdio.c"           -o "$OUT/_files.bc"

llvm-link "$OUT"/*.bc -o "$OUT/doom.bc"
# 64 KiB host page (the wasm/browser page); emit the export map so a driver can resolve `tick`/`main`.
"$TR" "$OUT/doom.bc" -o "$OUT/doom.svmb" --host-page 65536 --emit-syms "$OUT/doom.syms"
echo "built $OUT/doom.svmb ($(wc -c < "$OUT/doom.svmb") bytes); exports:"
grep -E '^(main|tick) ' "$OUT/doom.syms"
