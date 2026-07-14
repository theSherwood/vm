#!/bin/sh
# Build the headless frame-hash differential and run the native oracle (Doom slice 3c). Produces the
# artifacts the `svm-run` test `doom_diff` compares:
#   - the guest  $OUT/bcdiff/doom_diff.svmb   (doom_diff.c through the on-ramp)
#   - the native $OUT/native/native_frames.txt (doom_diff.c built with `cc`, run over doom1.wad)
# Both use the SAME doom_diff.c (a frame-hash-printing platform + a DOOM_DIFF_FRAMES loop) + a
# deterministic virtual clock, so their per-frame framebuffer hashes match iff the guest renders
# byte-identically to native — the §18 oracle for Doom's whole fixed-point renderer.
#
# Prereqs: `clang`/`cc`/`llvm-link` + the on-ramp translator built + `sh fetch.sh` run, plus the
# freely-distributable shareware WAD at $CACHE/doom1.wad. Usage: sh diff.sh [FRAMES] [CACHE]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
FRAMES="${1:-200}"          # 200 reaches demo1 (E1M1) gameplay — dynamic 3D frames, the strong proof
CACHE="${2:-/tmp/doomgeneric_cache}"
SRC="$CACHE/dg"
LUA="$REPO/crates/svm-llvm/tests/fixtures/lua"
TR="$REPO/crates/svm-llvm/target/release/svm-llvm-translate"
FL="-O2 -w -DNORMALUNIX -DLINUX -DDOOM_DIFF_FRAMES=$FRAMES"
SRCS="am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net f_finale \
f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound i_system i_timer memio \
m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc m_random p_ceilng p_doors p_enemy \
p_floor p_inter p_lights p_map p_maputl p_mobj p_plats p_pspr p_saveg p_setup p_sight p_spec p_switch \
p_telept p_tick p_user r_bsp r_data r_draw r_main r_plane r_segs r_sky r_things sha1 sounds statdump \
st_lib st_stuff s_sound tables v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc \
i_input i_video doomgeneric"

test -s "$CACHE/doom1.wad" || { echo "missing $CACHE/doom1.wad (shareware IWAD) — fetch it first"; exit 1; }

# --- guest: doom_diff.c through the on-ramp ---------------------------------------------------------
G="$CACHE/bcdiff"; mkdir -p "$G"; rm -f "$G"/*.bc
for s in $SRCS; do clang $FL -emit-llvm -c -fno-vectorize -fno-slp-vectorize -I"$SRC" "$SRC/$s.c" -o "$G/$s.bc"; done
clang $FL -emit-llvm -c -fno-vectorize -fno-slp-vectorize -I"$SRC" "$HERE/doom_diff.c" -o "$G/_diff.bc"
clang $FL -emit-llvm -c -fno-vectorize -fno-slp-vectorize          "$HERE/doom_libc.c" -o "$G/_libc.bc"
clang $FL -emit-llvm -c -fno-vectorize -fno-slp-vectorize "$LUA/lua_fmt_snprintf.c" -o "$G/_fmt.bc"
clang $FL -emit-llvm -c -fno-vectorize -fno-slp-vectorize "$LUA/lua_files_stdio.c"  -o "$G/_files.bc"
llvm-link "$G"/*.bc -o "$G/doom_diff.linked.bc"
"$TR" "$G/doom_diff.linked.bc" -o "$G/doom_diff.svmb" --host-page 65536
echo "guest: $G/doom_diff.svmb ($(wc -c < "$G/doom_diff.svmb") bytes)"

# --- native oracle: doom_diff.c with `cc` (+ the netgame stubs the guest gets from doom_libc.c) -----
N="$CACHE/native"; mkdir -p "$N"; rm -f "$N"/*.o
for s in $SRCS; do cc $FL -I"$SRC" -c "$SRC/$s.c" -o "$N/$s.o"; done
cc $FL -I"$SRC" -c "$HERE/doom_diff.c" -o "$N/doom_diff.o"
printf 'int drone=0;\nint net_client_connected=0;\n' > "$N/stubs.c"; cc -O2 -c "$N/stubs.c" -o "$N/stubs.o"
cc "$N"/*.o -lm -o "$N/doom_native"
( cd "$CACHE" && "$N/doom_native" 2>/dev/null | grep '^frame ' ) > "$N/native_frames.txt"
echo "native: $N/native_frames.txt ($(wc -l < "$N/native_frames.txt") frames, \
$(awk '{print $3}' "$N/native_frames.txt" | sort -u | wc -l) unique hashes)"

echo "now run:  cargo test -p svm-run --test doom_diff -- --ignored --nocapture"
