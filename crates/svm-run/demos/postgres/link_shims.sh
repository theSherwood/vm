#!/usr/bin/env bash
# Link the guest shims (os/libc/locale/time/proc/stdio) plus the bundled guest libm (openlibm) into
# the cached whole-program Postgres bitcode, producing `postgres_shimmed.bc` — the module the boot
# driver translates + runs. Run after build_bitcode.sh has produced `postgres_libm.bc`. See README
# "Booting" for the driver.
set -euo pipefail
CACHE="${SVM_PG_CACHE:-/tmp/svm_pg_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clang -O2 -emit-llvm -c -DSVM_GUEST -fno-vectorize -fno-slp-vectorize \
  -fno-builtin-memcpy -fno-builtin-memmove -fno-builtin-memset -fno-builtin-memcmp \
  -fno-builtin-strlen -fno-builtin-strcmp -fno-builtin-strncmp -I"$HERE" \
  "$HERE/pg_shims.c" -o "$CACHE/pg_shims.bc"
# `strerror_shim.c` is compiled *alone* with -D_GNU_SOURCE (the GNU `char *strerror_r`), isolated from
# the shared TU so it doesn't perturb `__isoc23_*`/`getrlimit`/… (see strerror_shim.c).
clang -O2 -emit-llvm -c -DSVM_GUEST -D_GNU_SOURCE -fno-vectorize -fno-slp-vectorize -I"$HERE" \
  "$HERE/strerror_shim.c" -o "$CACHE/strerror_shim.bc"

# --- Bundled guest libm (slice BQ). The SVM has no transcendental op, so `log`/`exp`/`pow`/`sin`/… stay
# guest code — a *real* libm (openlibm), llvm-linked in (bit-exact vs native, `libm_bundled_vs_native`).
# Without it every transcendental is an undefined external → a fail-closed trap stub; the query cost
# model (`cost_tuplesort` → `LOG2` → `log`) is the first path to reach one. `sqrt`/`fabs`/`ceil`/… the
# openlibm code calls resolve to on-ramp float ops. Fetched-not-vendored (BSD).
OL_VER="${SVM_OPENLIBM_VER:-0.8.5}"
OL_CACHE="${SVM_OPENLIBM_CACHE:-/tmp/svm_openlibm_cache}"
# Locate an openlibm source tree: an explicit override, a pre-staged tree, or the versioned fetch
# cache; only fetch from github if none is present (some environments block github egress).
OL_DIR=""
for cand in "${SVM_OPENLIBM_DIR:-}" /tmp/openlibm "$OL_CACHE/openlibm-$OL_VER"; do
  [ -n "$cand" ] && [ -f "$cand/src/e_log.c" ] && { OL_DIR="$cand"; break; }
done
if [ -z "$OL_DIR" ]; then
  OL_DIR="$OL_CACHE/openlibm-$OL_VER"
  mkdir -p "$OL_CACHE"
  curl -sfL --max-time 120 -o "$OL_CACHE/openlibm.tar.gz" \
    "https://github.com/JuliaMath/openlibm/archive/refs/tags/v$OL_VER.tar.gz" \
    || { echo "OPENLIBM FETCH FAILED (set SVM_OPENLIBM_DIR to a local tree)"; exit 21; }
  tar xf "$OL_CACHE/openlibm.tar.gz" -C "$OL_CACHE"
fi
echo "openlibm: $OL_DIR"
# The double-precision entry points + kernels (18 transcendentals; matches the `libm_bundled_vs_native`
# OPENLIBM_SRCS set). `pow`/`fmod`/… come from here now — not a hand-written shim.
OL_SRCS="e_log e_log10 e_log2 e_exp s_exp2 e_pow s_sin s_cos s_tan k_sin k_cos k_tan \
  e_rem_pio2 k_rem_pio2 e_asin e_acos s_atan e_atan2 e_sinh e_cosh s_tanh s_cbrt e_fmod \
  s_scalbn s_copysign s_fabs k_exp s_expm1"
OL_BCS=""
for name in $OL_SRCS; do
  clang -O2 -fno-vectorize -fno-slp-vectorize -DASSEMBLER=0 \
    -I"$OL_DIR" -I"$OL_DIR/include" -I"$OL_DIR/src" -I"$OL_DIR/amd64" \
    -emit-llvm -c "$OL_DIR/src/$name.c" -o "$CACHE/ol_$name.bc" \
    || { echo "OPENLIBM COMPILE FAILED: $name"; exit 22; }
  OL_BCS="$OL_BCS $CACHE/ol_$name.bc"
done

llvm-link "$CACHE/postgres_libm.bc" "$CACHE/pg_shims.bc" "$CACHE/strerror_shim.bc" $OL_BCS \
  -o "$CACHE/postgres_shimmed.bc"
echo "linked: $(stat -c%s "$CACHE/postgres_shimmed.bc") bytes -> $CACHE/postgres_shimmed.bc"
