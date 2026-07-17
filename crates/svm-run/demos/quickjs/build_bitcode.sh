#!/usr/bin/env bash
# Build the linked QuickJS-eval bitcode for the on-ramp, the exact pipeline the
# `demo_quickjs_eval_vs_native` test automates. Fetched-not-vendored: point
# QJS_DIR at an extracted Bellard QuickJS tree (2024-01-13), or let it fetch.
#
#   ./build_bitcode.sh            # fetch + build → /tmp/qjs_bc/qjs_linked.bc
#   QJS_DIR=/path/to/quickjs ./build_bitcode.sh
#
# The math object takes the *address* of the libm transcendentals
# (`js_math_funcs`: fabs/sin/cos/pow/atan2/log/cbrt/hypot/…), so a real guest
# libm must be linked for those funcrefs to resolve — openlibm, exactly as the
# Postgres capstone does (LLVM.md slice CO). Set OPENLIBM_DIR to a staged tree.
set -euo pipefail

VER=2024-01-13
CACHE=${CACHE:-/tmp/svm_quickjs_cache}
QJS_DIR=${QJS_DIR:-$CACHE/quickjs-$VER}
OUT=${OUT:-/tmp/qjs_bc}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$CACHE" "$OUT"
if [ ! -f "$QJS_DIR/quickjs.c" ]; then
  echo "fetching QuickJS $VER …"
  curl -sfL --max-time 120 -o "$CACHE/quickjs-$VER.tar.xz" \
    "https://bellard.org/quickjs/quickjs-$VER.tar.xz"
  tar xf "$CACHE/quickjs-$VER.tar.xz" -C "$CACHE"
fi

# On-ramp compile flags: -O2, no vectorization (the capstone convention),
# NDEBUG to drop the `assert()`/`__assert_fail` libc surface.
CFLAGS=(-O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize -DNDEBUG \
        -D_GNU_SOURCE "-DCONFIG_VERSION=\"$VER\"" -I"$QJS_DIR")

echo "compiling QuickJS TUs → bitcode …"
for f in quickjs libregexp libunicode cutils libbf; do
  clang "${CFLAGS[@]}" "$QJS_DIR/$f.c" -o "$OUT/$f.bc"
done
clang -O2 -emit-llvm -c -fno-vectorize -fno-slp-vectorize -DNDEBUG -D_GNU_SOURCE \
  -I"$QJS_DIR" "$HERE/qjs_eval.c" -o "$OUT/driver.bc"

echo "linking …"
llvm-link "$OUT"/{driver,quickjs,libregexp,libunicode,cutils,libbf}.bc \
  -o "$OUT/qjs_linked.bc"
echo "→ $OUT/qjs_linked.bc"
