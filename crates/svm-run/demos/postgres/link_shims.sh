#!/usr/bin/env bash
# Link the guest shims (os/libc/locale/time/proc/stdio) into the cached whole-program Postgres
# bitcode, producing `postgres_shimmed.bc` — the module the boot driver translates + runs. Run
# after build_bitcode.sh has produced `postgres_libm.bc`. See README "Booting" for the driver.
set -euo pipefail
CACHE="${SVM_PG_CACHE:-/tmp/svm_pg_cache}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clang -O2 -emit-llvm -c -DSVM_GUEST -fno-vectorize -fno-slp-vectorize -I"$HERE" \
  "$HERE/pg_shims.c" -o "$CACHE/pg_shims.bc"
llvm-link "$CACHE/postgres_libm.bc" "$CACHE/pg_shims.bc" -o "$CACHE/postgres_shimmed.bc"
echo "linked: $(stat -c%s "$CACHE/postgres_shimmed.bc") bytes -> $CACHE/postgres_shimmed.bc"
