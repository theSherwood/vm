#!/usr/bin/env bash
# Postgres `--single` on the LLVM on-ramp — the whole-program bitcode pipeline (slice BM).
# Fetch → configure → native oracle → per-TU bitcode → llvm-link the exact `postgres` link set →
# translate through the on-ramp. Fetched-not-vendored (PostgreSQL license). See README.md.
#
#   needs: clang-18, llvm-dis, llvm-link, flex, bison, perl, make, curl
#   env:   SVM_PG_CACHE (default /tmp/svm_pg_cache), SVM_PG_VER (default 17.5)
set -uo pipefail
CACHE="${SVM_PG_CACHE:-/tmp/svm_pg_cache}"
VER="${SVM_PG_VER:-17.5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$CACHE/postgresql-$VER"
PREFIX="$CACHE/inst"
export SVM_PG_SRC="$SRC" SVM_PG_CACHE="$CACHE"
mkdir -p "$CACHE"; cd "$CACHE"

echo "=== [1/6] fetch postgresql-$VER ==="
TB="postgresql-$VER.tar.bz2"
[ -f "$TB" ] || curl -sfL --max-time 300 -o "$TB" \
  "https://ftp.postgresql.org/pub/source/v$VER/$TB" || { echo "FETCH FAILED"; exit 11; }
[ -d "$SRC" ] || tar xf "$TB"

echo "=== [2/6] configure (minimal, clang; AVX-512 popcount off) ==="
cd "$SRC"
# Build-config lever (slice BV): force the AVX-512 popcount autodetect to "no" so configure
# leaves PG_POPCNT_OBJS empty and never defines USE_AVX512_POPCNT_WITH_RUNTIME_CHECK — i.e. the
# exact config a host lacking AVX-512 would produce. This drops `pg_popcount_avx512` /
# `pg_popcount_masked_avx512` (and their `<64 x i1>` AVX-512 vector bodies) from the link set at
# the source. On the guest these fast paths are *dead* anyway — the runtime `cpuid`→0 makes
# `pg_popcount_avx512_available()` false, so the scalar popcount is always chosen; numeric
# behavior is identical, so the native oracle stays a valid differential target.
[ -f config.status ] || ./configure CC=clang --prefix="$PREFIX" \
  --without-icu --without-readline --without-zlib --without-zstd --without-lz4 \
  --without-libxml --without-gssapi --disable-nls \
  pgac_cv_avx512_popcnt_intrinsics_=no \
  pgac_cv_avx512_popcnt_intrinsics__mavx512vpopcntdq__mavx512bw=no 2>&1 | tail -5
[ -f config.status ] || { echo "CONFIGURE FAILED"; exit 12; }

echo "=== [3/6] native oracle build ==="
make -j"$(nproc)" -s 2>&1 | tail -4
[ -f src/backend/postgres ] || { echo "BUILD FAILED"; exit 13; }
echo "postgres: $(stat -c%s src/backend/postgres) bytes"

echo "=== [4/6] per-TU bitcode (reuses makefile flags) ==="
python3 "$HERE/emit_bc.py"

echo "=== [5/6] llvm-link the exact postgres link set ==="
cd "$SRC/src/backend"; rm -f postgres
# The authoritative object list = the `postgres` link command; plus the two _srv archives' members.
make -n postgres 2>/dev/null | grep -oE '[A-Za-z0-9_./-]+\.o' | sort -u \
  | sed "s#^#$SRC/src/backend/#" | xargs -I{} realpath -m {} \
  | sed 's/\.o$/.bc/' > "$CACHE/link_bc.txt"
for a in src/common/libpgcommon_srv.a src/port/libpgport_srv.a; do
  ar t "$SRC/$a" 2>/dev/null | sed "s#^#$SRC/$(dirname "$a")/#" | sed 's/\.o$/.bc/'
done >> "$CACHE/link_bc.txt"
sort -u "$CACHE/link_bc.txt" | while read -r p; do [ -f "$p" ] && echo "$p"; done > "$CACHE/bcset.txt"
echo "linking $(wc -l < "$CACHE/bcset.txt") modules"
llvm-link $(cat "$CACHE/bcset.txt") -o "$CACHE/postgres.linked.bc" 2>"$CACHE/llvm-link.err" \
  || { echo "LINK FAILED:"; tail -3 "$CACHE/llvm-link.err"; exit 15; }
llvm-dis "$CACHE/postgres.linked.bc" -o "$CACHE/postgres.linked.ll"
echo "linked: $(stat -c%s "$CACHE/postgres.linked.bc") bytes .bc / $(stat -c%s "$CACHE/postgres.linked.ll") bytes .ll"

echo "=== [6/6] translate through the on-ramp (expect a fail-closed gap) ==="
TR="$HERE/../../../svm-llvm/target/release/svm-llvm-translate"
[ -x "$TR" ] || (cd "$HERE/../../../svm-llvm" && cargo build --release --bin svm-llvm-translate 2>&1 | tail -1)
"$TR" "$CACHE/postgres.linked.bc" -o "$CACHE/postgres.svm" 2>"$CACHE/translate.err"
echo "first gap: $(cat "$CACHE/translate.err")"
