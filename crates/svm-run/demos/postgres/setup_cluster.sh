#!/usr/bin/env bash
# Build the **demo data cluster** the browser image ships — the previously-manual half of the pipeline
# (`build_bitcode.sh` builds the guest *bitcode*; this builds the *data dir*). Runs the native
# `initdb`/`postgres` that `build_bitcode.sh` produced (`make install` into `$SVM_PG_CACHE/inst`),
# applies the sandbox-friendly config, mirrors the install `share/` tree where the fs cap can reach it,
# and boots once via `--single` so the on-disk cluster is **cleanly shut down** (no WAL recovery on the
# guest's first boot — the BOOTSPEED lever). The result is `$SVM_PG_DATA`, which `build_image` turns
# into the shippable `pgdata.img`.
#
#   needs: a native `initdb`/`postgres` installed at $INST (build_bitcode.sh + `make install`)
#   env:   SVM_PG_CACHE (default /tmp/svm_pg_cache), SVM_PG_DATA (default $SVM_PG_CACHE/pgdata)
#   note:  initdb refuses to run as root — run as an unprivileged user (CI runners already are).
set -euo pipefail
CACHE="${SVM_PG_CACHE:-/tmp/svm_pg_cache}"
INST="$CACHE/inst"
DATA="${SVM_PG_DATA:-$CACHE/pgdata}"

[ -x "$INST/bin/initdb" ] || { echo "no initdb at $INST/bin — run build_bitcode.sh + 'make install' first"; exit 31; }

echo "=== [1/4] initdb (trust auth, no fsync) ==="
rm -rf "$DATA"
"$INST/bin/initdb" -D "$DATA" --no-sync -U postgres -A trust >/dev/null

echo "=== [2/4] sandbox config (GMT, sysv DSM, fsync off, tiny buffers) ==="
# One identity, no cross-process IPC: GMT (parsed, no tz-data file), SysV DSM, no startup data-dir
# sync, small shmem/buffers (cheap init on the interpreter). Matches README "Demo cluster setup".
cat >> "$DATA/postgresql.conf" <<'CONF'
timezone = 'GMT'
log_timezone = 'GMT'
dynamic_shared_memory_type = sysv
fsync = off
shared_buffers = 1MB
max_connections = 10
CONF

echo "=== [3/4] find_my_exec marker + mirror the install share/ tree ==="
# argv[0] = "./postgres": a slashed argv0 + an executable `postgres` in the data dir so `find_my_exec`
# resolves. It is never executed (the SVM module is), so a bare shell stub suffices.
printf '#!/bin/sh\n' > "$DATA/postgres"; chmod +x "$DATA/postgres"
# The guest opens its compiled-in `$INST/share/postgresql` (timezonesets, encodings, …) by absolute
# path; the fs cap roots "/" at the data dir, so mirror the tree at `$DATA` + the absolute prefix.
mkdir -p "$DATA$INST/share"
cp -r "$INST/share/postgresql" "$DATA$INST/share/"

echo "=== [4/4] one --single run → clean shutdown checkpoint (no WAL recovery on guest boot) ==="
printf 'SELECT 1;\n' | "$INST/bin/postgres" --single -D "$DATA" -O -j postgres >/dev/null
echo "cluster ready: $DATA ($(du -sh "$DATA" | cut -f1))"
