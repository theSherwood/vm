#!/bin/sh
# Fetch-and-cache doomgeneric's sources (not vendored — id Software's Doom source under the Doom
# Source License; ~1 MB). Idempotent: skips files already present. Two transports, because the Doom
# spike must reproduce wherever the harness runs:
#   - the GitHub archive tarball (fast; what CI uses), else
#   - a per-file fetch from raw.githubusercontent.com (works where the archive host is gated),
#     with the file list from the jsDelivr package API.
#
# Usage:  sh fetch.sh [DEST]      (DEST defaults to /tmp/doomgeneric_cache/dg)
set -eu
DEST="${1:-/tmp/doomgeneric_cache/dg}"
REPO="ozkl/doomgeneric"
REF="master"
mkdir -p "$DEST"

if [ -s "$DEST/doomgeneric.h" ] && [ -s "$DEST/z_zone.c" ]; then
  echo "doomgeneric already cached in $DEST"
  exit 0
fi

# 1) archive (preferred).
TMP="$(mktemp -d)"
if curl -sSfL --max-time 120 -o "$TMP/dg.tar.gz" \
    "https://github.com/$REPO/archive/refs/heads/$REF.tar.gz" 2>/dev/null \
   && tar -xzf "$TMP/dg.tar.gz" -C "$TMP" 2>/dev/null; then
  cp "$TMP/doomgeneric-$REF/doomgeneric/"*.c "$TMP/doomgeneric-$REF/doomgeneric/"*.h "$DEST/" 2>/dev/null || true
fi
rm -rf "$TMP"

# 2) per-file fallback (raw.githubusercontent), file list from jsDelivr's package API.
if [ ! -s "$DEST/z_zone.c" ]; then
  echo "archive unavailable; fetching per-file from raw.githubusercontent.com"
  FILES="$(curl -sSf --max-time 30 \
    "https://data.jsdelivr.com/v1/packages/gh/$REPO@$REF?structure=flat" \
    | grep -oE '"/doomgeneric/[^"]+\.(c|h)"' | tr -d '"' | sort -u)"
  for f in $FILES; do
    out="$DEST/$(basename "$f")"
    [ -s "$out" ] || curl -sSf --max-time 30 \
      "https://raw.githubusercontent.com/$REPO/$REF$f" -o "$out"
  done
fi

test -s "$DEST/z_zone.c" && echo "doomgeneric cached in $DEST ($(ls "$DEST"/*.c | wc -l) .c files)"
