// Stage the **PostgreSQL playground artifacts** into `web/assets/` so the playground's "PostgreSQL"
// example (`web/play.js`) can fetch them. Unlike the on-ramp assets (Lua/SQLite), Postgres is *not*
// built here — its module is a whole `postgres --single` translated from LLVM bitcode through the
// shim + openlibm pipeline, and its data image is an `initdb`'d, cleanly-shut-down cluster. That
// pipeline is heavy and lives in `crates/svm-run/demos/postgres` + `BOOTSPEED.md` "Reproducing"; this
// script just copies the two finished artifacts from a cache dir into place.
//
//   node build-pg-assets.mjs
//     env: SVM_PG_CACHE (default /tmp/svm_pg_cache) — must hold postgres_resolved.svmb + pgdata.img
//
// The two artifacts (~20 MB module + ~40 MB image) are gitignored (`/web/assets/*.svmb`, `*.img`), so
// staging them is a local/deploy step; the playground degrades gracefully when they're absent (the
// example shows a "run build-pg-assets.mjs" hint instead of booting).
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync, mkdirSync, copyFileSync, statSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, 'web', 'assets');
const CACHE = process.env.SVM_PG_CACHE ?? '/tmp/svm_pg_cache';
mkdirSync(ASSETS, { recursive: true });

const artifacts = [
  ['postgres_resolved.svmb', 'the pre-translated + resolved Postgres module'],
  ['pgdata.img', 'the cleanly-shut-down data image'],
];

let staged = 0;
for (const [name, what] of artifacts) {
  const src = join(CACHE, name);
  if (!existsSync(src)) {
    console.log(`  ✗ ${name} missing in ${CACHE} — ${what}; see BOOTSPEED.md "Reproducing" to build it`);
    continue;
  }
  const dst = join(ASSETS, name);
  copyFileSync(src, dst);
  const mb = (statSync(dst).size / 1e6).toFixed(1);
  console.log(`  ✓ ${name} (${mb} MB) — ${what}`);
  staged++;
}

if (staged < artifacts.length) {
  console.log(`staged ${staged}/${artifacts.length} — the playground's Postgres example needs both to boot`);
  process.exit(staged === 0 ? 0 : 0); // best-effort: absence is not a hard error (the example degrades)
} else {
  console.log('staged both Postgres artifacts — the playground example is ready');
}
