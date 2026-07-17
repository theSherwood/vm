// Build the **PostgreSQL playground artifacts** and stage them into `web/assets/`, the way the deployed
// playground stays in sync with the code (no pre-built blob to drift). Two halves:
//
//   • CODE-INDEPENDENT INPUTS (cached; rebuilt only when Postgres/​the shims change) — the whole-program
//     Postgres bitcode (`postgres_shimmed.bc`, via `build_bitcode.sh` + `link_shims.sh`) and the demo
//     data cluster (`setup_cluster.sh`). These depend on the *Postgres version* + the shim/openlibm
//     sources, NOT on the SVM translator/verifier/encoder, so a build cache keyed on those inputs is
//     sound. Building them cold is expensive (compiles Postgres from source, ~10-15 min); this script
//     runs that only when the cached output is absent.
//
//   • CODE-COUPLED OUTPUTS (regenerated every run) — `postgres_resolved.svmb` (`svm-llvm-translate` +
//     `prep_svmb`) and `pgdata.img` (`build_image`). These ARE tied to current SVM code, so they are
//     always regenerated from the cached inputs — the shipped artifacts can never silently diverge from
//     the translator/encoder that produced them, and `browser-test.mjs` boots them as a drift guard.
//
// Usage:  node build-pg-assets.mjs
//   env:  SVM_PG_CACHE (default /tmp/svm_pg_cache)   — the input cache (bitcode + cluster + native pg)
//         SVM_PG_DATA   (default $SVM_PG_CACHE/pgdata) — the demo cluster the image is built from
//         SVM_PG_VER    (default 17.5)                — Postgres version (passed to build_bitcode.sh)
// Needs (only on a cold input cache): clang-18/llvm-18, flex, bison, perl, make, curl, and a NON-root
// user (initdb refuses root). The regenerate half needs only cargo + the workspace.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync, mkdirSync, copyFileSync, statSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = dirname(HERE);
const ASSETS = join(HERE, 'web', 'assets');
const CACHE = process.env.SVM_PG_CACHE ?? '/tmp/svm_pg_cache';
const DATA = process.env.SVM_PG_DATA ?? join(CACHE, 'pgdata');
const PGDIR = join(REPO, 'crates', 'svm-run', 'demos', 'postgres');
const INST = join(CACHE, 'inst');
mkdirSync(ASSETS, { recursive: true });

const env = { ...process.env, SVM_PG_CACHE: CACHE, SVM_PG_DATA: DATA };
const sh = (cmd, args, opts = {}) =>
  execFileSync(cmd, args, { stdio: 'inherit', env, cwd: REPO, ...opts });
const mb = (p) => (statSync(p).size / 1e6).toFixed(1);

// ---- 1) code-independent input: the whole-program Postgres bitcode (cached) --------------------
const shimmedBc = join(CACHE, 'postgres_shimmed.bc');
if (existsSync(shimmedBc)) {
  console.log(`✓ cached bitcode: postgres_shimmed.bc (${mb(shimmedBc)} MB)`);
} else {
  console.log('building Postgres bitcode from source (cold — ~10-15 min)…');
  // fetch → configure → native build → per-TU bitcode → llvm-link the `postgres` link set → translate
  sh('bash', [join(PGDIR, 'build_bitcode.sh')]);
  // Install the native build so `setup_cluster.sh` has `initdb`/`postgres` + the `share/` tree, and the
  // compiled-in `--prefix` ($INST) the guest resolves at runtime is populated.
  const src = join(CACHE, `postgresql-${process.env.SVM_PG_VER ?? '17.5'}`);
  sh('make', ['-C', src, 'install'], { stdio: 'inherit' });
  // `link_shims.sh` reads `postgres_libm.bc`; `build_bitcode.sh` emits the same module as
  // `postgres.linked.bc`. Bridge the name, then link the guest shims + bundled openlibm.
  copyFileSync(join(CACHE, 'postgres.linked.bc'), join(CACHE, 'postgres_libm.bc'));
  sh('bash', [join(PGDIR, 'link_shims.sh')]);
  console.log(`✓ built bitcode: postgres_shimmed.bc (${mb(shimmedBc)} MB)`);
}

// ---- 2) code-independent input: the demo data cluster (cached) ---------------------------------
if (existsSync(join(DATA, 'PG_VERSION'))) {
  console.log(`✓ cached cluster: ${DATA}`);
} else {
  console.log('building the demo data cluster (initdb + config + clean shutdown)…');
  sh('bash', [join(PGDIR, 'setup_cluster.sh')]);
}

// ---- 3) code-coupled outputs: regenerate from current SVM code every run -----------------------
// Build the on-ramp translator (release) if absent, then translate the browser-target module.
const TR = join(REPO, 'crates', 'svm-llvm', 'target', 'release', 'svm-llvm-translate');
if (!existsSync(TR)) {
  console.log('building svm-llvm-translate…');
  sh('cargo', ['build', '--release', '--bin', 'svm-llvm-translate'], {
    cwd: join(REPO, 'crates', 'svm-llvm'),
  });
}
const rawSvmb = join(CACHE, 'postgres.svmb');
console.log('translating bitcode → browser-target .svmb (--host-page 65536 --stub-externs)…');
sh(TR, [shimmedBc, '-o', rawSvmb, '--binary', '--host-page', '65536', '--stub-externs']);

// resolve caps + verify + re-serialize → the shipped module; then encode the cluster → the shipped image.
console.log('resolving + verifying (prep_svmb) and encoding the data image (build_image)…');
sh('cargo', ['run', '--release', '-p', 'svm-run', '--example', 'prep_svmb', '--',
  rawSvmb, join(ASSETS, 'postgres_resolved.svmb')]);
sh('cargo', ['run', '--release', '-p', 'svm-run', '--example', 'build_image', '--',
  DATA, join(ASSETS, 'pgdata.img')]);

console.log(
  `✓ staged web/assets/postgres_resolved.svmb (${mb(join(ASSETS, 'postgres_resolved.svmb'))} MB) ` +
  `+ pgdata.img (${mb(join(ASSETS, 'pgdata.img'))} MB) — regenerated from current code`,
);
