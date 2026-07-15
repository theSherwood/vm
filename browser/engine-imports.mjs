// The import object every instantiation of the **engine** wasm (`svm_browser.wasm`) must supply.
// Besides the optional shared `memory` (the threads build imports it; the plain build owns its own),
// the wasm32 build imports `svm_host.webgpu_op` — the `webgpu` capability's host seam (a guest ships a
// WGSL shader / asks the host to present a frame). Only the playground's main thread (`web/par.js`)
// services it for real against `navigator.gpu`; every other instantiation — the corpus/bench
// differentials, the parallel-Worker vCPUs — has no GPU surface, so it passes this no-op stub and a
// guest that resolves the `webgpu` cap there simply gets -1 back and skips. Returns a BigInt (i64).
//
// (Emitted wasm-JIT *units* are a different module with their own imports — `env.{memory,trap,
// call_interp}`, no `svm_host` — so they do NOT use this.)
export function engineImports(memory) {
  const imports = { svm_host: { webgpu_op: () => -1n } };
  if (memory) imports.env = { memory };
  return imports;
}
