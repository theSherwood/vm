//! The `webgpu` capability's host seam (wasm build only): a single imported function the embedder
//! (the playground's `par.js` / `play.js`) supplies, backed by the browser's `navigator.gpu`. The
//! guest reaches it through the ordinary §7 host-defined-capability surface (`__vm_cap_resolve
//! ("webgpu")` + `__vm_host_call`); the `HostFn` closure in `grant_onramp_caps` marshals each op to
//! this import. Kept off the escape-TCB: the guest only ever names a masked handle and passes bytes
//! (WGSL source) + scalars; all GPU work — validation included — happens host-side in WebGPU.
//!
//! Protocol (`webgpu_op(op, a, b, c, ptr, len) -> i64`):
//! - op 0 `set_shader`: `ptr`/`len` are the WGSL source (a full-screen `vs` + a `fs` entry). Compiles
//!   the render pipeline. → 0 on success, -1 on a compile/validation error.
//! - op 1 `present`: `a` = frame counter, `b` = width, `c` = height. Writes the `(frame, w, h)`
//!   uniform and renders one frame to the page canvas. → 0.

#[link(wasm_import_module = "svm_host")]
extern "C" {
    /// The embedder-supplied WebGPU servicer. `ptr`/`len` point into this module's linear memory (the
    /// bytes the `HostFn` copied out of the guest window), so JS reads them as
    /// `new Uint8Array(memory.buffer, ptr, len)`.
    pub fn webgpu_op(op: i32, a: i64, b: i64, c: i64, ptr: *const u8, len: usize) -> i64;
}
