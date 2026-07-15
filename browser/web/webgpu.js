// The playground's **WebGPU servicer** for the guest `webgpu` capability. `initWebGPU(canvas)` sets up
// a `navigator.gpu` device + the canvas context and installs `globalThis.__svm_webgpu_op`; the wasm
// engine's `svm_host.webgpu_op` import (see par.js) delegates to it. The guest ships a WGSL shader
// once (op 0) and asks the host to present a frame each tick (op 1); the parallel pixel work runs on
// the GPU. All per-op work is **synchronous** (device creation — the only async part — is awaited here
// before the reactor loop starts), which is required: the reactor drives on the main thread, where a
// blocking wait is not allowed. Only *data* (the WGSL source) and scalars cross the boundary; the
// guest never holds a GPU object — validation, not raw access, is the boundary (§2a).

let G = null; // { device, ctx, format, canvas, uniform, pipeline, bindGroup }

export function webgpuAvailable() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export async function initWebGPU(canvas) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter =
    (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ||
    (await navigator.gpu.requestAdapter({ forceFallbackAdapter: true }));
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  device.addEventListener?.('uncapturederror', (e) =>
    console.error('webgpu device error:', e.error?.message ?? e.error),
  );
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const uniform = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  G = { device, ctx, format, canvas, uniform, pipeline: null, bindGroup: null };
  globalThis.__svm_webgpu_op = servicer;
}

export function teardownWebGPU() {
  globalThis.__svm_webgpu_op = null;
  if (G && G.device) {
    try {
      G.device.destroy();
    } catch {
      /* ignore */
    }
  }
  G = null;
}

// Test-only: render the current pipeline to an **offscreen** texture and read the pixels back. Headless
// Chromium cannot screenshot / read back the WebGPU *canvas swapchain*, but an offscreen texture reads
// back fine — so the browser test verifies the guest's shader really renders through the full stack
// (wasm cap → servicer → WebGPU) by checking these pixels. Returns RGBA8 `Uint8Array` of `w*h*4`.
export async function readbackForTest(frame, w, h) {
  if (!G || !G.wgsl) throw new Error('no shader set');
  const mod = G.device.createShaderModule({ code: G.wgsl });
  const pipeline = G.device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: mod, entryPoint: 'vs' },
    fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  });
  const tex = G.device.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const bindGroup = G.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: G.uniform } }],
  });
  G.device.queue.writeBuffer(G.uniform, 0, new Float32Array([frame, w, h, 0]));
  const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
  const rb = G.device.createBuffer({
    size: bytesPerRow * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = G.device.createCommandEncoder();
  const rp = enc.beginRenderPass({
    colorAttachments: [
      { view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
    ],
  });
  rp.setPipeline(pipeline);
  rp.setBindGroup(0, bindGroup);
  rp.draw(3);
  rp.end();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow, rowsPerImage: h }, { width: w, height: h });
  G.device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(rb.getMappedRange().slice(0));
  const out = new Uint8Array(w * h * 4); // strip the 256-byte row padding
  for (let y = 0; y < h; y++) out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
  rb.unmap();
  return out;
}

// The synchronous op servicer the wasm import calls. `a`/`b`/`c` are BigInt (wasm i64); `ptr`/`len`
// are Numbers (wasm i32) indexing the shared linear memory. Returns a Number (par.js widens to i64).
function servicer(op, a, b, c, ptr, len, memory) {
  if (!G) return -1;
  try {
    if (op === 0) {
      // set_shader(ptr, len): compile the render pipeline from the guest's WGSL (a `vs` + `fs`).
      // `.slice()` copies out of the SharedArrayBuffer (the threads build's memory) into a plain
      // ArrayBuffer — `TextDecoder` refuses a view backed by shared memory.
      const src = new TextDecoder().decode(
        new Uint8Array(memory.buffer, Number(ptr), Number(len)).slice(),
      );
      const mod = G.device.createShaderModule({ code: src });
      const pipeline = G.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: G.format }] },
        primitive: { topology: 'triangle-list' },
      });
      G.pipeline = pipeline;
      G.wgsl = src; // kept for the test's offscreen readback path
      G.bindGroup = G.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: G.uniform } }],
      });
      return 0;
    }
    if (op === 1) {
      // present(frame, w, h): write the uniform and render one frame to the canvas.
      if (!G.pipeline) return -1;
      const frame = Number(a);
      const w = Number(b);
      const h = Number(c);
      if (G.canvas.width !== w || G.canvas.height !== h) {
        G.canvas.width = w;
        G.canvas.height = h;
      }
      G.device.queue.writeBuffer(G.uniform, 0, new Float32Array([frame, w, h, 0]));
      const enc = G.device.createCommandEncoder();
      const rp = enc.beginRenderPass({
        colorAttachments: [
          {
            view: G.ctx.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      rp.setPipeline(G.pipeline);
      rp.setBindGroup(0, G.bindGroup);
      rp.draw(3);
      rp.end();
      G.device.queue.submit([enc.finish()]);
      return 0;
    }
    return -1;
  } catch (e) {
    console.error('webgpu_op op', op, 'failed:', e.message);
    return -1;
  }
}
