/* GPU fragment-shader demo for the playground — a Mandelbrot zoom rendered on the **GPU** through a
 * `webgpu` capability, presented to the page's <canvas>. Unlike `mandelzoom.c` (which computes every
 * pixel on the CPU in-guest), here the guest ships a **WGSL fragment shader** once and then just asks
 * the host to present a frame each tick with an updated time — the massively-parallel escape-time
 * loop runs on the GPU, so it stays smooth at full resolution.
 *
 * The `webgpu` capability (browser: serviced against `navigator.gpu`; the guest never holds a GPU
 * pointer, only a small integer handle — validation, not raw access, is the boundary, §2a):
 *   g = __vm_cap_resolve("webgpu", 6);
 *   __vm_host_call(g, 0, wgsl_ptr, wgsl_len, 0, 0);   // op 0 = set_shader(wgsl): compile once → 0/-1
 *   __vm_host_call(g, 1, frame, W, H, 0);             // op 1 = present(frame, w, h): render + show
 *
 * A reactor: `main()` sets the shader once; the host calls `tick()` once per animation frame, which
 * presents with an incrementing frame counter (the shader zooms/animates off it). No CPU pixel work,
 * no readback — only the tiny (frame, w, h) scalars cross the boundary per frame. */

extern int __vm_cap_resolve(const char *name, long len);
extern long __vm_host_call(int h, int op, long a, long b, long c, long d);

enum { GPU_SET_SHADER = 0, GPU_PRESENT = 1 };

#define W 640
#define H 480

static int gpu;
static int frame;

/* A fullscreen-triangle vertex shader + a Mandelbrot-zoom fragment shader. `p.frame` drives the zoom
 * (exp-decay scale toward a seahorse-valley point) and a smooth cosine-palette color cycle. All the
 * math (`exp`/`cos`/the escape loop) runs on the GPU. */
static const char shader[] =
    "struct P { frame: f32, w: f32, h: f32, pad: f32 };\n"
    "@group(0) @binding(0) var<uniform> p: P;\n"
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "  var v = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));\n"
    "  return vec4f(v[i], 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs(@builtin(position) c: vec4f) -> @location(0) vec4f {\n"
    "  let t = p.frame;\n"
    "  let scale = 1.4 * exp(-t * 0.02);\n"
    "  let center = vec2f(-0.743643887037151, 0.131825904205330);\n"
    "  let uv = (c.xy - vec2f(p.w, p.h) * 0.5) / (p.h * 0.5);\n"
    "  let cc = center + uv * scale;\n"
    "  var z = vec2f(0.0, 0.0);\n"
    "  var it = 0.0;\n"
    "  let maxit = 240.0;\n"
    "  loop {\n"
    "    if (it >= maxit) { break; }\n"
    "    z = vec2f(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + cc;\n"
    "    if (dot(z, z) > 4.0) { break; }\n"
    "    it = it + 1.0;\n"
    "  }\n"
    "  if (it >= maxit) { return vec4f(0.0, 0.0, 0.0, 1.0); }\n"
    "  let m = it / maxit;\n"
    "  let col = 0.5 + 0.5 * cos(6.28318 * (m + vec3f(0.0, 0.33, 0.67)) + t * 0.15);\n"
    "  return vec4f(col, 1.0);\n"
    "}\n";

int main(void) {
  gpu = __vm_cap_resolve("webgpu", 6);
  if (gpu >= 0) {
    __vm_host_call(gpu, GPU_SET_SHADER, (long)shader, sizeof shader - 1, 0, 0);
  }
  frame = 0;
  return 0;
}

int tick(void) {
  if (gpu >= 0) {
    __vm_host_call(gpu, GPU_PRESENT, frame, W, H, 0);
  }
  frame++;
  return 0;
}
