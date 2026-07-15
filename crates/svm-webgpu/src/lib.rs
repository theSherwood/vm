//! **A headless WebGPU compute capability for the SVM (LLVM.md §"GPU via a WebGPU capability").**
//!
//! The host holds a real GPU device via [`wgpu`]; the guest drives it through the same generic
//! host-defined-capability surface the `fs`/LMDB shims use — `__vm_cap_resolve("webgpu")` +
//! `__vm_host_call(handle, op, …)` — so **no translator change** is needed (§7 `HostFn`). WebGPU is
//! the *right* GPU waist for a sandbox: no raw pointers, validated buffers/bind-groups, and WGSL
//! shaders that are safe by construction — the guest never holds a GPU pointer, only *data* crosses
//! the window boundary (a buffer's contents up, a compute buffer's results back). This is the §2a
//! thesis again: **validation, not raw access, is the boundary**, and here wgpu does the validating.
//!
//! This first slice is the **headless compute data plane** (demo 1 in the doc): create buffer →
//! upload → set a WGSL compute shader → dispatch → read back, checked against a CPU reference —
//! with zero windowing. It runs on any wgpu backend; in CI it uses **lavapipe** (Mesa's software
//! Vulkan) so it needs no physical GPU.
//!
//! Kept in its own workspace-excluded crate (like `svm-llvm`) so the heavy `wgpu` dependency never
//! enters the default `cargo build`/`test`. Promoting the cap into `svm-run` behind a feature is a
//! follow-up once the shape is proven.
//!
//! ## Op protocol (`__vm_host_call(handle, op, a, b, c, d) -> i64`)
//! | op | name | args | returns |
//! |----|------|------|---------|
//! | 0 | `create_buffer` | `(size_bytes, _, _, _)` | buffer id ≥ 0 / `-1` |
//! | 1 | `write_buffer`  | `(buf_id, guest_ptr, len, _)` | 0 / `-1` |
//! | 2 | `set_shader`    | `(wgsl_ptr, wgsl_len, _, _)` | pipeline id ≥ 0 / `-1` (compile error) |
//! | 3 | `dispatch`      | `(pipeline_id, buf0_id, buf1_id\|-1, groups_x)` | 0 / `-1` |
//! | 4 | `read_buffer`   | `(buf_id, guest_ptr, len, _)` | 0 / `-1` |
//!
//! A buffer is bound at `@binding(0)` (and `@binding(1)` when `buf1_id >= 0`) of `@group(0)`; the
//! shader entry point is `main`; the workgroup count is `(groups_x, 1, 1)`.

use svm_interp::{GuestMem, Trap};
use svm_run::HostCap;

/// The lazily-initialized GPU context (device + queue), shared for a run's lifetime.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// Request a **fallback (software) adapter** so this works with no physical GPU (lavapipe in CI),
    /// then a default device. Returns `None` if no adapter/device is available.
    fn init() -> Option<Gpu> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                })
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;
            Some(Gpu { device, queue })
        })
    }
}

/// The capability's per-run state: the GPU context (lazy) plus the resource tables the guest indexes
/// by id. `dead` latches once init fails so a GPU-less host fails every op cleanly instead of retrying.
#[derive(Default)]
struct WebGpuState {
    gpu: Option<Gpu>,
    dead: bool,
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
}

impl WebGpuState {
    fn ctx(&mut self) -> Option<&Gpu> {
        if self.gpu.is_none() && !self.dead {
            self.gpu = Gpu::init();
            if self.gpu.is_none() {
                self.dead = true;
            }
        }
        self.gpu.as_ref()
    }

    fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
        let a = |i: usize| *args.get(i).unwrap_or(&0);
        match op {
            0 => self.create_buffer(a(0)),
            1 => self.write_buffer(a(0), a(1) as u64, a(2) as u64, mem),
            2 => self.set_shader(a(0) as u64, a(1) as u64, mem),
            3 => self.dispatch(a(0), a(1), a(2), a(3)),
            4 => self.read_buffer(a(0), a(1) as u64, a(2) as u64, mem),
            _ => -1,
        }
    }

    fn create_buffer(&mut self, size: i64) -> i64 {
        if size <= 0 {
            return -1;
        }
        let Some(gpu) = self.ctx() else { return -1 };
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffers.push(buf);
        (self.buffers.len() - 1) as i64
    }

    fn write_buffer(&mut self, id: i64, ptr: u64, len: u64, mem: Option<&mut dyn GuestMem>) -> i64 {
        let Some(mem) = mem else { return -1 };
        let Some(bytes) = mem.read_bytes(ptr, len) else {
            return -1;
        };
        let Some(gpu) = self.gpu.as_ref() else {
            return -1;
        };
        let Some(buf) = self.buffers.get(id as usize) else {
            return -1;
        };
        gpu.queue.write_buffer(buf, 0, &bytes);
        gpu.queue.submit(std::iter::empty()); // flush the staged write
        0
    }

    fn set_shader(&mut self, ptr: u64, len: u64, mem: Option<&mut dyn GuestMem>) -> i64 {
        let Some(mem) = mem else { return -1 };
        let Some(src) = mem.read_bytes(ptr, len) else {
            return -1;
        };
        let Ok(src) = String::from_utf8(src) else {
            return -1;
        };
        let Some(gpu) = self.ctx() else { return -1 };
        // A WGSL compile error would otherwise panic via the default error scope; capture it and
        // fail the op cleanly (a bad shader is a guest bug, not a host crash).
        gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if pollster::block_on(gpu.device.pop_error_scope()).is_some() {
            return -1; // validation error (bad WGSL / bad entry)
        }
        self.pipelines.push(pipeline);
        (self.pipelines.len() - 1) as i64
    }

    fn dispatch(&mut self, pipe: i64, buf0: i64, buf1: i64, groups: i64) -> i64 {
        let Some(gpu) = self.gpu.as_ref() else {
            return -1;
        };
        let Some(pipeline) = self.pipelines.get(pipe as usize) else {
            return -1;
        };
        let mut entries = Vec::new();
        for (binding, id) in [buf0, buf1].into_iter().enumerate() {
            if id < 0 {
                continue;
            }
            let Some(buf) = self.buffers.get(id as usize) else {
                return -1;
            };
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buf.as_entire_binding(),
            });
        }
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.max(0) as u32, 1, 1);
        }
        gpu.queue.submit(Some(enc.finish()));
        0
    }

    fn read_buffer(&mut self, id: i64, ptr: u64, len: u64, mem: Option<&mut dyn GuestMem>) -> i64 {
        let Some(mem) = mem else { return -1 };
        let Some(gpu) = self.gpu.as_ref() else {
            return -1;
        };
        let Some(src) = self.buffers.get(id as usize) else {
            return -1;
        };
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &readback, 0, len);
        gpu.queue.submit(Some(enc.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range().to_vec();
        readback.unmap();
        if mem.write_bytes(ptr, &data).is_none() {
            return -1;
        }
        0
    }
}

/// The `webgpu` capability: grant it under a name (e.g. `"webgpu"`) and the guest reaches it via
/// `__vm_cap_resolve("webgpu")` + `__vm_host_call`. Each host builds a fresh GPU context on first use.
pub fn webgpu_cap() -> HostCap {
    HostCap::host_fn(0, || {
        let mut st = WebGpuState::default();
        Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>|
                  -> Result<Vec<i64>, Trap> { Ok(vec![st.handle(op, args, mem)]) },
        )
    })
}

/// Whether a wgpu (software) adapter is available in this environment — the test gates on it to skip
/// cleanly on a host without lavapipe / a GPU, rather than fail.
pub fn adapter_available() -> bool {
    Gpu::init().is_some()
}
