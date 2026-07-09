# Magic ring buffer over the SharedRegion capability

`ring_demo.c` proves the **two-window-offsets** case of multi-mapping (`MMAP_CAPABILITY.md` §4e): one
region of `CAP` bytes mapped at two *adjacent* window offsets, so the second mapping continues the
first. A span that runs off the end of the ring wraps to the start seamlessly — a single `memcpy`
crosses the boundary, no split, no branch. This is exactly what `SharedRegion`'s multi-offset aliasing
exists for; here a guest reaches it through the granted `host_fs_mmap` capability.

- **guest** (`-DSVM_GUEST`): `FS_MAP_REGION` mints a file-backed `SharedRegion`; the guest maps it
  twice with `SharedRegion.map` (`__vm_region_call`) over a page-aligned `2*CAP` window span.
- **native oracle**: a `memfd` double-mapped with raw `mmap(MAP_SHARED | MAP_FIXED)` — the classic
  magic-ring-buffer setup.

Same driver → byte-identical stdout: the capability primitive matches the OS one. Driven by
`demo_ring_buffer_magic_mapping_vs_native` in `crates/svm-llvm/tests/translate.rs`.
