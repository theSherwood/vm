//! SPEC.md slice 6 — the §17/D58 SIMD rows: reference lane semantics for every
//! `v128` value op, written from the `svm-ir` op documentation (wasm-parity lane
//! rules), never from a backend.
//!
//! Shape notes that drive the harness (`crates/svm/tests/spec_simd.rs`):
//! - `v128` cannot cross the JIT entry ABI (the trampoline's i64 slots zero it), so a
//!   row's `v128` inputs are baked as `v128.const` immediates and a `v128` result is
//!   observed as two `i64x2.extract_lane`s — every vector is a fresh module, batched
//!   many-per-module ([`module_for_simd_batch`]) to amortize JIT compiles.
//! - Every SIMD value op is **total** (§17: no traps), so a batch always completes.
//! - Float-lane results carry `nan_lanes`: a NaN lane's bit pattern is unpinned
//!   (D58 — same policy as scalar floats), compared lane-wise as "is a NaN".

use crate::{Enc, SpecVal};
use svm_ir::*;

/// Constructor of a SIMD row's op over its baked inputs' value indices.
pub type SimdBuildFn = Box<dyn Fn(&[ValIdx]) -> Inst>;
/// A SIMD row's reference lane semantics (total — SIMD value ops never trap).
pub type SimdEvalFn = Box<dyn Fn(&[SpecVal]) -> SpecVal>;

/// One concrete SIMD op row (op × shape × flags).
pub struct SimdRow {
    pub id: String,
    /// Input types, in operand order (`V128` baked as consts; scalars baked as consts).
    pub inputs: Vec<ValType>,
    /// Result type: `V128`, or the scalar for extracts/reductions.
    pub result: ValType,
    /// `Some(shape)` when the result is a `v128` with **float lanes whose NaNs are
    /// unpinned** — the conformance compare treats those lanes as NaN-class.
    pub nan_lanes: Option<VShape>,
    pub encoding: Enc,
    /// Construct the op over the baked inputs' value indices.
    pub build: SimdBuildFn,
    /// Reference lane semantics (total — SIMD value ops never trap).
    pub eval: SimdEvalFn,
}

// --- lane plumbing --------------------------------------------------------------------

type B16 = [u8; 16];

fn v(x: SpecVal) -> B16 {
    match x {
        SpecVal::V128(b) => b,
        _ => unreachable!("simd row fed a non-v128 input"),
    }
}
fn s_i32(x: SpecVal) -> i32 {
    match x {
        SpecVal::I32(v) => v,
        _ => unreachable!("simd row fed a non-i32 input"),
    }
}

/// Read the lanes of `bytes` at `shape`'s width, sign-extended to i64.
fn lanes_s(shape: VShape, b: &B16) -> Vec<i64> {
    let w = shape.lane_bytes() as usize;
    (0..16 / w)
        .map(|i| {
            let mut raw = [0u8; 8];
            raw[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
            let u = u64::from_le_bytes(raw);
            match w {
                1 => u as u8 as i8 as i64,
                2 => u as u16 as i16 as i64,
                4 => u as u32 as i32 as i64,
                _ => u as i64,
            }
        })
        .collect()
}

/// Read the lanes zero-extended to u64.
fn lanes_u(shape: VShape, b: &B16) -> Vec<u64> {
    let w = shape.lane_bytes() as usize;
    (0..16 / w)
        .map(|i| {
            let mut raw = [0u8; 8];
            raw[..w].copy_from_slice(&b[i * w..(i + 1) * w]);
            u64::from_le_bytes(raw)
        })
        .collect()
}

/// Write lanes (each truncated to the lane width) back into 16 bytes.
fn from_lanes(shape: VShape, lanes: &[u64]) -> B16 {
    let w = shape.lane_bytes() as usize;
    let mut out = [0u8; 16];
    for (i, l) in lanes.iter().enumerate() {
        out[i * w..(i + 1) * w].copy_from_slice(&l.to_le_bytes()[..w]);
    }
    out
}

fn lanes_f32(b: &B16) -> [f32; 4] {
    core::array::from_fn(|i| f32::from_le_bytes(b[i * 4..(i + 1) * 4].try_into().unwrap()))
}
fn lanes_f64(b: &B16) -> [f64; 2] {
    core::array::from_fn(|i| f64::from_le_bytes(b[i * 8..(i + 1) * 8].try_into().unwrap()))
}
fn from_f32(l: [f32; 4]) -> B16 {
    let mut out = [0u8; 16];
    for (i, x) in l.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&x.to_le_bytes());
    }
    out
}
fn from_f64(l: [f64; 2]) -> B16 {
    let mut out = [0u8; 16];
    for (i, x) in l.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&x.to_le_bytes());
    }
    out
}

/// Truncate an i64 lane value to the lane width and sign-extend back (the canonical
/// lane residue: what storing then sign-reading yields).
fn wrap_lane(shape: VShape, x: i64) -> i64 {
    match shape.lane_bytes() {
        1 => x as i8 as i64,
        2 => x as i16 as i64,
        4 => x as i32 as i64,
        _ => x,
    }
}

fn lane_mask(on: bool) -> u64 {
    if on {
        u64::MAX
    } else {
        0
    }
}

fn sat_lane(shape: VShape, x: i64, signed: bool) -> i64 {
    let bits = shape.lane_bytes() * 8;
    if signed {
        let max = (1i64 << (bits - 1)) - 1;
        let min = -(1i64 << (bits - 1));
        x.clamp(min, max)
    } else {
        x.clamp(0, (1i64 << bits) - 1)
    }
}

// --- per-family reference semantics ----------------------------------------------------

fn vint_bin(shape: VShape, op: VIntBinOp, a: &B16, b: &B16) -> B16 {
    let (xa, xb) = (lanes_s(shape, a), lanes_s(shape, b));
    let (ua, ub) = (lanes_u(shape, a), lanes_u(shape, b));
    let out: Vec<u64> = (0..xa.len())
        .map(|i| {
            (match op {
                VIntBinOp::Add => wrap_lane(shape, xa[i].wrapping_add(xb[i])),
                VIntBinOp::Sub => wrap_lane(shape, xa[i].wrapping_sub(xb[i])),
                VIntBinOp::Mul => wrap_lane(shape, xa[i].wrapping_mul(xb[i])),
                VIntBinOp::MinS => xa[i].min(xb[i]),
                VIntBinOp::MaxS => xa[i].max(xb[i]),
                VIntBinOp::MinU => return ua[i].min(ub[i]),
                VIntBinOp::MaxU => return ua[i].max(ub[i]),
            }) as u64
        })
        .collect();
    from_lanes(shape, &out)
}

fn vint_cmp(shape: VShape, op: VICmpOp, a: &B16, b: &B16) -> B16 {
    let (xa, xb) = (lanes_s(shape, a), lanes_s(shape, b));
    let (ua, ub) = (lanes_u(shape, a), lanes_u(shape, b));
    let out: Vec<u64> = (0..xa.len())
        .map(|i| {
            lane_mask(match op {
                VICmpOp::Eq => xa[i] == xb[i],
                VICmpOp::Ne => xa[i] != xb[i],
                VICmpOp::LtS => xa[i] < xb[i],
                VICmpOp::LtU => ua[i] < ub[i],
                VICmpOp::GtS => xa[i] > xb[i],
                VICmpOp::GtU => ua[i] > ub[i],
                VICmpOp::LeS => xa[i] <= xb[i],
                VICmpOp::LeU => ua[i] <= ub[i],
                VICmpOp::GeS => xa[i] >= xb[i],
                VICmpOp::GeU => ua[i] >= ub[i],
            })
        })
        .collect();
    from_lanes(shape, &out)
}

/// Ordered compares are false on NaN; `ne` is the unordered negation (true on NaN).
fn vfloat_cmp(shape: VShape, op: VFCmpOp, a: &B16, b: &B16) -> B16 {
    let cmp = |x: f64, y: f64| -> bool {
        match op {
            VFCmpOp::Eq => x == y,
            VFCmpOp::Ne => x != y,
            VFCmpOp::Lt => x < y,
            VFCmpOp::Gt => x > y,
            VFCmpOp::Le => x <= y,
            VFCmpOp::Ge => x >= y,
        }
    };
    match shape {
        VShape::F32x4 => {
            let (xa, xb) = (lanes_f32(a), lanes_f32(b));
            let out: Vec<u64> = (0..4)
                .map(|i| lane_mask(cmp(xa[i] as f64, xb[i] as f64)))
                .collect();
            from_lanes(VShape::I32x4, &out)
        }
        _ => {
            let (xa, xb) = (lanes_f64(a), lanes_f64(b));
            let out: Vec<u64> = (0..2).map(|i| lane_mask(cmp(xa[i], xb[i]))).collect();
            from_lanes(VShape::I64x2, &out)
        }
    }
}

fn vshift(shape: VShape, op: VShiftOp, a: &B16, amt: i32) -> B16 {
    let bits = shape.lane_bytes() * 8;
    let k = (amt as u32) % bits; // amount mod the lane bit-width (the wasm rule)
    let xs = lanes_s(shape, a);
    let us = lanes_u(shape, a);
    let out: Vec<u64> = (0..xs.len())
        .map(|i| match op {
            VShiftOp::Shl => wrap_lane(shape, xs[i] << k) as u64,
            VShiftOp::ShrS => (xs[i] >> k) as u64,
            // Logical: shift the zero-extended lane value.
            VShiftOp::ShrU => us[i] >> k,
        })
        .collect();
    from_lanes(shape, &out)
}

fn vint_un(shape: VShape, op: VIntUnOp, a: &B16) -> B16 {
    let out: Vec<u64> = lanes_s(shape, a)
        .into_iter()
        .map(|x| {
            (match op {
                VIntUnOp::Abs => wrap_lane(shape, x.wrapping_abs()), // abs(MIN) == MIN (wrap)
                VIntUnOp::Neg => wrap_lane(shape, 0i64.wrapping_sub(x)),
            }) as u64
        })
        .collect();
    from_lanes(shape, &out)
}

fn vsat_bin(shape: VShape, op: VSatBinOp, a: &B16, b: &B16) -> B16 {
    let (xa, xb) = (lanes_s(shape, a), lanes_s(shape, b));
    let (ua, ub) = (lanes_u(shape, a), lanes_u(shape, b));
    let out: Vec<u64> = (0..xa.len())
        .map(|i| {
            (match op {
                VSatBinOp::AddS => sat_lane(shape, xa[i] + xb[i], true),
                VSatBinOp::SubS => sat_lane(shape, xa[i] - xb[i], true),
                VSatBinOp::AddU => sat_lane(shape, (ua[i] + ub[i]) as i64, false),
                VSatBinOp::SubU => sat_lane(shape, ua[i] as i64 - ub[i] as i64, false),
            }) as u64
        })
        .collect();
    from_lanes(shape, &out)
}

/// Widen: take the low/high half of the narrower source lanes and sign/zero-extend.
fn vwiden(wide: VShape, op: VWidenOp, a: &B16) -> B16 {
    let narrow = wide.narrower().unwrap();
    let n = wide.lanes() as usize;
    let (signed, high) = match op {
        VWidenOp::LowS => (true, false),
        VWidenOp::LowU => (false, false),
        VWidenOp::HighS => (true, true),
        VWidenOp::HighU => (false, true),
    };
    let src: Vec<u64> = if signed {
        lanes_s(narrow, a).into_iter().map(|x| x as u64).collect()
    } else {
        lanes_u(narrow, a)
    };
    let base = if high { n } else { 0 };
    let out: Vec<u64> = (0..n).map(|i| src[base + i]).collect();
    from_lanes(wide, &out)
}

/// Narrow: read both sources' wider lanes as **signed**, saturate to the narrow
/// signed/unsigned range, and concatenate `a`'s lanes then `b`'s.
fn vnarrow(narrow: VShape, op: VNarrowOp, a: &B16, b: &B16) -> B16 {
    let wide = narrow.wider().unwrap();
    let signed = matches!(op, VNarrowOp::S);
    let mut out: Vec<u64> = Vec::with_capacity(narrow.lanes() as usize);
    for src in [a, b] {
        for x in lanes_s(wide, src) {
            out.push(sat_lane(narrow, x, signed) as u64);
        }
    }
    from_lanes(narrow, &out)
}

/// The non-trapping lane conversions: `trunc_sat` (NaN→0, clamp), int→float
/// (round-to-nearest-even), and the width changes (low lanes; high lanes zeroed on
/// demote / trunc-from-f64).
fn vconvert(op: VCvtOp, a: &B16) -> B16 {
    match op {
        VCvtOp::F32x4ConvertI32x4S => {
            let x = lanes_s(VShape::I32x4, a);
            from_f32(core::array::from_fn(|i| x[i] as f32))
        }
        VCvtOp::F32x4ConvertI32x4U => {
            let x = lanes_u(VShape::I32x4, a);
            from_f32(core::array::from_fn(|i| x[i] as u32 as f32))
        }
        VCvtOp::I32x4TruncSatF32x4S => {
            let x = lanes_f32(a);
            let out: Vec<u64> = (0..4).map(|i| x[i] as i32 as u32 as u64).collect();
            from_lanes(VShape::I32x4, &out)
        }
        VCvtOp::I32x4TruncSatF32x4U => {
            let x = lanes_f32(a);
            let out: Vec<u64> = (0..4).map(|i| x[i] as u32 as u64).collect();
            from_lanes(VShape::I32x4, &out)
        }
        VCvtOp::F32x4DemoteF64x2Zero => {
            let x = lanes_f64(a);
            from_f32([x[0] as f32, x[1] as f32, 0.0, 0.0])
        }
        VCvtOp::F64x2PromoteLowF32x4 => {
            let x = lanes_f32(a);
            from_f64([x[0] as f64, x[1] as f64])
        }
        VCvtOp::F64x2ConvertLowI32x4S => {
            let x = lanes_s(VShape::I32x4, a);
            from_f64([x[0] as f64, x[1] as f64])
        }
        VCvtOp::F64x2ConvertLowI32x4U => {
            let x = lanes_u(VShape::I32x4, a);
            from_f64([x[0] as f64, x[1] as f64])
        }
        VCvtOp::I32x4TruncSatF64x2SZero => {
            let x = lanes_f64(a);
            let out: Vec<u64> = vec![x[0] as i32 as u32 as u64, x[1] as i32 as u32 as u64, 0, 0];
            from_lanes(VShape::I32x4, &out)
        }
        VCvtOp::I32x4TruncSatF64x2UZero => {
            let x = lanes_f64(a);
            let out: Vec<u64> = vec![x[0] as u32 as u64, x[1] as u32 as u64, 0, 0];
            from_lanes(VShape::I32x4, &out)
        }
    }
}

/// wasm `pmin`/`pmax`: a plain compare-and-select (`pmin = b < a ? b : a`,
/// `pmax = a < b ? b : a`) — NaN and ±0 follow the select, not IEEE min/max.
fn vpminmax(shape: VShape, op: VPMinMaxOp, a: &B16, b: &B16) -> B16 {
    match shape {
        VShape::F32x4 => {
            let (xa, xb) = (lanes_f32(a), lanes_f32(b));
            from_f32(core::array::from_fn(|i| match op {
                VPMinMaxOp::Pmin => {
                    if xb[i] < xa[i] {
                        xb[i]
                    } else {
                        xa[i]
                    }
                }
                VPMinMaxOp::Pmax => {
                    if xa[i] < xb[i] {
                        xb[i]
                    } else {
                        xa[i]
                    }
                }
            }))
        }
        _ => {
            let (xa, xb) = (lanes_f64(a), lanes_f64(b));
            from_f64(core::array::from_fn(|i| match op {
                VPMinMaxOp::Pmin => {
                    if xb[i] < xa[i] {
                        xb[i]
                    } else {
                        xa[i]
                    }
                }
                VPMinMaxOp::Pmax => {
                    if xa[i] < xb[i] {
                        xb[i]
                    } else {
                        xa[i]
                    }
                }
            }))
        }
    }
}

/// Lane-wise IEEE float binary ops; `Min`/`Max` are the NaN-propagating, `-0 < +0`
/// semantics matching the scalar `FBinOp` (via the same [`crate`] helpers).
fn vfloat_bin(shape: VShape, op: VFloatBinOp, a: &B16, b: &B16) -> B16 {
    let f = |x: f64, y: f64| -> f64 {
        match op {
            VFloatBinOp::Add => x + y,
            VFloatBinOp::Sub => x - y,
            VFloatBinOp::Mul => x * y,
            VFloatBinOp::Div => x / y,
            VFloatBinOp::Min => crate::fbin_f64(FBinOp::Min, x, y),
            VFloatBinOp::Max => crate::fbin_f64(FBinOp::Max, x, y),
        }
    };
    match shape {
        VShape::F32x4 => {
            let (xa, xb) = (lanes_f32(a), lanes_f32(b));
            // f32 lanes computed at f32 precision — promote/demote per lane would
            // double-round Add/Mul/Div, so use the scalar f32 helpers directly.
            from_f32(core::array::from_fn(|i| {
                crate::fbin_f32(
                    match op {
                        VFloatBinOp::Add => FBinOp::Add,
                        VFloatBinOp::Sub => FBinOp::Sub,
                        VFloatBinOp::Mul => FBinOp::Mul,
                        VFloatBinOp::Div => FBinOp::Div,
                        VFloatBinOp::Min => FBinOp::Min,
                        VFloatBinOp::Max => FBinOp::Max,
                    },
                    xa[i],
                    xb[i],
                )
            }))
        }
        _ => {
            let (xa, xb) = (lanes_f64(a), lanes_f64(b));
            from_f64(core::array::from_fn(|i| f(xa[i], xb[i])))
        }
    }
}

fn vfloat_un(shape: VShape, op: VFloatUnOp, a: &B16) -> B16 {
    let fop = match op {
        VFloatUnOp::Abs => FUnOp::Abs,
        VFloatUnOp::Neg => FUnOp::Neg,
        VFloatUnOp::Sqrt => FUnOp::Sqrt,
        VFloatUnOp::Ceil => FUnOp::Ceil,
        VFloatUnOp::Floor => FUnOp::Floor,
        VFloatUnOp::Trunc => FUnOp::Trunc,
        VFloatUnOp::Nearest => FUnOp::Nearest,
    };
    match shape {
        VShape::F32x4 => {
            let x = lanes_f32(a);
            from_f32(core::array::from_fn(|i| crate::fun_f32(fop, x[i])))
        }
        _ => {
            let x = lanes_f64(a);
            from_f64(core::array::from_fn(|i| crate::fun_f64(fop, x[i])))
        }
    }
}

// --- the input pool --------------------------------------------------------------------

/// 16-byte input patterns. One shape-agnostic pool serves every op — the same bytes
/// reinterpret per instruction, exactly like hardware (§17). Chosen so every lane
/// width sees zeros, all-ones, sign boundaries, ramps, and float specials.
pub fn v128_pool() -> Vec<SpecVal> {
    let mut pool: Vec<B16> = vec![
        [0u8; 16],
        [0xff; 16],
        core::array::from_fn(|i| i as u8),          // ramp 00..0f
        core::array::from_fn(|i| (0xf0 + i) as u8), // high-byte ramp
        core::array::from_fn(|i| if i % 2 == 0 { 0x80 } else { 0x7f }), // sign boundaries
        core::array::from_fn(|i| (i as u8).wrapping_mul(31) ^ 0xa5), // varied pattern
    ];
    // Integer lane boundaries: i32 lanes MIN/MAX/-1/+1, i64 lanes MIN/MAX.
    let mut b = [0u8; 16];
    b[..4].copy_from_slice(&i32::MIN.to_le_bytes());
    b[4..8].copy_from_slice(&i32::MAX.to_le_bytes());
    b[8..12].copy_from_slice(&(-1i32).to_le_bytes());
    b[12..].copy_from_slice(&1i32.to_le_bytes());
    pool.push(b);
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&i64::MIN.to_le_bytes());
    b[8..].copy_from_slice(&i64::MAX.to_le_bytes());
    pool.push(b);
    // Float lanes: f32x4 = [1.0, -2.5, NaN, +inf]; f64x2 = [-0.0, qNaN(payload)].
    let mut b = [0u8; 16];
    for (i, bits) in [0x3f80_0000u32, 0xc020_0000, 0x7fc0_0000, 0x7f80_0000]
        .iter()
        .enumerate()
    {
        b[i * 4..(i + 1) * 4].copy_from_slice(&bits.to_le_bytes());
    }
    pool.push(b);
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&0x8000_0000_0000_0000u64.to_le_bytes()); // -0.0
    b[8..].copy_from_slice(&0x7ff8_0000_0000_0001u64.to_le_bytes()); // qNaN payload
    pool.push(b);
    // f32x4 conversion boundaries: [2^31, -2^31, 2.5, -0.5] (trunc_sat clamp lattice).
    let mut b = [0u8; 16];
    for (i, bits) in [0x4f00_0000u32, 0xcf00_0000, 0x4020_0000, 0xbf00_0000]
        .iter()
        .enumerate()
    {
        b[i * 4..(i + 1) * 4].copy_from_slice(&bits.to_le_bytes());
    }
    pool.push(b);
    pool.into_iter().map(SpecVal::V128).collect()
}

/// Scalar pools for the mixed-operand rows (shift amounts, splat/replace scalars).
fn simd_scalar_pool(t: ValType) -> Vec<SpecVal> {
    match t {
        ValType::I32 => [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, -1, i32::MIN]
            .into_iter()
            .map(SpecVal::I32)
            .collect(),
        ValType::I64 => [0, 1, -1, i64::MIN, i64::MAX, 0x0102_0304_0506_0708]
            .into_iter()
            .map(SpecVal::I64)
            .collect(),
        ValType::F32 => [0x3f80_0000u32, 0x7fc0_0000, 0x8000_0000, 0xff80_0000]
            .into_iter()
            .map(SpecVal::F32)
            .collect(),
        ValType::F64 => [
            0x3ff0_0000_0000_0000u64,
            0x7ff8_0000_0000_0001,
            0x8000_0000_0000_0000,
        ]
        .into_iter()
        .map(SpecVal::F64)
        .collect(),
        _ => v128_pool(),
    }
}

/// Deterministic strided cross product over the row's input pools (cap shared with
/// the scalar suite: every unary/binary row takes its full product).
pub fn simd_vectors_for(row: &SimdRow) -> Vec<Vec<SpecVal>> {
    let pools: Vec<Vec<SpecVal>> = row.inputs.iter().map(|t| simd_scalar_pool(*t)).collect();
    let total: usize = pools.iter().map(|p| p.len()).product();
    let stride = total.div_ceil(crate::VECTOR_CAP.min(256)).max(1);
    (0..total)
        .step_by(stride)
        .map(|i| {
            let mut rest = i;
            pools
                .iter()
                .map(|p| {
                    let v = p[rest % p.len()];
                    rest /= p.len();
                    v
                })
                .collect()
        })
        .collect()
}

// --- the rows ---------------------------------------------------------------------------

fn enc(sub: u8) -> Enc {
    Enc::Prefixed(0xFE, sub)
}

/// All slice-6 rows: every §17 `v128` **value** op (the memory-facing `v128.load`/
/// `store` ride the memory suite's boundary lattice in `spec_simd.rs` instead).
pub fn simd_rows() -> Vec<SimdRow> {
    use ValType as V;
    let mut rows: Vec<SimdRow> = Vec::new();
    let int_shapes = [VShape::I8x16, VShape::I16x8, VShape::I32x4, VShape::I64x2];
    let float_shapes = [VShape::F32x4, VShape::F64x2];

    // v128.const: identity (the immediate round-trips through the backends).
    rows.push(SimdRow {
        id: "v128.const".into(),
        inputs: vec![V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x00),
        build: Box::new(|_| unreachable!("const row is built from its input directly")),
        eval: Box::new(|x| x[0]),
    });

    for shape in VShape::ALL {
        rows.push(SimdRow {
            id: format!("{}.splat", shape.name()),
            inputs: vec![shape.lane_val()],
            result: V::V128,
            nan_lanes: None, // splat moves bits, it computes nothing
            encoding: enc(0x03),
            build: Box::new(move |v| Inst::Splat { shape, a: v[0] }),
            eval: Box::new(move |x| {
                let bits: u64 = match x[0] {
                    SpecVal::I32(v) => v as u32 as u64,
                    SpecVal::I64(v) => v as u64,
                    SpecVal::F32(b) => b as u64,
                    SpecVal::F64(b) => b,
                    SpecVal::V128(_) => unreachable!(),
                };
                let n = shape.lanes() as usize;
                SpecVal::V128(from_lanes(shape, &vec![bits; n]))
            }),
        });
        // Extract low and high lanes; signed and unsigned variants for narrow lanes.
        let signed_variants: &[bool] = if shape.lane_bytes() <= 2 {
            &[true, false]
        } else {
            &[false]
        };
        for &signed in signed_variants {
            for lane in [0, shape.lanes() - 1] {
                rows.push(SimdRow {
                    id: format!(
                        "{}.extract_lane{} {lane}",
                        shape.name(),
                        if shape.lane_bytes() <= 2 {
                            if signed {
                                "_s"
                            } else {
                                "_u"
                            }
                        } else {
                            ""
                        }
                    ),
                    inputs: vec![V::V128],
                    result: shape.lane_val(),
                    nan_lanes: None,
                    encoding: enc(0x04),
                    build: Box::new(move |v| Inst::ExtractLane {
                        shape,
                        lane,
                        signed,
                        a: v[0],
                    }),
                    eval: Box::new(move |x| {
                        let b = v(x[0]);
                        let i = lane as usize;
                        match shape {
                            VShape::F32x4 => SpecVal::F32(lanes_u(shape, &b)[i] as u32),
                            VShape::F64x2 => SpecVal::F64(lanes_u(shape, &b)[i]),
                            VShape::I64x2 => SpecVal::I64(lanes_s(shape, &b)[i]),
                            _ => SpecVal::I32(if signed {
                                lanes_s(shape, &b)[i] as i32
                            } else {
                                lanes_u(shape, &b)[i] as i32
                            }),
                        }
                    }),
                });
            }
        }
        for lane in [0, shape.lanes() - 1] {
            rows.push(SimdRow {
                id: format!("{}.replace_lane {lane}", shape.name()),
                inputs: vec![V::V128, shape.lane_val()],
                result: V::V128,
                nan_lanes: None, // a bit move
                encoding: enc(0x05),
                build: Box::new(move |v| Inst::ReplaceLane {
                    shape,
                    lane,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    let mut lanes = lanes_u(shape, &v(x[0]));
                    lanes[lane as usize] = match x[1] {
                        SpecVal::I32(s) => s as u32 as u64,
                        SpecVal::I64(s) => s as u64,
                        SpecVal::F32(b) => b as u64,
                        SpecVal::F64(b) => b,
                        SpecVal::V128(_) => unreachable!(),
                    };
                    SpecVal::V128(from_lanes(shape, &lanes))
                }),
            });
        }
    }

    for shape in int_shapes {
        for op in VIntBinOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x06),
                build: Box::new(move |v| Inst::VIntBin {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vint_bin(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        for op in VICmpOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.cmp_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x0F),
                build: Box::new(move |v| Inst::VIntCmp {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vint_cmp(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        for op in VShiftOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::I32],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x11),
                build: Box::new(move |v| Inst::VShift {
                    shape,
                    op,
                    a: v[0],
                    amt: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vshift(shape, op, &v(x[0]), s_i32(x[1])))),
            });
        }
        for op in VIntUnOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.un_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x12),
                build: Box::new(move |v| Inst::VIntUn { shape, op, a: v[0] }),
                eval: Box::new(move |x| SpecVal::V128(vint_un(shape, op, &v(x[0])))),
            });
        }
        rows.push(SimdRow {
            id: format!("{}.all_true", shape.name()),
            inputs: vec![V::V128],
            result: V::I32,
            nan_lanes: None,
            encoding: enc(0x14),
            build: Box::new(move |v| Inst::VAllTrue { shape, a: v[0] }),
            eval: Box::new(move |x| {
                SpecVal::I32(lanes_u(shape, &v(x[0])).iter().all(|&l| l != 0) as i32)
            }),
        });
        rows.push(SimdRow {
            id: format!("{}.bitmask", shape.name()),
            inputs: vec![V::V128],
            result: V::I32,
            nan_lanes: None,
            encoding: enc(0x15),
            build: Box::new(move |v| Inst::VBitmask { shape, a: v[0] }),
            eval: Box::new(move |x| {
                let mut m = 0i32;
                for (i, l) in lanes_s(shape, &v(x[0])).iter().enumerate() {
                    if *l < 0 {
                        m |= 1 << i;
                    }
                }
                SpecVal::I32(m)
            }),
        });
    }

    for shape in [VShape::I8x16, VShape::I16x8] {
        for op in VSatBinOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.sat_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x16),
                build: Box::new(move |v| Inst::VSatBin {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vsat_bin(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        rows.push(SimdRow {
            id: format!("{}.avgr_u", shape.name()),
            inputs: vec![V::V128, V::V128],
            result: V::V128,
            nan_lanes: None,
            encoding: enc(0x1C),
            build: Box::new(move |v| Inst::VAvgr {
                shape,
                a: v[0],
                b: v[1],
            }),
            eval: Box::new(move |x| {
                let (ua, ub) = (lanes_u(shape, &v(x[0])), lanes_u(shape, &v(x[1])));
                let out: Vec<u64> = (0..ua.len()).map(|i| (ua[i] + ub[i] + 1) >> 1).collect();
                SpecVal::V128(from_lanes(shape, &out))
            }),
        });
        for op in VNarrowOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.narrow_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x18),
                build: Box::new(move |v| Inst::VNarrow {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vnarrow(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
    }

    for shape in [VShape::I16x8, VShape::I32x4, VShape::I64x2] {
        for op in VWidenOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.widen_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x17),
                build: Box::new(move |v| Inst::VWiden { shape, op, a: v[0] }),
                eval: Box::new(move |x| SpecVal::V128(vwiden(shape, op, &v(x[0])))),
            });
            rows.push(SimdRow {
                id: format!("{}.extmul_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x1E),
                build: Box::new(move |v| Inst::VExtMul {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| {
                    // Widen both operands' selected half, then multiply lane-wise
                    // (wide, so the product wraps only at the wide width).
                    let wa = lanes_s(shape, &vwiden(shape, op, &v(x[0])));
                    let wb = lanes_s(shape, &vwiden(shape, op, &v(x[1])));
                    let out: Vec<u64> = (0..wa.len())
                        .map(|i| wrap_lane(shape, wa[i].wrapping_mul(wb[i])) as u64)
                        .collect();
                    SpecVal::V128(from_lanes(shape, &out))
                }),
            });
        }
    }

    for shape in [VShape::I16x8, VShape::I32x4] {
        for signed in [true, false] {
            rows.push(SimdRow {
                id: format!(
                    "{}.extadd_pairwise_{}",
                    shape.name(),
                    if signed { "s" } else { "u" }
                ),
                inputs: vec![V::V128],
                result: V::V128,
                nan_lanes: None,
                encoding: enc(0x1F),
                build: Box::new(move |v| Inst::VExtAddPairwise {
                    shape,
                    signed,
                    a: v[0],
                }),
                eval: Box::new(move |x| {
                    let narrow = shape.narrower().unwrap();
                    let src: Vec<i64> = if signed {
                        lanes_s(narrow, &v(x[0]))
                    } else {
                        lanes_u(narrow, &v(x[0]))
                            .into_iter()
                            .map(|u| u as i64)
                            .collect()
                    };
                    let out: Vec<u64> = (0..shape.lanes() as usize)
                        .map(|i| wrap_lane(shape, src[2 * i] + src[2 * i + 1]) as u64)
                        .collect();
                    SpecVal::V128(from_lanes(shape, &out))
                }),
            });
        }
    }

    // Fixed-shape integer rows.
    rows.push(SimdRow {
        id: "i8x16.popcnt".into(),
        inputs: vec![V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x1B),
        build: Box::new(|v| Inst::VPopcnt { a: v[0] }),
        eval: Box::new(|x| {
            let mut b = v(x[0]);
            for byte in &mut b {
                *byte = byte.count_ones() as u8;
            }
            SpecVal::V128(b)
        }),
    });
    rows.push(SimdRow {
        id: "i32x4.dot_i16x8_s".into(),
        inputs: vec![V::V128, V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x1D),
        build: Box::new(|v| Inst::VDot { a: v[0], b: v[1] }),
        eval: Box::new(|x| {
            let (xa, xb) = (
                lanes_s(VShape::I16x8, &v(x[0])),
                lanes_s(VShape::I16x8, &v(x[1])),
            );
            let out: Vec<u64> = (0..4)
                .map(|i| {
                    let p = (xa[2 * i] * xb[2 * i]).wrapping_add(xa[2 * i + 1] * xb[2 * i + 1]);
                    wrap_lane(VShape::I32x4, p) as u64
                })
                .collect();
            SpecVal::V128(from_lanes(VShape::I32x4, &out))
        }),
    });
    rows.push(SimdRow {
        id: "i16x8.dot_i8x16_s".into(),
        inputs: vec![V::V128, V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x22),
        build: Box::new(|v| Inst::VDotI8 { a: v[0], b: v[1] }),
        eval: Box::new(|x| {
            let (xa, xb) = (
                lanes_s(VShape::I8x16, &v(x[0])),
                lanes_s(VShape::I8x16, &v(x[1])),
            );
            let out: Vec<u64> = (0..8)
                .map(|i| {
                    let p = (xa[2 * i] * xb[2 * i]).wrapping_add(xa[2 * i + 1] * xb[2 * i + 1]);
                    wrap_lane(VShape::I16x8, p) as u64 // wraps at i16 (svm-ir doc)
                })
                .collect();
            SpecVal::V128(from_lanes(VShape::I16x8, &out))
        }),
    });
    rows.push(SimdRow {
        id: "i16x8.q15mulr_sat_s".into(),
        inputs: vec![V::V128, V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x20),
        build: Box::new(|v| Inst::VQ15MulrSat { a: v[0], b: v[1] }),
        eval: Box::new(|x| {
            let (xa, xb) = (
                lanes_s(VShape::I16x8, &v(x[0])),
                lanes_s(VShape::I16x8, &v(x[1])),
            );
            let out: Vec<u64> = (0..8)
                .map(|i| sat_lane(VShape::I16x8, (xa[i] * xb[i] + 0x4000) >> 15, true) as u64)
                .collect();
            SpecVal::V128(from_lanes(VShape::I16x8, &out))
        }),
    });

    // Float families.
    for shape in float_shapes {
        for op in VFloatBinOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.f_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: Some(shape),
                encoding: enc(0x07),
                build: Box::new(move |v| Inst::VFloatBin {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vfloat_bin(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        for op in VFloatUnOp::ALL {
            // Abs/Neg are pure sign-bit moves — NaN payloads pass through exactly.
            let nan = !matches!(op, VFloatUnOp::Abs | VFloatUnOp::Neg);
            rows.push(SimdRow {
                id: format!("{}.fu_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128],
                result: V::V128,
                nan_lanes: nan.then_some(shape),
                encoding: enc(0x08),
                build: Box::new(move |v| Inst::VFloatUn { shape, op, a: v[0] }),
                eval: Box::new(move |x| SpecVal::V128(vfloat_un(shape, op, &v(x[0])))),
            });
        }
        for op in VFCmpOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.fcmp_{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: None, // masks are integer lanes — exact
                encoding: enc(0x10),
                build: Box::new(move |v| Inst::VFloatCmp {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vfloat_cmp(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        for op in VPMinMaxOp::ALL {
            rows.push(SimdRow {
                id: format!("{}.{:?}", shape.name(), op).to_lowercase(),
                inputs: vec![V::V128, V::V128],
                result: V::V128,
                nan_lanes: Some(shape),
                encoding: enc(0x1A),
                build: Box::new(move |v| Inst::VPMinMax {
                    shape,
                    op,
                    a: v[0],
                    b: v[1],
                }),
                eval: Box::new(move |x| SpecVal::V128(vpminmax(shape, op, &v(x[0]), &v(x[1])))),
            });
        }
        for neg in [false, true] {
            rows.push(SimdRow {
                id: format!("{}.fma neg={neg}", shape.name()),
                inputs: vec![V::V128; 3],
                result: V::V128,
                nan_lanes: Some(shape),
                encoding: enc(0x21),
                build: Box::new(move |v| Inst::VFma {
                    shape,
                    neg,
                    a: v[0],
                    b: v[1],
                    c: v[2],
                }),
                eval: Box::new(move |x| {
                    // a·b + c (or −a·b + c) with a single rounding, lane-wise.
                    let (a, b, c) = (v(x[0]), v(x[1]), v(x[2]));
                    SpecVal::V128(match shape {
                        VShape::F32x4 => {
                            let (xa, xb, xc) = (lanes_f32(&a), lanes_f32(&b), lanes_f32(&c));
                            from_f32(core::array::from_fn(|i| {
                                let m = if neg { -xa[i] } else { xa[i] };
                                m.mul_add(xb[i], xc[i])
                            }))
                        }
                        _ => {
                            let (xa, xb, xc) = (lanes_f64(&a), lanes_f64(&b), lanes_f64(&c));
                            from_f64(core::array::from_fn(|i| {
                                let m = if neg { -xa[i] } else { xa[i] };
                                m.mul_add(xb[i], xc[i])
                            }))
                        }
                    })
                }),
            });
        }
    }

    for op in VCvtOp::ALL {
        // Only int→float conversions compute float lanes (NaN can only pass through
        // trunc_sat as 0, and demote/promote of a NaN input yields a NaN lane).
        let nan = match op {
            VCvtOp::F32x4DemoteF64x2Zero => Some(VShape::F32x4),
            VCvtOp::F64x2PromoteLowF32x4 => Some(VShape::F64x2),
            _ => None,
        };
        rows.push(SimdRow {
            id: format!("vcvt.{op:?}").to_lowercase(),
            inputs: vec![V::V128],
            result: V::V128,
            nan_lanes: nan,
            encoding: enc(0x19),
            build: Box::new(move |v| Inst::VConvert { op, a: v[0] }),
            eval: Box::new(move |x| SpecVal::V128(vconvert(op, &v(x[0])))),
        });
    }

    // Whole-vector bit ops.
    for op in VBitBinOp::ALL {
        rows.push(SimdRow {
            id: format!("v128.{op:?}").to_lowercase(),
            inputs: vec![V::V128, V::V128],
            result: V::V128,
            nan_lanes: None,
            encoding: enc(0x09),
            build: Box::new(move |v| Inst::VBitBin {
                op,
                a: v[0],
                b: v[1],
            }),
            eval: Box::new(move |x| {
                let (a, b) = (v(x[0]), v(x[1]));
                let mut out = [0u8; 16];
                for i in 0..16 {
                    out[i] = match op {
                        VBitBinOp::And => a[i] & b[i],
                        VBitBinOp::Or => a[i] | b[i],
                        VBitBinOp::Xor => a[i] ^ b[i],
                        VBitBinOp::AndNot => a[i] & !b[i],
                    };
                }
                SpecVal::V128(out)
            }),
        });
    }
    rows.push(SimdRow {
        id: "v128.not".into(),
        inputs: vec![V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x0A),
        build: Box::new(|v| Inst::VNot { a: v[0] }),
        eval: Box::new(|x| {
            let mut b = v(x[0]);
            for byte in &mut b {
                *byte = !*byte;
            }
            SpecVal::V128(b)
        }),
    });
    rows.push(SimdRow {
        id: "v128.bitselect".into(),
        inputs: vec![V::V128; 3],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x0B),
        build: Box::new(|v| Inst::Bitselect {
            a: v[0],
            b: v[1],
            mask: v[2],
        }),
        eval: Box::new(|x| {
            let (a, b, m) = (v(x[0]), v(x[1]), v(x[2]));
            let mut out = [0u8; 16];
            for i in 0..16 {
                out[i] = (a[i] & m[i]) | (b[i] & !m[i]);
            }
            SpecVal::V128(out)
        }),
    });
    rows.push(SimdRow {
        id: "v128.any_true".into(),
        inputs: vec![V::V128],
        result: V::I32,
        nan_lanes: None,
        encoding: enc(0x13),
        build: Box::new(|v| Inst::VAnyTrue { a: v[0] }),
        eval: Box::new(|x| SpecVal::I32(v(x[0]).iter().any(|&b| b != 0) as i32)),
    });
    // Shuffle at a few immediate lane patterns (`lanes[i] < 16` picks from `a`, else
    // from `b`); swizzle's per-byte runtime indices (≥ 16 ⇒ 0).
    let shuffles: [[u8; 16]; 4] = [
        core::array::from_fn(|i| i as u8),                // identity (a)
        core::array::from_fn(|i| 15 - i as u8),           // reverse a
        core::array::from_fn(|i| 16 + i as u8),           // identity (b)
        core::array::from_fn(|i| (i as u8 * 7 + 3) % 32), // cross-pick
    ];
    for lanes in shuffles {
        rows.push(SimdRow {
            id: format!("i8x16.shuffle {lanes:?}"),
            inputs: vec![V::V128, V::V128],
            result: V::V128,
            nan_lanes: None,
            encoding: enc(0x0C),
            build: Box::new(move |v| Inst::Shuffle {
                lanes,
                a: v[0],
                b: v[1],
            }),
            eval: Box::new(move |x| {
                let (a, b) = (v(x[0]), v(x[1]));
                let mut out = [0u8; 16];
                for i in 0..16 {
                    let l = lanes[i] as usize;
                    out[i] = if l < 16 { a[l] } else { b[l - 16] };
                }
                SpecVal::V128(out)
            }),
        });
    }
    rows.push(SimdRow {
        id: "i8x16.swizzle".into(),
        inputs: vec![V::V128, V::V128],
        result: V::V128,
        nan_lanes: None,
        encoding: enc(0x0D),
        build: Box::new(|v| Inst::Swizzle { a: v[0], b: v[1] }),
        eval: Box::new(|x| {
            let (a, b) = (v(x[0]), v(x[1]));
            let mut out = [0u8; 16];
            for i in 0..16 {
                out[i] = if (b[i] as usize) < 16 {
                    a[b[i] as usize]
                } else {
                    0
                };
            }
            SpecVal::V128(out)
        }),
    });
    rows.push(SimdRow {
        id: "simd.width_bytes".into(),
        inputs: vec![],
        result: V::I32,
        nan_lanes: None,
        encoding: enc(0x0E),
        build: Box::new(|_| Inst::SimdWidthBytes),
        eval: Box::new(|_| SpecVal::I32(16)), // fixed-128 floor (D58)
    });

    rows
}

// --- module construction ----------------------------------------------------------------

/// Bake one input as a const instruction.
fn const_inst(x: SpecVal) -> Inst {
    match x {
        SpecVal::I32(v) => Inst::ConstI32(v),
        SpecVal::I64(v) => Inst::ConstI64(v),
        SpecVal::F32(b) => Inst::ConstF32(b),
        SpecVal::F64(b) => Inst::ConstF64(b),
        SpecVal::V128(b) => Inst::ConstV128(b),
    }
}

/// One function packing a **batch** of a row's vectors: per vector, const inputs → op
/// → (for a `v128` result) two `i64x2.extract_lane`s; all observations returned
/// together. SIMD value ops are total, so the whole batch always completes — this
/// amortizes one JIT compile over the batch.
pub fn module_for_simd_batch(row: &SimdRow, batch: &[Vec<SpecVal>]) -> Module {
    let mut insts: Vec<Inst> = Vec::new();
    let mut results: Vec<ValIdx> = Vec::new();
    let mut result_tys: Vec<ValType> = Vec::new();
    let mut idx: ValIdx = 0;
    for vector in batch {
        let base = idx;
        for (i, x) in vector.iter().enumerate() {
            // v128.const's "input" IS the op — don't bake it twice.
            if row.id == "v128.const" && i == 0 {
                continue;
            }
            insts.push(const_inst(*x));
            idx += 1;
            let _ = base;
        }
        let opnds: Vec<ValIdx> = (base..idx).collect();
        if row.id == "v128.const" {
            insts.push(const_inst(vector[0]));
        } else {
            insts.push((row.build)(&opnds));
        }
        let r = idx;
        idx += 1;
        if row.result == ValType::V128 {
            for lane in [0u8, 1] {
                insts.push(Inst::ExtractLane {
                    shape: VShape::I64x2,
                    lane,
                    signed: false,
                    a: r,
                });
                results.push(idx);
                idx += 1;
                result_tys.push(ValType::I64);
            }
        } else {
            results.push(r);
            result_tys.push(row.result);
        }
    }
    Module {
        funcs: vec![Func {
            params: vec![],
            results: result_tys,
            blocks: vec![Block {
                params: vec![],
                insts,
                term: Terminator::Return(results),
            }],
        }],
        ..Default::default()
    }
}

/// A raw single-inst module for the encoding suite (indices dangle — `encode`/`decode`
/// are structural, so the round-trip and opcode pin don't need a verifiable module).
pub fn encode_probe_module(inst: Inst) -> Module {
    Module {
        funcs: vec![Func {
            params: vec![],
            results: vec![],
            blocks: vec![Block {
                params: vec![],
                insts: vec![inst],
                term: Terminator::Return(vec![]),
            }],
        }],
        ..Default::default()
    }
}
