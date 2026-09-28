# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NVFP4 (E2M1) Block-Scaled Grouped Quantization -- CuTe-DSL Implementation.

Port of nvfp4-group-quant.cpp to Python CuTe-DSL.

This kernel quantizes grouped expert inputs from fp16/bf16 to NVFP4 (E2M1)
with E4M3 block scale factors, laid out in the 128x4 swizzled tile format
required by blockscale GEMM.

Key properties:
  - Input:  (M, K) in fp16/bf16, row-major
  - Output: (M, K/2) in packed uint8 (2 x E2M1 per byte)
  - SF_VEC_SIZE = 16: each thread computes 1 scale factor over 16 elements
  - SF format = E4M3, stored in 128x4 swizzled tile layout
  - SF count/row = K/16
  - One 128x128 input tile -> 2 SF K-tiles (each 128x4 = 512 bytes)
  - Shared memory: 1024 bytes (two 512-byte SF tiles)
  - Two TMA bulk copies per input tile to flush both SF K-tiles

Thread mapping (128 threads, ThrLayout (16,8):(8,1)):
  thread t -> row_in_subtile = t/8, k_col = t%8
  Each thread loads 16 input elements, produces 1 SF + 8 packed FP4 bytes.
  k_col 0..3 -> SF K-tile 0, k_col 4..7 -> SF K-tile 1

Requires: CUTLASS 4.x with CuTe-DSL, SM100+ GPU, CUDA 12.9+
"""

import functools
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# =====================================================================
# Constants
# =====================================================================
FP4_BLOCK_M = 128
FP4_BLOCK_K = 128
FP4_BLOCK_K_BYTES = 64  # 128 / 2
FP4_THREAD_BLOCK_SIZE = 128
E2M1_MAX = 6.0
SF_VEC_SIZE = 16
SF_TILE_BYTES = 512  # 128x4 bytes per SF tile


# =====================================================================
# PTX Intrinsics
# =====================================================================


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 (low byte of result)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32_and_rcp(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32 AND compute reciprocal."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .u32 exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, result;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                rcp.approx.ftz.f32 result, fp8_float;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert 8 float32 to 8 E2M1 (4-bit) packed into uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v).ir_value(loc=loc, ip=ip)
                for v in [v0, v1, v2, v3, v4, v5, v6, v7]
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =====================================================================
# Half2 / BFloat2 SIMD Intrinsics
# =====================================================================


@dsl_user_op
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $1;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            max.f32 $0, f0, f1;
        }
        """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $1, 0xFFFF;
            shr.b32 hi, $1, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            max.f32 $0, f0, f1;
        }
        """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


# =====================================================================
# Vectorized Memory Access
# =====================================================================


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: cutlass.Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def st_global_u64(base_ptr: cutlass.Int64, value: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(
    tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None
) -> cutlass.Int64:
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return cutlass.Int64(ptr_int)


# =====================================================================
# TMA Intrinsics (SF tile flush: shared -> global)
# =====================================================================


@dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_bulk_s2g(
    dst_gmem: cutlass.Int64, src_smem: Int32, nbytes: Int32, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(dst_gmem).ir_value(loc=loc, ip=ip),
            Int32(src_smem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_bulk_commit(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_bulk_wait_group_read_0(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group.read 0;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =====================================================================
# Helper Functions
# =====================================================================


@cute.jit
def load_8_vec2(mInput: cute.Tensor, row: Int32, col: Int32, K: int):
    """Load 16 elements as 8 Uint32 (half2/bfloat2) via 2x 128-bit loads."""
    h2 = cute.make_rmem_tensor((8,), Uint32)
    ptr0 = get_ptr_as_int64(mInput, row * K + col)
    ptr1 = get_ptr_as_int64(mInput, row * K + col + Int32(8))
    h2[0], h2[1], h2[2], h2[3] = ld_global_v4_u32(ptr0)
    h2[4], h2[5], h2[6], h2[7] = ld_global_v4_u32(ptr1)
    return h2


@cute.jit
def half2_max_abs_8(h2: cute.Tensor) -> Uint32:
    """Max |value| across 8 half2 pairs via tree reduction."""
    abs_vals = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_vals[i] = habs2(h2[i])
    m01 = hmax2(abs_vals[0], abs_vals[1])
    m23 = hmax2(abs_vals[2], abs_vals[3])
    m45 = hmax2(abs_vals[4], abs_vals[5])
    m67 = hmax2(abs_vals[6], abs_vals[7])
    return hmax2(hmax2(m01, m23), hmax2(m45, m67))


@cute.jit
def bfloat2_max_abs_8(h2: cute.Tensor) -> Uint32:
    """Max |value| across 8 bfloat2 pairs via tree reduction."""
    abs_vals = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_vals[i] = bfloat2_habs2(h2[i])
    m01 = bfloat2_hmax2(abs_vals[0], abs_vals[1])
    m23 = bfloat2_hmax2(abs_vals[2], abs_vals[3])
    m45 = bfloat2_hmax2(abs_vals[4], abs_vals[5])
    m67 = bfloat2_hmax2(abs_vals[6], abs_vals[7])
    return bfloat2_hmax2(bfloat2_hmax2(m01, m23), bfloat2_hmax2(m45, m67))


@cute.jit
def half2_to_float16_scaled(h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 half2 to 16 float32, each multiplied by scale."""
    y = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y[i * 2], y[i * 2 + 1] = half2_to_float2_scaled(h2[i], scale)
    return y


@cute.jit
def bfloat2_to_float16_scaled(h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 bfloat2 to 16 float32, each multiplied by scale."""
    y = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y[i * 2], y[i * 2 + 1] = bfloat2_to_float2_scaled(h2[i], scale)
    return y


@cute.jit
def quantize_and_pack_16(y_f32: cute.Tensor) -> Uint64:
    """Convert 16 float32 values to E2M1, pack into uint64."""
    packed_lo = cvt_e2m1x8_f32(
        y_f32[0], y_f32[1], y_f32[2], y_f32[3], y_f32[4], y_f32[5], y_f32[6], y_f32[7]
    )
    packed_hi = cvt_e2m1x8_f32(
        y_f32[8],
        y_f32[9],
        y_f32[10],
        y_f32[11],
        y_f32[12],
        y_f32[13],
        y_f32[14],
        y_f32[15],
    )
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


# =====================================================================
# Kernel Class
# =====================================================================


class NVFP4GroupQuantKernel:
    """
    NVFP4 Block-Scaled Grouped Quantization Kernel.

    Processes multiple expert groups. Each group has its own (M, N, K) problem
    size and offset into the flat input/output buffers. The kernel tiles the
    input in 128x128 blocks, quantizes each thread's 16 elements to FP4,
    computes an E4M3 scale factor, and writes the SF into a 128x4 swizzled
    tile layout via shared memory + TMA bulk copy.

    Thread mapping mirrors the C++ version:
      128 threads, ThrLayout (16,8):(8,1)
      thread t -> row_in_subtile = t/8, k_col = t%8
      k_col 0..3 -> SF K-tile 0
      k_col 4..7 -> SF K-tile 1
    """

    def __init__(self, dtype: Type[cutlass.Numeric], is_fp16: bool):
        self.dtype = dtype
        self.is_fp16 = is_fp16

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,  # flat input buffer, shape (total_rows * K,)
        sf_scale_val: cute.Tensor,  # (num_groups,)
        mQuantOut: cute.Tensor,  # flat output buffer, shape (total_rows * K/2,)
        mScaleFactor: cute.Tensor,  # flat SF buffer, shape (total_sf,)
        mProblemSizes: cute.Tensor,  # (num_groups, 3) -> (M, N, K) per group
        mExpertOffsets: cute.Tensor,  # (num_groups,) row offsets
        mBlockscaleOffsets: cute.Tensor,  # (num_groups,) SF row offsets
        num_groups: Int32,
        num_ctas: Int32,
        stream,
    ):
        # C++ uses SM_count * max_active_blocks_per_sm.
        # Caller passes this as num_ctas.
        self.kernel(
            mInput,
            mQuantOut,
            mScaleFactor,
            mProblemSizes,
            mExpertOffsets,
            mBlockscaleOffsets,
            sf_scale_val,
            num_groups,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[FP4_THREAD_BLOCK_SIZE, 1, 1],
            smem=1024,  # 2 x 512-byte SF tiles
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mQuantOut: cute.Tensor,
        mScaleFactor: cute.Tensor,
        mProblemSizes: cute.Tensor,
        mExpertOffsets: cute.Tensor,
        mBlockscaleOffsets: cute.Tensor,
        sf_scale: cute.Tensor,
        num_groups: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        is_fp16 = self.is_fp16

        # SF tile layout: ((32,4),4):((16,4),1) -- 128x4 = 512 bytes
        # This is the swizzled layout required by blockscale GEMM.

        # Allocate shared memory for 2 SF tiles (1024 bytes total)
        smem = cutlass.utils.SmemAllocator()
        sf_smem_0 = smem.allocate_tensor(
            Uint8,
            cute.make_layout(((32, 4), 4), stride=((16, 4), 1)),
            byte_alignment=512,
        )
        sf_smem_1 = smem.allocate_tensor(
            Uint8,
            cute.make_layout(((32, 4), 4), stride=((16, 4), 1)),
            byte_alignment=512,
        )

        # Thread decomposition (matches C++ ThrLayout (16,8):(8,1))
        k_col = tidx % Int32(8)  # 0..7
        row_in_subtile = tidx // Int32(8)  # 0..15
        is_sf_ktile0 = k_col < Int32(4)
        sf_col = k_col % Int32(4)

        # Precompute constants
        fp4_max_rcp = rcp_approx_ftz(Float32(E2M1_MAX))

        # Track whether we need to wait for a previous TMA bulk copy.
        # C++ does: wait at subtile t==0 of the NEXT tile, not at end of current.
        # We track this with is_first_tile; on subsequent tiles thread 0 waits.
        is_first_tile = Int32(1)

        # Persistent kernel: iterate over groups and tiles
        group_total_tiles = Int32(0)
        head_cta_id = Int32(0)

        for g in cutlass.range(num_groups):
            # Read problem size for this group
            m = mProblemSizes[g * Int32(3) + Int32(0)]
            k = mProblemSizes[g * Int32(3) + Int32(2)]
            # C++ uses int64_t cast for safety, but expert_offsets is int32 array.
            # For practical MoE sizes, int32 element offsets suffice.
            expert_offset = mExpertOffsets[g]
            blockscale_offset = mBlockscaleOffsets[g]
            sf_scale_val = sf_scale[g]

            k_input_tiles = k // Int32(FP4_BLOCK_K)  # K/128
            k_sf_tiles = k // Int32(64)  # K/16/4 = K/64
            m_tiles = (m + Int32(FP4_BLOCK_M - 1)) // Int32(FP4_BLOCK_M)
            total_tiles = m_tiles * k_input_tiles

            group_total_tiles = group_total_tiles + total_tiles
            blk_offset = (bidx + (gdim - head_cta_id)) % gdim
            head_cta_id = group_total_tiles % gdim

            while blk_offset < total_tiles:
                m_tile = blk_offset // k_input_tiles
                k_tile = blk_offset % k_input_tiles

                tile_row_base = m_tile * Int32(FP4_BLOCK_M)
                tile_col_base = k_tile * Int32(FP4_BLOCK_K)

                # ---- Subtile loop: 8 subtiles of 16 rows each ----
                # Matches C++: for (int t = 0; t < tile_loop_count; t++)
                for subtile in cutlass.range(8):
                    row = tile_row_base + subtile * Int32(16) + row_in_subtile
                    col = tile_col_base + k_col * Int32(SF_VEC_SIZE)

                    # Compute SF value (default 0 for OOB rows)
                    sf_fp8 = Uint8(0)

                    if row < m:
                        global_row = expert_offset + row

                        # ---- G2R: Load 16 elements as 8 half2/bfloat2 ----
                        # C++ G2R: AlignedArray<T_IN, 16> = 256 bits
                        # We use 2x ld.global.v4.u32 = 2x128 = 256 bits
                        h2 = load_8_vec2(mInput, global_row, col, k)

                        # ---- Compute max-abs over 16 elements (no shuffle) ----
                        if cutlass.const_expr(is_fp16):
                            max_h2 = half2_max_abs_8(h2)
                            vec_max = hmax_to_f32(max_h2)
                        else:
                            max_h2 = bfloat2_max_abs_8(h2)
                            vec_max = bfloat2_hmax_to_f32(max_h2)

                        # ---- Compute E4M3 scale factor ----
                        # C++: sf_value = sf_scale_val * (vec_max * rcp(6.0))
                        #      __nv_fp8_e4m3 tmp(sf_value)  // satfinite
                        # cvt.rn.satfinite.e4m3x2 already clamps, no fmin needed
                        sf_value = sf_scale_val * (vec_max * fp4_max_rcp)
                        sf_fp8_u32 = cvt_f32_to_e4m3(sf_value)
                        sf_fp8 = Uint8(sf_fp8_u32 & Uint32(0xFF))

                        # C++: output_scale = rcp(sf_float * rcp(sf_scale_val))
                        #    = sf_scale_val / sf_float
                        #    = rcp(sf_float) * sf_scale_val
                        # fp8_e4m3_to_f32_and_rcp returns 0 when input is 0
                        inv_sf = fp8_e4m3_to_f32_and_rcp(sf_fp8_u32)
                        output_scale = inv_sf * sf_scale_val

                        # ---- Convert to float32 with output_scale ----
                        if cutlass.const_expr(is_fp16):
                            y_f32 = half2_to_float16_scaled(h2, output_scale)
                        else:
                            y_f32 = bfloat2_to_float16_scaled(h2, output_scale)

                        # ---- Quantize to E2M1 and pack ----
                        packed64 = quantize_and_pack_16(y_f32)

                        # ---- R2G: Store packed FP4 output (64 bits) ----
                        # C++ R2G: AlignedArray<uint8_t, 8> = 64 bits
                        out_col = col // Int32(2)
                        out_ptr = get_ptr_as_int64(
                            mQuantOut,
                            global_row * (k // Int32(2)) + out_col,
                        )
                        st_global_u64(out_ptr, packed64)

                    # ---- Deferred TMA wait (C++ pattern) ----
                    # C++: if (t == 0 && threadIdx.x == 0)
                    #        cp_async_bulk_wait_group_read<0>();
                    # Wait for previous tile's TMA before writing new SF to smem.
                    if subtile == 0:
                        if is_first_tile == 0:
                            if tidx == 0:
                                cp_async_bulk_wait_group_read_0()
                    # __syncthreads() before SF write — matches C++
                    cute.arch.barrier()

                    # ---- R2S: Write SF to shared memory ----
                    # C++ R2S: 1 byte per thread, ThrLayout (16,4):(4,1)
                    # R2S logical thread = (t/8)*4 + (t%4)
                    # maps to SF tile position via the swizzled layout.
                    #
                    tile_local_row = subtile * Int32(16) + row_in_subtile
                    sf_logical_idx = sf_col * Int32(128) + tile_local_row

                    if is_sf_ktile0:
                        sf_smem_0[sf_logical_idx] = sf_fp8
                    else:
                        sf_smem_1[sf_logical_idx] = sf_fp8

                    # __syncthreads() after SF write — matches C++
                    cute.arch.barrier()

                # ---- Flush SF tiles to global via TMA bulk copy ----
                # C++: fence_proxy_async(space_shared); __syncthreads();
                fence_proxy_async_shared()
                cute.arch.barrier()

                if tidx == 0:
                    # SF K-tile indices in the zipped SF layout
                    sf_idx0 = m_tile * k_sf_tiles + k_tile * Int32(2)
                    sf_idx1 = sf_idx0 + Int32(1)

                    # Global byte offset: base_ptr + sf_idx * 512
                    # C++: scale_factor + blockscale_offset * (k/16) is the base
                    sf_base = blockscale_offset * (k // Int32(16))
                    sf_global_offset0 = sf_base + sf_idx0 * Int32(SF_TILE_BYTES)
                    sf_global_offset1 = sf_base + sf_idx1 * Int32(SF_TILE_BYTES)

                    dst0 = get_ptr_as_int64(mScaleFactor, sf_global_offset0)
                    src0 = sf_smem_0.iterator.toint()
                    cp_async_bulk_s2g(dst0, src0, Int32(SF_TILE_BYTES))

                    dst1 = get_ptr_as_int64(mScaleFactor, sf_global_offset1)
                    src1 = sf_smem_1.iterator.toint()
                    cp_async_bulk_s2g(dst1, src1, Int32(SF_TILE_BYTES))

                    cp_async_bulk_commit()

                # C++: __syncthreads() after TMA issue
                # The wait happens at the START of the next tile (deferred).
                cute.arch.barrier()
                is_first_tile = Int32(0)

                blk_offset = blk_offset + gdim


# =====================================================================
# PyTorch API
# =====================================================================


@functools.cache
def _get_compiled_kernel(is_fp16: bool):
    """Compile and cache the kernel for a given dtype."""
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16
    kernel_obj = NVFP4GroupQuantKernel(dtype=cutlass_dtype, is_fp16=is_fp16)

    # Symbolic sizes for dynamic dimensions
    sym_input_size = cute.sym_int()
    sym_output_size = cute.sym_int()
    sym_sf_size = cute.sym_int()
    sym_ps_size = cute.sym_int()
    sym_groups = cute.sym_int()

    # Create fake tensors for compilation (1D flat buffers)
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_input_size,), assumed_align=128
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        Uint8, (sym_output_size,), assumed_align=128
    )
    sf_fake = cute.runtime.make_fake_compact_tensor(
        Uint8, (sym_sf_size,), assumed_align=128
    )
    ps_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_ps_size,), assumed_align=4
    )
    eo_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_groups,), assumed_align=4
    )
    bso_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_groups,), assumed_align=4
    )
    sf_scale_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (sym_groups,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile(
        kernel_obj,
        input_fake,
        sf_scale_fake,
        output_fake,
        sf_fake,
        ps_fake,
        eo_fake,
        bso_fake,
        Int32(1),  # dummy num_groups
        Int32(1),  # dummy num_ctas
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        input_tensor: torch.Tensor,
        sf_scale: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        quant_output: torch.Tensor,
        scale_factor: torch.Tensor,
    ) -> None:
        num_groups = problem_sizes.shape[0]
        # Match C++: SM_count * max_active_blocks_per_sm
        # With 1024 bytes smem and 128 threads, occupancy is high.
        sm_count = torch.cuda.get_device_properties(
            input_tensor.device
        ).multi_processor_count
        # Conservative estimate: most SMs can run multiple blocks
        num_ctas = sm_count * 4
        compiled(
            input_tensor.flatten(),
            sf_scale.flatten(),
            quant_output.flatten().view(torch.uint8),
            scale_factor.flatten().view(torch.uint8),
            problem_sizes.flatten().contiguous(),
            expert_offsets.contiguous(),
            blockscale_offsets.contiguous(),
            Int32(num_groups),
            Int32(num_ctas),
        )

    return tensor_api


def cutedsl_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    sf_scale: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    quant_output: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    """
    Launch NVFP4 block-scaled grouped quantization kernel.

    Args:
        input_tensor: Flat input buffer in fp16/bf16, shape (total_rows, K)
        sf_scale: Global scale factor for SF computation
        problem_sizes: (num_groups, 3) tensor of (M, N, K) per expert group
        expert_offsets: (num_groups,) row offsets into input_tensor
        blockscale_offsets: (num_groups,) row offsets into scale_factor
        quant_output: Output buffer for packed FP4, shape (total_rows, K/2)
        scale_factor: Output buffer for E4M3 scale factors
    """
    is_fp16 = input_tensor.dtype == torch.float16
    tensor_api = _get_compiled_kernel(is_fp16)
    tensor_api(
        input_tensor,
        sf_scale,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )
