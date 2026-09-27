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

"""FP8 (E4M3) Group Quantization + Scatter -- CuTe-DSL implementation.

Source-organized persistent kernel: each CTA reads ONE source BF16 row
(coalesced), quantizes to FP8 with per-128 group scales, and scatters the
quantized row + scales to `top_k` expert destinations via TMA bulk copies.

Key optimizations vs the Triton baseline (_fused_quant_scatter_kernel):
  * Compile-time specialization on (top_k, K, scale_ue8m0) -- full loop
    unrolling, no runtime division.
  * topk_ids / src2dst cached in registers per token (eliminates ~448
    redundant global loads per token).
  * Double-buffered shared memory + `cp.async.bulk.wait_group.read 1`
    overlaps TMA of token N-1 with quantization of token N.
  * 128-bit shared stores (`st.shared.v4.b32`) cut L1 transactions 4x.
  * Scale row (scale_size*4 bytes) written to smem once per group, then
    bulk-copied to global, avoiding top_k * scale_size scattered scalar
    stores per token.
  * Single-thread TMA issue + commit (PTX `commit_group` is per-thread;
    parallel issue without matching commits is unsafe).

Thread layout: 128 threads arranged as (16, 8):(8, 1)
  thread t -> group_in_batch = t / 8, k_col = t % 8
  16 groups per batch, 8 threads/group, each handling 16 elements.

Requires: CUTLASS 4.x with CuTe-DSL, an SM10x GPU, and CUDA 12.9+.
"""

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# =====================================================================
# Constants
# =====================================================================
FP8_GROUP_SIZE = 128
THREADS_PER_BLOCK = 128
THREADS_PER_GROUP = 8
GROUPS_PER_BATCH = THREADS_PER_BLOCK // THREADS_PER_GROUP  # 16
ELEMS_PER_THREAD = FP8_GROUP_SIZE // THREADS_PER_GROUP  # 16
FP8_E4M3_MAX = 448.0


# =====================================================================
# PTX Intrinsics
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
def st_global_f32(base_ptr: cutlass.Int64, val: Float32, *, loc=None, ip=None):
    """Global store with .cs (cache-streaming) hint.

    The scale writes are write-once-never-read within this kernel and the
    destination scale tensor is sparse (top_k different rows per source).
    Using .cs tells the memory system the cache line won't be reused, avoiding
    L1/L2 pollution from 400+ scattered stores per token.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.cs.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_v4_u32(
    smem_byte_addr: Int32,
    u0: Uint32,
    u1: Uint32,
    u2: Uint32,
    u3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Single 128-bit shared-memory store (one L1 transaction per 16 bytes)."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_byte_addr).ir_value(loc=loc, ip=ip),
            Uint32(u0).ir_value(loc=loc, ip=ip),
            Uint32(u1).ir_value(loc=loc, ip=ip),
            Uint32(u2).ir_value(loc=loc, ip=ip),
            Uint32(u3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_f32(smem_byte_addr: Int32, val: Float32, *, loc=None, ip=None):
    """Single 32-bit shared-memory store."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_byte_addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(
    tensor: cute.Tensor, offset: Int64, *, loc=None, ip=None
) -> cutlass.Int64:
    byte_offset = Int64(offset) * tensor.element_type.width // 8
    ptr_int = tensor.iterator.toint() + byte_offset
    return cutlass.Int64(ptr_int)


# ---- BFloat2 SIMD ----


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
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
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
        [
            Uint32(bf2).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
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


# ---- FP8 E4M3 conversion ----


@dsl_user_op
def cvt_f32x2_to_e4m3x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Uint32:
    """Two float32 -> two E4M3 packed in low 16 bits.
    Result[7:0] = e4m3(lo), Result[15:8] = e4m3(hi)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            """
        {
            .reg .b16 pair;
            cvt.rn.satfinite.e4m3x2.f32 pair, $1, $2;
            cvt.u32.u16 $0, pair;
        }
        """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ---- Warp shuffle & reduction ----


@dsl_user_op
def shfl_xor_f32(val: Float32, lane_mask: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(val).ir_value(loc=loc, ip=ip),
                Int32(lane_mask).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def max_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


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
def ue8m0_round(val: Float32, *, loc=None, ip=None) -> Float32:
    """Round a positive normal FP32 scale up to an exact power of two."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            """
        {
            .reg .pred has_mantissa;
            .reg .u32 bits, exponent, mantissa, increment, out_bits;
            mov.b32 bits, $1;
            and.b32 mantissa, bits, 0x007fffff;
            setp.ne.u32 has_mantissa, mantissa, 0;
            shr.u32 exponent, bits, 23;
            and.b32 exponent, exponent, 0x000000ff;
            selp.u32 increment, 1, 0, has_mantissa;
            add.u32 exponent, exponent, increment;
            max.u32 exponent, exponent, 1;
            min.u32 exponent, exponent, 254;
            shl.b32 out_bits, exponent, 23;
            mov.b32 $0, out_bits;
        }
        """,
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ---- TMA S2G (shared -> global) ----


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


@dsl_user_op
def cp_async_bulk_wait_group_read_1(*, loc=None, ip=None):
    """Wait until at most 1 bulk group is still reading (enables double buffer)."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group.read 1;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =====================================================================
# Helper Functions
# =====================================================================


@cute.jit
def load_16_bf16(mInput: cute.Tensor, row: Int32, col: Int32, K: Int32):
    """Load 16 BF16 as 8 Uint32 (bfloat2) via 2x 128-bit loads."""
    h2 = cute.make_rmem_tensor((8,), Uint32)
    offset = Int64(row) * Int64(K) + Int64(col)
    ptr0 = get_ptr_as_int64(mInput, offset)
    ptr1 = get_ptr_as_int64(mInput, offset + Int64(8))
    h2[0], h2[1], h2[2], h2[3] = ld_global_v4_u32(ptr0)
    h2[4], h2[5], h2[6], h2[7] = ld_global_v4_u32(ptr1)
    return h2


@cute.jit
def bf16_max_abs_16(h2: cute.Tensor) -> Float32:
    """Max |value| across 16 BF16 (8 bfloat2 pairs) -> float32."""
    abs_vals = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_vals[i] = bfloat2_habs2(h2[i])
    m01 = bfloat2_hmax2(abs_vals[0], abs_vals[1])
    m23 = bfloat2_hmax2(abs_vals[2], abs_vals[3])
    m45 = bfloat2_hmax2(abs_vals[4], abs_vals[5])
    m67 = bfloat2_hmax2(abs_vals[6], abs_vals[7])
    max_h2 = bfloat2_hmax2(bfloat2_hmax2(m01, m23), bfloat2_hmax2(m45, m67))
    return bfloat2_hmax_to_f32(max_h2)


@cute.jit
def reduce_max_8(val: Float32) -> Float32:
    """Butterfly max across 8 consecutive threads (XOR deltas 4,2,1)."""
    v = val
    v = max_f32(v, shfl_xor_f32(v, Int32(4)))
    v = max_f32(v, shfl_xor_f32(v, Int32(2)))
    v = max_f32(v, shfl_xor_f32(v, Int32(1)))
    return v


@cute.jit
def quant_16_to_smem_v4(
    h2: cute.Tensor,
    inv_scale: Float32,
    smem_byte_addr: Int32,
):
    """Scale 16 BF16 -> 16 E4M3, pack to 4 uint32, write to smem as one v4.b32."""
    pairs = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        f0, f1 = bfloat2_to_float2_scaled(h2[i], inv_scale)
        pairs[i] = cvt_f32x2_to_e4m3x2(f1, f0)

    u0 = (pairs[1] << Uint32(16)) | (pairs[0] & Uint32(0xFFFF))
    u1 = (pairs[3] << Uint32(16)) | (pairs[2] & Uint32(0xFFFF))
    u2 = (pairs[5] << Uint32(16)) | (pairs[4] & Uint32(0xFFFF))
    u3 = (pairs[7] << Uint32(16)) | (pairs[6] & Uint32(0xFFFF))

    st_shared_v4_u32(smem_byte_addr, u0, u1, u2, u3)


# =====================================================================
# Kernel Class
# =====================================================================


class FP8QuantScatterKernel:
    """Source-organized persistent kernel with double-buffered TMA scatter.

    Each CTA handles one token at a time. For that token, it:
      1. Reads one BF16 row from `mInput`.
      2. Splits the row into 128-element quant groups, computes one scale per
         group, and writes the quantized FP8 row plus the scale row into the
         current smem slot.
      3. Reuses the same smem slot as the source of `top_k` scatter copies:
         thread 0 sends the FP8 row and scale row to each destination row
         described by `src2dst`.
      4. Switches to the other smem slot for the next token, `wait_group.read`
         prevents overwriting a slot whose previous TMA copy is still in flight.

    `top_k`, `K`, and `scale_ue8m0` are compile-time constants selected in
    `_get_compiled_kernel`, so the group loops and top-k loop are fully unrolled.
    """

    def __init__(self, scale_ue8m0: bool, top_k: int, K: int):
        assert K % FP8_GROUP_SIZE == 0, "K must be multiple of 128"
        self.scale_ue8m0 = scale_ue8m0
        self.top_k = top_k  # 8
        self.K = K  # 7168
        self.scale_size = K // FP8_GROUP_SIZE  # 7168 // 128 = 56
        # 56 // 16 = 3
        self.num_full_group_batches = self.scale_size // GROUPS_PER_BATCH
        # 56 - 3 * 16 = 8
        self.num_remaining_groups = (
            self.scale_size - self.num_full_group_batches * GROUPS_PER_BATCH
        )
        # ((56 * 4 + 15) // 16) * 16 = 224 bytes (pad to 16B boundary)
        self.scale_bytes_per_slot = ((self.scale_size * 4 + 15) // 16) * 16
        # One smem slot packs [FP8 row, K bytes] + [scale row, padded bytes].
        # 7168 + 224 = 7392 bytes.
        self.slot_stride_bytes = self.K + self.scale_bytes_per_slot

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutputFP8: cute.Tensor,
        mOutputScale: cute.Tensor,
        mSrc2Dst: cute.Tensor,
        mTopkIds: cute.Tensor,
        num_tokens: Int32,
        num_ctas: Int32,
        stream,
    ):
        self.kernel(
            mInput,
            mOutputFP8,
            mOutputScale,
            mSrc2Dst,
            mTopkIds,
            num_tokens,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            smem=2 * self.slot_stride_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutputFP8: cute.Tensor,
        mOutputScale: cute.Tensor,
        mSrc2Dst: cute.Tensor,
        mTopkIds: cute.Tensor,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        group_in_batch = tidx // Int32(THREADS_PER_GROUP)  # 0..15
        k_col = tidx % Int32(THREADS_PER_GROUP)  # 0..7

        # Constexpr pulled out of `self` so codegen burns them as literals.
        scale_ue8m0 = self.scale_ue8m0
        top_k = self.top_k
        K_const = self.K
        scale_size = self.scale_size
        num_full_group_batches = self.num_full_group_batches
        num_remaining_groups = self.num_remaining_groups
        slot_stride_bytes = self.slot_stride_bytes
        scale_bytes = scale_size * 4  # payload size of one scale TMA copy

        rcp_e4m3_max = Float32(1.0 / FP8_E4M3_MAX)
        quant_eps = Float32(1.0e-10)

        # k_col_base: BF16 element offset (read side).
        # k_col_byte_off: FP8 byte offset in smem (write side).
        # Numerically equal, because each thread reads 16 BF16 elements
        # and writes the quantized result as 16 FP8 bytes.
        k_col_base = k_col * Int32(ELEMS_PER_THREAD)
        k_col_byte_off = k_col * Int32(ELEMS_PER_THREAD)

        # Two slots of (FP8 row | scale row) for double buffering
        total_smem_u32 = (2 * slot_stride_bytes) // 4  # (2*7392)//4=3696
        smem = cutlass.utils.SmemAllocator()
        smem_slots = smem.allocate_tensor(
            Uint32,
            cute.make_layout((total_smem_u32,)),
            byte_alignment=128,
        )
        smem_slots_base_int = smem_slots.iterator.toint()

        step = Int32(0)

        token_idx = bidx
        while token_idx < num_tokens:
            slot_id = step & Int32(1)
            smem_slot_base = smem_slots_base_int + slot_id * Int32(slot_stride_bytes)
            smem_fp8_addr = smem_slot_base
            smem_scale_addr = smem_slot_base + Int32(K_const)

            # Reusing a double-buffer slot: wait until the previous TMA read ready.
            if step >= Int32(2):
                if tidx == Int32(0):
                    cp_async_bulk_wait_group_read_1()
            cute.arch.barrier()

            # Load this token's routing once.
            topk_ids_r = cute.make_rmem_tensor((top_k,), Int32)
            src2dst_r = cute.make_rmem_tensor((top_k,), Int32)
            for k in cutlass.range_constexpr(top_k):
                topk_ids_r[k] = mTopkIds[token_idx * Int32(top_k) + Int32(k)]
                src2dst_r[k] = mSrc2Dst[token_idx * Int32(top_k) + Int32(k)]

            # Each CTA processes one token at a time. The K dimension has
            # K / FP8_GROUP_SIZE quant groups, handled GROUPS_PER_BATCH groups.
            # (K // FP8_GROUP_SIZE) // GROUPS_PER_BATCH = (7168//128)//16 = 3
            for group_batch in cutlass.range_constexpr(num_full_group_batches):
                batch_start = group_batch * GROUPS_PER_BATCH
                group_idx = Int32(batch_start) + group_in_batch
                col = group_idx * Int32(FP8_GROUP_SIZE) + k_col_base

                # 128 threads are split into GROUPS_PER_BATCH=16 8-thread groups;
                # each group covers one 128-element quant group,
                # and each thread loads 16 BF16 values.
                h2 = load_16_bf16(mInput, token_idx, col, K_const)
                # The 8 threads reduce their local maxima into one group amax.
                local_max = bf16_max_abs_16(h2)
                group_max = reduce_max_8(local_max)

                scale_val = max_f32(group_max, quant_eps) * rcp_e4m3_max
                if cutlass.const_expr(scale_ue8m0):
                    scale_val = ue8m0_round(scale_val)
                inv_scale = rcp_approx_ftz(scale_val)

                smem_write_addr = (
                    smem_fp8_addr + group_idx * Int32(FP8_GROUP_SIZE) + k_col_byte_off
                )
                quant_16_to_smem_v4(h2, inv_scale, smem_write_addr)

                # One smem store (by the group leader) replaces top_k scattered
                # 4B global stores; TMA carries the scale row at the end.
                if k_col == Int32(0):
                    st_shared_f32(smem_scale_addr + group_idx * Int32(4), scale_val)

            if cutlass.const_expr(num_remaining_groups > 0):
                group_idx_r = (
                    Int32(num_full_group_batches * GROUPS_PER_BATCH) + group_in_batch
                )
                if group_in_batch < Int32(num_remaining_groups):
                    col_r = group_idx_r * Int32(FP8_GROUP_SIZE) + k_col_base

                    h2r = load_16_bf16(mInput, token_idx, col_r, K_const)
                    local_max_r = bf16_max_abs_16(h2r)
                    group_max_r = reduce_max_8(local_max_r)

                    scale_val_r = max_f32(group_max_r, quant_eps) * rcp_e4m3_max
                    if cutlass.const_expr(scale_ue8m0):
                        scale_val_r = ue8m0_round(scale_val_r)
                    inv_scale_r = rcp_approx_ftz(scale_val_r)

                    smem_write_addr_r = (
                        smem_fp8_addr
                        + group_idx_r * Int32(FP8_GROUP_SIZE)
                        + k_col_byte_off
                    )
                    quant_16_to_smem_v4(h2r, inv_scale_r, smem_write_addr_r)

                    if k_col == Int32(0):
                        st_shared_f32(
                            smem_scale_addr + group_idx_r * Int32(4), scale_val_r
                        )

            # Make all smem stores visible to the async TMA proxy.
            cute.arch.barrier()
            fence_proxy_async_shared()

            # `cp.async.bulk.commit_group` only commits bulk copies issued by
            # the same thread, so thread 0 issues and commits all FP8/scale
            # copies here. That lets the next iteration's `wait_group.read`
            # drain the whole slot consistently before reuse.
            if tidx == Int32(0):
                for k in cutlass.range_constexpr(top_k):
                    eid_t = topk_ids_r[k]
                    if eid_t >= Int32(0):
                        dr_t = src2dst_r[k]
                        if dr_t >= Int32(0):
                            dr_i64 = Int64(dr_t)
                            fp8_dst_ptr = get_ptr_as_int64(
                                mOutputFP8, dr_i64 * Int64(K_const)
                            )
                            cp_async_bulk_s2g(
                                fp8_dst_ptr, smem_fp8_addr, Int32(K_const)
                            )
                            scale_dst_ptr = get_ptr_as_int64(
                                mOutputScale, dr_i64 * Int64(scale_size)
                            )
                            cp_async_bulk_s2g(
                                scale_dst_ptr, smem_scale_addr, Int32(scale_bytes)
                            )
                cp_async_bulk_commit()

            step = step + Int32(1)
            token_idx = token_idx + gdim

        # Drain remaining in-flight TMAs before CTA exits.
        if tidx == Int32(0):
            cp_async_bulk_wait_group_read_0()


# =====================================================================
# Compilation & PyTorch API
# =====================================================================


@functools.cache
def _get_compiled_kernel(scale_ue8m0: bool, top_k: int, K: int):
    kernel_obj = FP8QuantScatterKernel(scale_ue8m0=scale_ue8m0, top_k=top_k, K=K)

    sym_tokens = cute.sym_int()
    sym_rows = cute.sym_int()
    sym_routes = cute.sym_int()

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_tokens, K), stride_order=(1, 0), assumed_align=128
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        Uint8, (sym_rows, K), stride_order=(1, 0), assumed_align=128
    )
    scale_fake = cute.runtime.make_fake_compact_tensor(
        Float32,
        (sym_rows, K // FP8_GROUP_SIZE),
        stride_order=(1, 0),
        assumed_align=4,
    )
    s2d_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_routes,), assumed_align=4
    )
    tid_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_routes,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile(
        kernel_obj,
        input_fake,
        output_fake,
        scale_fake,
        s2d_fake,
        tid_fake,
        Int32(1),  # num_tokens
        Int32(1),  # num_ctas
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled


def cutedsl_fp8_quant_scatter(
    input_bf16: torch.Tensor,
    output_fp8: torch.Tensor,
    output_scale: torch.Tensor,
    src2dst: torch.Tensor,
    topk_ids: torch.Tensor,
    num_tokens: int,
    K: int,
    top_k: int,
    scale_ue8m0: bool = False,
) -> None:
    """Fused BF16 -> FP8(E4M3) group quantization + top_k scatter.

    Arguments:
        input_bf16:   [num_tokens, K] BF16 source hidden states.
        output_fp8:   [m_sum_upper, K] FP8 output buffer (written).
        output_scale: [m_sum_upper, K/128] FP32 scale buffer (written).
        src2dst:      [num_tokens * top_k] int32, source row -> dest row.
        topk_ids:     [num_tokens, top_k] int32 expert ids (-1 = skip).
        num_tokens:   logical source token count (<= input_bf16.shape[0]).
        K:            hidden dimension (must be multiple of 128).
        top_k:        number of destinations per source token.
        scale_ue8m0:  if True, round scales to powers of 2 (UE8M0 format).

    Grid is set to `SM * 4` blocks (the kernel is L1/memory bound).

    Each (scale_ue8m0, top_k, K) triggers a single JIT compile and cached.
    """
    if input_bf16.dtype != torch.bfloat16 or input_bf16.ndim != 2:
        raise ValueError("input_bf16 must be a 2D BF16 tensor")
    if not input_bf16.is_cuda or not input_bf16.is_contiguous():
        raise ValueError("input_bf16 must be a contiguous CUDA tensor")
    if output_fp8.dtype != torch.float8_e4m3fn or output_fp8.ndim != 2:
        raise ValueError("output_fp8 must be a 2D float8_e4m3fn tensor")
    if output_scale.dtype != torch.float32 or not output_scale.is_contiguous():
        raise ValueError("output_scale must be a contiguous FP32 tensor")
    if src2dst.dtype != torch.int32 or topk_ids.dtype != torch.int32:
        raise ValueError("src2dst and topk_ids must use int32")
    if K % (4 * FP8_GROUP_SIZE) != 0:
        raise ValueError("K must be divisible by 512 for 16-byte scale TMA copies")
    if tuple(input_bf16.shape) != (num_tokens, K):
        raise ValueError(
            f"input shape {tuple(input_bf16.shape)} does not match "
            f"(num_tokens={num_tokens}, K={K})"
        )
    if topk_ids.numel() != num_tokens * top_k:
        raise ValueError("topk_ids shape does not match num_tokens * top_k")
    if src2dst.numel() != topk_ids.numel():
        raise ValueError("src2dst and topk_ids must contain the same number of routes")

    compiled = _get_compiled_kernel(scale_ue8m0, int(top_k), int(K))
    sm_count = torch.cuda.get_device_properties(input_bf16.device).multi_processor_count
    num_ctas = min(sm_count * 4, max(num_tokens, sm_count))

    compiled(
        input_bf16,
        output_fp8.view(torch.uint8),
        output_scale,
        src2dst.flatten(),
        topk_ids.view(-1),
        Int32(num_tokens),
        Int32(num_ctas),
    )
