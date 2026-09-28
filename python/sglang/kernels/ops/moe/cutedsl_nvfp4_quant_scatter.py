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

"""Fused NVFP4 group quantization and scatter for contiguous MoE."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from sglang.kernels.ops.moe.cutedsl_fp8_quant_scatter import (
    bf16_max_abs_16,
    get_ptr_as_int64,
    load_16_bf16,
    max_f32,
)
from sglang.kernels.ops.moe.cutedsl_nvfp4_common import (
    E2M1_MAX,
    NVFP4_GROUP_SIZE,
    THREADS_PER_BLOCK,
    abs_f32,
    cvt_f32_to_e4m3,
    e4m3_to_f32_and_rcp,
    quant_16_bf16_to_fp4_u64,
    st_global_u64,
)

GROUPS_PER_BATCH = THREADS_PER_BLOCK


@dsl_user_op
def st_global_u8(base_ptr: cutlass.Int64, value: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u8 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class NVFP4QuantScatterKernel:
    def __init__(self, top_k: int, hidden_size: int):
        assert hidden_size % NVFP4_GROUP_SIZE == 0
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.fp4_row_bytes = hidden_size // 2
        self.scale_size = hidden_size // NVFP4_GROUP_SIZE
        self.full_batches = self.scale_size // GROUPS_PER_BATCH
        self.remaining_groups = self.scale_size % GROUPS_PER_BATCH

    @cute.jit
    def __call__(
        self,
        input_bf16: cute.Tensor,
        input_global_scale: cute.Tensor,
        output_fp4: cute.Tensor,
        output_scale: cute.Tensor,
        src2dst: cute.Tensor,
        num_tokens: Int32,
        stream,
    ):
        self.kernel(
            input_bf16,
            input_global_scale,
            output_fp4,
            output_scale,
            src2dst,
            num_tokens,
        ).launch(
            grid=[num_tokens, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_bf16: cute.Tensor,
        input_global_scale: cute.Tensor,
        output_fp4: cute.Tensor,
        output_scale: cute.Tensor,
        src2dst: cute.Tensor,
        num_tokens: Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()

        global_scale = input_global_scale[Int32(0)]
        global_scale_abs = abs_f32(global_scale)
        min_amax = Float32(1.0e-12)

        for batch in cutlass.range_constexpr(self.full_batches):
            group_idx = Int32(batch * GROUPS_PER_BATCH) + thread_idx
            self.quantize_and_scatter_group(
                input_bf16,
                output_fp4,
                output_scale,
                src2dst,
                token_idx,
                group_idx,
                global_scale,
                global_scale_abs,
                min_amax,
            )

        if cutlass.const_expr(self.remaining_groups > 0):
            group_idx = Int32(self.full_batches * GROUPS_PER_BATCH) + thread_idx
            if thread_idx < Int32(self.remaining_groups):
                self.quantize_and_scatter_group(
                    input_bf16,
                    output_fp4,
                    output_scale,
                    src2dst,
                    token_idx,
                    group_idx,
                    global_scale,
                    global_scale_abs,
                    min_amax,
                )

    @cute.jit
    def quantize_and_scatter_group(
        self,
        input_bf16: cute.Tensor,
        output_fp4: cute.Tensor,
        output_scale: cute.Tensor,
        src2dst: cute.Tensor,
        token_idx: Int32,
        group_idx: Int32,
        global_scale: Float32,
        global_scale_abs: Float32,
        min_amax: Float32,
    ):
        column = group_idx * Int32(NVFP4_GROUP_SIZE)
        values = load_16_bf16(input_bf16, token_idx, column, Int32(self.hidden_size))
        local_max = bf16_max_abs_16(values) * global_scale_abs
        amax = max_f32(local_max, min_amax)
        scale_u32 = cvt_f32_to_e4m3(amax * Float32(1.0 / E2M1_MAX))
        inv_scale = e4m3_to_f32_and_rcp(scale_u32) * global_scale
        packed = quant_16_bf16_to_fp4_u64(values, inv_scale)
        scale_u8 = scale_u32 & Uint32(0xFF)

        for route_idx in cutlass.range_constexpr(self.top_k):
            dst_row = src2dst[token_idx * Int32(self.top_k) + Int32(route_idx)]
            if dst_row >= Int32(0):
                dst_row_i64 = Int64(dst_row)
                output_offset = dst_row_i64 * Int64(self.fp4_row_bytes) + Int64(
                    group_idx
                ) * Int64(NVFP4_GROUP_SIZE // 2)
                st_global_u64(get_ptr_as_int64(output_fp4, output_offset), packed)
                scale_offset = dst_row_i64 * Int64(self.scale_size) + Int64(group_idx)
                st_global_u8(get_ptr_as_int64(output_scale, scale_offset), scale_u8)


@functools.cache
def _get_compiled_kernel(top_k: int, hidden_size: int):
    kernel = NVFP4QuantScatterKernel(top_k, hidden_size)
    sym_tokens = cute.sym_int()
    sym_rows = cute.sym_int()
    sym_routes = cute.sym_int()

    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_tokens, hidden_size),
        stride_order=(1, 0),
        assumed_align=128,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        Float32, (1,), assumed_align=4
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        Uint8,
        (sym_rows, hidden_size // 2),
        stride_order=(1, 0),
        assumed_align=128,
    )
    output_scale_fake = cute.runtime.make_fake_compact_tensor(
        Uint8,
        (sym_rows, hidden_size // NVFP4_GROUP_SIZE),
        stride_order=(1, 0),
        assumed_align=4,
    )
    src2dst_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_routes,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    options = os.environ.get(
        "SGLANG_CUTEDSL_NVFP4_OPTIONS", "--enable-tvm-ffi --opt-level=3"
    )
    return cute.compile(
        kernel,
        input_fake,
        global_scale_fake,
        output_fake,
        output_scale_fake,
        src2dst_fake,
        Int32(1),
        stream_fake,
        options=options,
    )


def cutedsl_nvfp4_quant_scatter(
    input_bf16: torch.Tensor,
    input_global_scale: torch.Tensor,
    output_fp4: torch.Tensor,
    output_scale: torch.Tensor,
    src2dst: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
) -> None:
    """Quantize BF16 rows to packed NVFP4 and scatter each local route."""
    if input_bf16.dtype != torch.bfloat16:
        raise ValueError("input_bf16 must use BF16")
    if input_global_scale.dtype != torch.float32 or input_global_scale.numel() != 1:
        raise ValueError("input_global_scale must be one FP32 value")
    if output_fp4.dtype not in (torch.int8, torch.uint8):
        raise ValueError("output_fp4 must use int8 or uint8 storage")
    if output_scale.dtype != torch.float8_e4m3fn:
        raise ValueError("output_scale must use E4M3 storage")
    if hidden_size % NVFP4_GROUP_SIZE != 0:
        raise ValueError("hidden_size must be divisible by 16")

    compiled = _get_compiled_kernel(int(top_k), int(hidden_size))
    compiled(
        input_bf16,
        input_global_scale.contiguous().view(-1),
        output_fp4.view(torch.uint8),
        output_scale.view(torch.uint8),
        src2dst.flatten(),
        Int32(num_tokens),
    )
