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

"""Fused FP8 and NVFP4 preprocessing for DeepGEMM contiguous grouped GEMM."""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

FP8_GROUP_SIZE = 128
NVFP4_GROUP_SIZE = 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@functools.cache
def _is_cutedsl_available_on_current_device() -> bool:
    """Return whether the optimized scatter kernel can run on this device."""
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            return False
        from sglang.kernels.ops.moe.cutedsl_fp8_quant_scatter import (  # noqa: F401
            cutedsl_fp8_quant_scatter,
        )

        return True
    except (ImportError, RuntimeError):
        return False


def _should_use_cutedsl(all_tokens: int, hidden_size: int) -> bool:
    # TMA bulk copies require 16-byte payloads. Four FP32 scale values cover
    # 512 activation values, so smaller K alignments use the Triton fallback.
    return (
        all_tokens > 0
        and hidden_size % (4 * FP8_GROUP_SIZE) == 0
        and _is_cutedsl_available_on_current_device()
    )


@functools.cache
def _is_nvfp4_cutedsl_available_on_current_device() -> bool:
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            return False
        from sglang.kernels.ops.moe.cutedsl_nvfp4_quant_scatter import (  # noqa: F401
            cutedsl_nvfp4_quant_scatter,
        )

        return True
    except (ImportError, RuntimeError):
        return False


def _should_use_nvfp4_cutedsl(all_tokens: int, hidden_size: int) -> bool:
    return (
        all_tokens > 0
        and hidden_size % (4 * NVFP4_GROUP_SIZE) == 0
        and _is_nvfp4_cutedsl_available_on_current_device()
    )


@triton.jit
def _count_and_local_offset_kernel(
    flat_ids,
    expert_counts,
    src2dst,
    num_routes,
    expert_start,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count local routes and retain each route's offset within its expert."""
    route_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    route_mask = route_offsets < num_routes
    global_expert = tl.load(flat_ids + route_offsets, mask=route_mask, other=-1)
    local_expert = global_expert - expert_start
    is_local = route_mask & (local_expert >= 0) & (local_expert < NUM_LOCAL_EXPERTS)

    tl.store(src2dst + route_offsets, -1, mask=route_mask)
    local_offset = tl.atomic_add(expert_counts + local_expert, 1, mask=is_local)
    tl.store(src2dst + route_offsets, local_offset.to(tl.int32), mask=is_local)


@triton.jit
def _prefix_and_fill_kernel(
    expert_counts,
    aligned_offsets,
    m_indices,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_ALIGNMENT: tl.constexpr,
    BLOCK_E: tl.constexpr,
    FILL_BLOCK: tl.constexpr,
):
    """Build aligned expert offsets and fill live rows in m_indices."""
    expert_id = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < NUM_EXPERTS
    counts = tl.load(expert_counts + expert_offsets, mask=expert_mask, other=0)
    aligned_counts = (
        (counts + EXPERT_ALIGNMENT - 1) // EXPERT_ALIGNMENT
    ) * EXPERT_ALIGNMENT
    starts = tl.cumsum(aligned_counts) - aligned_counts

    if expert_id == 0:
        tl.store(aligned_offsets + expert_offsets, starts, mask=expert_mask)
        tl.store(aligned_offsets + NUM_EXPERTS, tl.sum(aligned_counts))

    count = tl.sum(tl.where(expert_offsets == expert_id, counts, 0))
    if count == 0:
        return
    start = tl.sum(tl.where(expert_offsets == expert_id, starts, 0))
    fill_offsets = tl.arange(0, FILL_BLOCK)
    for base in tl.range(0, count, FILL_BLOCK):
        offsets = base + fill_offsets
        tl.store(
            m_indices + start + offsets,
            expert_id,
            mask=offsets < count,
        )


@triton.jit
def _fixup_src2dst_kernel(
    flat_ids,
    aligned_offsets,
    src2dst,
    num_routes,
    expert_start,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert per-expert route offsets into absolute compact-layout rows."""
    route_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    route_mask = route_offsets < num_routes
    global_expert = tl.load(flat_ids + route_offsets, mask=route_mask, other=-1)
    local_expert = global_expert - expert_start
    is_local = route_mask & (local_expert >= 0) & (local_expert < NUM_LOCAL_EXPERTS)
    expert_start_row = tl.load(aligned_offsets + local_expert, mask=is_local, other=0)
    local_offset = tl.load(src2dst + route_offsets, mask=is_local, other=0)
    tl.store(
        src2dst + route_offsets,
        (expert_start_row + local_offset).to(tl.int32),
        mask=is_local,
    )


@triton.jit
def _ceil_to_ue8m0_power2(value):
    """Round a positive normal FP32 value up to an exact power of two."""
    return tl.inline_asm_elementwise(
        asm="""
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
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quant_scatter_kernel(
    input_bf16,
    output_fp8,
    output_scale,
    src2dst,
    top_k: tl.constexpr,
    hidden_size,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """Quantize one source row and scatter it to all of its local routes."""
    token_idx = tl.program_id(0).to(tl.int64)
    route_base = src2dst + token_idx * top_k
    input_row = input_bf16 + token_idx * hidden_size
    num_groups = hidden_size // GROUP_SIZE
    group_offsets = tl.arange(0, GROUP_SIZE)

    for block_idx in tl.range(0, tl.cdiv(num_groups, GROUPS_PER_BLOCK), num_stages=2):
        for group_in_block in tl.static_range(GROUPS_PER_BLOCK):
            group_idx = block_idx * GROUPS_PER_BLOCK + group_in_block
            valid_group = group_idx < num_groups
            column_offsets = group_idx * GROUP_SIZE + group_offsets
            values = tl.load(
                input_row + column_offsets,
                mask=valid_group,
                other=0.0,
            ).to(tl.float32)

            amax = tl.maximum(tl.max(tl.abs(values)), 1.0e-10)
            scale = amax / 448.0
            if SCALE_UE8M0:
                scale = _ceil_to_ue8m0_power2(scale)
            quantized = tl.clamp(values / scale, -448.0, 448.0).to(tl.float8e4nv)

            for route_idx in range(top_k):
                dst_idx = tl.load(route_base + route_idx).to(tl.int64)
                if dst_idx >= 0:
                    output_row = output_fp8 + dst_idx * hidden_size
                    tl.store(
                        output_row + column_offsets,
                        quantized,
                        mask=valid_group,
                    )
                    tl.store(
                        output_scale + dst_idx * num_groups + group_idx,
                        scale,
                        mask=valid_group,
                    )


@triton.jit
def _quantize_to_fp4_e2m1_code(value):
    magnitude = tl.minimum(tl.abs(value), 6.0)
    code = (magnitude > 0.25).to(tl.int32)
    code += (magnitude > 0.75).to(tl.int32)
    code += (magnitude > 1.25).to(tl.int32)
    code += (magnitude > 1.75).to(tl.int32)
    code += (magnitude > 2.5).to(tl.int32)
    code += (magnitude > 3.5).to(tl.int32)
    code += (magnitude > 5.0).to(tl.int32)
    sign = (value < 0.0) & (code != 0)
    return code | (sign.to(tl.int32) << 3)


@triton.jit
def _nvfp4_quant_scatter_kernel(
    input_bf16,
    input_global_scale,
    output_fp4,
    output_scale,
    src2dst,
    top_k: tl.constexpr,
    hidden_size,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """Quantize one BF16 source row to NVFP4 and scatter its local routes."""
    token_idx = tl.program_id(0).to(tl.int64)
    route_base = src2dst + token_idx * top_k
    input_row = input_bf16 + token_idx * hidden_size
    num_groups = hidden_size // GROUP_SIZE
    group_offsets = tl.arange(0, GROUP_SIZE)
    pair_offsets = tl.arange(0, GROUP_SIZE // 2)
    global_scale = tl.load(input_global_scale).to(tl.float32)

    for block_idx in tl.range(0, tl.cdiv(num_groups, GROUPS_PER_BLOCK), num_stages=2):
        for group_in_block in tl.static_range(GROUPS_PER_BLOCK):
            group_idx = block_idx * GROUPS_PER_BLOCK + group_in_block
            valid_group = group_idx < num_groups
            column_base = group_idx * GROUP_SIZE
            values = tl.load(
                input_row + column_base + group_offsets,
                mask=valid_group,
                other=0.0,
            ).to(tl.float32)

            scaled_values = values * global_scale
            amax = tl.maximum(tl.max(tl.abs(scaled_values)), 1.0e-12)
            scale = (amax / 6.0).to(tl.float8e4nv)
            scale_f32 = scale.to(tl.float32)
            low, high = tl.split(tl.reshape(scaled_values, (GROUP_SIZE // 2, 2)))
            low_code = _quantize_to_fp4_e2m1_code(low / scale_f32)
            high_code = _quantize_to_fp4_e2m1_code(high / scale_f32)
            packed = (low_code & 0x0F) | ((high_code & 0x0F) << 4)

            for route_idx in range(top_k):
                dst_idx = tl.load(route_base + route_idx).to(tl.int64)
                if dst_idx >= 0:
                    output_row = output_fp4 + dst_idx * (hidden_size // 2)
                    tl.store(
                        output_row + column_base // 2 + pair_offsets,
                        packed.to(tl.int8),
                        mask=valid_group,
                    )
                    tl.store(
                        output_scale + dst_idx * num_groups + group_idx,
                        scale,
                        mask=valid_group,
                    )


def _empty_scale(
    rows: int,
    num_groups: int,
    device: torch.device,
    scale_ue8m0: bool,
) -> torch.Tensor:
    if scale_ue8m0:
        return torch.empty(
            (_ceil_div(num_groups, 4), rows),
            device=device,
            dtype=torch.int32,
        ).transpose(0, 1)
    return torch.empty((rows, num_groups), device=device, dtype=torch.float32)


def moe_tp_deepgemm_preprocess(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    hidden_states: torch.Tensor,
    top_k: int,
    block_shape: Optional[Sequence[int]] = None,
    *,
    expert_alignment: int,
    all_tokens: int,
    expert_start: int = 0,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and quantize DeepGEMM's compact expert layout in one pass.

    The caller owns the compact-layout policy and supplies expert_alignment
    and its routing-independent all_tokens upper bound. Global expert IDs
    outside [expert_start, expert_start + num_local_experts) map to -1.

    Returns (fp8_input, input_scale, m_indices, src2dst). Scales are already
    in the TMA-aligned format consumed by DeepGEMM.
    """
    if topk_ids.ndim != 2 or topk_ids.shape[1] != top_k:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be a contiguous int32 tensor")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be a 2D BF16 tensor")
    if not hidden_states.is_cuda or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be a contiguous CUDA tensor")
    if hidden_states.device != topk_ids.device:
        raise ValueError("hidden_states and topk_ids must be on the same device")
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError("hidden_states and topk_ids must have the same token count")
    if num_local_experts <= 0 or expert_alignment <= 0 or all_tokens < 0:
        raise ValueError(
            "expert count/alignment must be positive and all_tokens nonnegative"
        )

    block_k = block_shape[1] if block_shape else FP8_GROUP_SIZE
    if block_k != FP8_GROUP_SIZE:
        raise ValueError("fused DeepGEMM preprocessing requires FP8 group size 128")

    num_tokens, hidden_size = hidden_states.shape
    if hidden_size % block_k != 0:
        raise ValueError("hidden size must be divisible by the FP8 group size")
    num_groups = hidden_size // block_k
    num_routes = topk_ids.numel()

    output_fp8 = torch.empty(
        (all_tokens, hidden_size),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    m_indices = torch.full(
        (all_tokens,), -1, device=hidden_states.device, dtype=torch.int32
    )
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)

    if num_routes == 0:
        return (
            output_fp8,
            _empty_scale(all_tokens, num_groups, hidden_states.device, scale_ue8m0),
            m_indices,
            src2dst,
        )

    expert_counts = torch.zeros(
        num_local_experts, device=hidden_states.device, dtype=torch.int32
    )
    aligned_offsets = torch.empty(
        num_local_experts + 1,
        device=hidden_states.device,
        dtype=torch.int32,
    )

    route_block = 64 if num_routes < 8192 else 256
    route_grid = (triton.cdiv(num_routes, route_block),)
    _count_and_local_offset_kernel[route_grid](
        topk_ids,
        expert_counts,
        src2dst,
        num_routes,
        expert_start,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=route_block,
    )

    _prefix_and_fill_kernel[(num_local_experts,)](
        expert_counts,
        aligned_offsets,
        m_indices,
        NUM_EXPERTS=num_local_experts,
        EXPERT_ALIGNMENT=expert_alignment,
        BLOCK_E=triton.next_power_of_2(num_local_experts),
        FILL_BLOCK=1024,
    )
    _fixup_src2dst_kernel[route_grid](
        topk_ids,
        aligned_offsets,
        src2dst,
        num_routes,
        expert_start,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=route_block,
    )

    # The UE8M0 packer transforms the entire conservative buffer, including
    # expert padding and its unused tail. Initialize those scale rows to zero.
    output_scale = (torch.zeros if scale_ue8m0 else torch.empty)(
        (all_tokens, num_groups),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    if _should_use_cutedsl(all_tokens, hidden_size):
        from sglang.kernels.ops.moe.cutedsl_fp8_quant_scatter import (
            cutedsl_fp8_quant_scatter,
        )

        cutedsl_fp8_quant_scatter(
            hidden_states,
            output_fp8,
            output_scale,
            src2dst,
            topk_ids,
            num_tokens,
            hidden_size,
            top_k,
            scale_ue8m0=scale_ue8m0,
        )
    else:
        _quant_scatter_kernel[(num_tokens,)](
            hidden_states,
            output_fp8,
            output_scale,
            src2dst,
            top_k=top_k,
            hidden_size=hidden_size,
            GROUP_SIZE=block_k,
            GROUPS_PER_BLOCK=4,
            SCALE_UE8M0=scale_ue8m0,
        )

    if scale_ue8m0:
        from deep_gemm.utils.layout import (
            get_mn_major_tma_aligned_packed_ue8m0_tensor,
        )

        output_scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(output_scale)
    else:
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale

        output_scale = tma_align_input_scale(output_scale)

    return output_fp8, output_scale, m_indices, src2dst


def _pack_e4m3_scale_to_int32_mn_major(scale: torch.Tensor) -> torch.Tensor:
    """Pack four adjacent E4M3 scales into DeepGEMM's MN-major int32 view."""
    if scale.dtype != torch.float8_e4m3fn or scale.ndim != 2:
        raise ValueError("NVFP4 scales must be a 2D E4M3 tensor")
    if scale.shape[1] % 4 != 0:
        raise ValueError("NVFP4 scale width must be divisible by four")

    packed = (
        scale.contiguous().view(torch.int32).view(scale.shape[0], scale.shape[1] // 4)
    )
    return packed.transpose(0, 1).contiguous().transpose(0, 1)


def moe_tp_deepgemm_nvfp4_preprocess(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    hidden_states: torch.Tensor,
    top_k: int,
    input_global_scale: torch.Tensor,
    block_shape: Optional[Sequence[int]] = None,
    *,
    expert_alignment: int,
    all_tokens: int,
    expert_start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DeepGEMM's compact expert layout and quantize it to NVFP4.

    The output activation stores two E2M1 values per byte. Every 16 input
    values share one E4M3 scale, and four adjacent scale bytes are returned as
    an int32 MN-major tensor. The scalar FP32 input_global_scale follows the
    ModelOpt NVFP4 convention.

    Returns (fp4_input, packed_input_scale, m_indices, src2dst).
    """
    if topk_ids.ndim != 2 or topk_ids.shape[1] != top_k:
        raise ValueError("topk_ids must have shape [num_tokens, top_k]")
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be a contiguous int32 tensor")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be a 2D BF16 tensor")
    if not hidden_states.is_cuda or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be a contiguous CUDA tensor")
    if hidden_states.device != topk_ids.device:
        raise ValueError("hidden_states and topk_ids must be on the same device")
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError("hidden_states and topk_ids must have the same token count")
    if (
        input_global_scale.dtype != torch.float32
        or input_global_scale.numel() != 1
        or input_global_scale.device != hidden_states.device
    ):
        raise ValueError(
            "input_global_scale must be one FP32 value on the input device"
        )
    if num_local_experts <= 0 or expert_alignment <= 0 or all_tokens < 0:
        raise ValueError(
            "expert count/alignment must be positive and all_tokens nonnegative"
        )

    block_k = block_shape[1] if block_shape else NVFP4_GROUP_SIZE
    if block_k != NVFP4_GROUP_SIZE:
        raise ValueError("NVFP4 DeepGEMM preprocessing requires group size 16")

    num_tokens, hidden_size = hidden_states.shape
    if hidden_size % (4 * NVFP4_GROUP_SIZE) != 0:
        raise ValueError("hidden size must be divisible by 64 for packed NVFP4 scales")
    num_groups = hidden_size // NVFP4_GROUP_SIZE
    num_routes = topk_ids.numel()

    output_fp4 = torch.empty(
        (all_tokens, hidden_size // 2),
        device=hidden_states.device,
        dtype=torch.int8,
    )
    output_scale_e4m3 = torch.zeros(
        (all_tokens, num_groups),
        device=hidden_states.device,
        dtype=torch.float8_e4m3fn,
    )
    m_indices = torch.full(
        (all_tokens,), -1, device=hidden_states.device, dtype=torch.int32
    )
    src2dst = torch.empty_like(topk_ids, dtype=torch.int32)

    if num_routes == 0:
        return (
            output_fp4,
            _pack_e4m3_scale_to_int32_mn_major(output_scale_e4m3),
            m_indices,
            src2dst,
        )

    expert_counts = torch.zeros(
        num_local_experts, device=hidden_states.device, dtype=torch.int32
    )
    aligned_offsets = torch.empty(
        num_local_experts + 1,
        device=hidden_states.device,
        dtype=torch.int32,
    )

    route_block = 64 if num_routes < 8192 else 256
    route_grid = (triton.cdiv(num_routes, route_block),)
    _count_and_local_offset_kernel[route_grid](
        topk_ids,
        expert_counts,
        src2dst,
        num_routes,
        expert_start,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=route_block,
    )
    _prefix_and_fill_kernel[(num_local_experts,)](
        expert_counts,
        aligned_offsets,
        m_indices,
        NUM_EXPERTS=num_local_experts,
        EXPERT_ALIGNMENT=expert_alignment,
        BLOCK_E=triton.next_power_of_2(num_local_experts),
        FILL_BLOCK=1024,
    )
    _fixup_src2dst_kernel[route_grid](
        topk_ids,
        aligned_offsets,
        src2dst,
        num_routes,
        expert_start,
        NUM_LOCAL_EXPERTS=num_local_experts,
        BLOCK_SIZE=route_block,
    )

    input_global_scale = input_global_scale.contiguous().view(-1)
    if _should_use_nvfp4_cutedsl(all_tokens, hidden_size):
        from sglang.kernels.ops.moe.cutedsl_nvfp4_quant_scatter import (
            cutedsl_nvfp4_quant_scatter,
        )

        cutedsl_nvfp4_quant_scatter(
            hidden_states,
            input_global_scale,
            output_fp4,
            output_scale_e4m3,
            src2dst,
            num_tokens,
            hidden_size,
            top_k,
        )
    else:
        _nvfp4_quant_scatter_kernel[(num_tokens,)](
            hidden_states,
            input_global_scale,
            output_fp4,
            output_scale_e4m3,
            src2dst,
            top_k=top_k,
            hidden_size=hidden_size,
            GROUP_SIZE=NVFP4_GROUP_SIZE,
            GROUPS_PER_BLOCK=4,
        )

    return (
        output_fp4,
        _pack_e4m3_scale_to_int32_mn_major(output_scale_e4m3),
        m_indices,
        src2dst,
    )
