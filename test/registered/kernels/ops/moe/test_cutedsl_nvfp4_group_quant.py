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

import pytest
import torch

from sglang.kernels.ops.moe.cutedsl_nvfp4_group_quant import (
    cutedsl_scaled_fp4_experts_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _make_inputs(m_sizes, hidden_size, dtype):
    num_groups = len(m_sizes)
    total_rows = sum(m_sizes)
    input_tensor = torch.randn(total_rows, hidden_size, device="cuda", dtype=dtype)
    problem_sizes = torch.empty(num_groups, 3, device="cuda", dtype=torch.int32)
    expert_offsets = torch.empty(num_groups, device="cuda", dtype=torch.int32)
    blockscale_offsets = torch.empty(num_groups, device="cuda", dtype=torch.int32)

    row_offset = 0
    blockscale_offset = 0
    for group_idx, m_size in enumerate(m_sizes):
        problem_sizes[group_idx] = torch.tensor(
            [m_size, hidden_size, hidden_size], dtype=torch.int32
        )
        expert_offsets[group_idx] = row_offset
        blockscale_offsets[group_idx] = blockscale_offset
        row_offset += m_size
        blockscale_offset += ((m_size + 127) // 128) * 128

    quantized = torch.zeros(
        total_rows, hidden_size // 2, device="cuda", dtype=torch.uint8
    )
    scale_factor = torch.zeros(
        blockscale_offset * (hidden_size // 16),
        device="cuda",
        dtype=torch.uint8,
    )
    return (
        input_tensor,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quantized,
        scale_factor,
    )


def _scale_offset(
    blockscale_row: int, row: int, group_idx: int, hidden_size: int
) -> int:
    tile_idx = (row // 128) * (hidden_size // 64) + group_idx // 4
    row_in_tile = row % 128
    byte_in_tile = (row_in_tile % 32) * 16 + (row_in_tile // 32) * 4
    byte_in_tile += group_idx % 4
    return blockscale_row * (hidden_size // 16) + tile_idx * 512 + byte_in_tile


def _reference_codes(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = values.float() / scale.float().unsqueeze(-1)
    magnitude = scaled.abs().clamp(max=6.0)
    table = torch.tensor(E2M1_VALUES, device=values.device)
    codes = (magnitude.unsqueeze(-1) - table).abs().argmin(dim=-1).to(torch.uint8)
    return codes | ((scaled < 0).to(torch.uint8) << 3)


@pytest.mark.parametrize(
    "m_sizes,hidden_size,dtype,sf_scale_value",
    [
        ([128], 128, torch.float16, 1.0),
        ([200], 128, torch.bfloat16, 1.0),
        ([128, 256, 64], 256, torch.bfloat16, 0.1),
    ],
)
def test_cutedsl_nvfp4_group_quant_matches_reference(
    m_sizes, hidden_size, dtype, sf_scale_value
):
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("CuTe-DSL NVFP4 group quant requires an SM10x GPU")

    torch.manual_seed(20260928)
    inputs = _make_inputs(m_sizes, hidden_size, dtype)
    x, problem_sizes, expert_offsets, blockscale_offsets, output, output_scale = inputs
    sf_scale = torch.full(
        (len(m_sizes),), sf_scale_value, device="cuda", dtype=torch.float32
    )

    cutedsl_scaled_fp4_experts_quant(
        x,
        sf_scale,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        output,
        output_scale,
    )
    torch.cuda.synchronize()

    output_codes = torch.stack((output & 0x0F, output >> 4), dim=-1).flatten(1)
    row_offset = 0
    for expert_idx, m_size in enumerate(m_sizes):
        blockscale_row = int(blockscale_offsets[expert_idx])
        for row in range(m_size):
            global_row = row_offset + row
            blocks = x[global_row].float().view(-1, 16)
            expected_scale = (sf_scale_value * blocks.abs().amax(dim=1) / 6.0).to(
                torch.float8_e4m3fn
            )
            actual_scale = torch.stack(
                [
                    output_scale[
                        _scale_offset(blockscale_row, row, group_idx, hidden_size)
                    ]
                    for group_idx in range(hidden_size // 16)
                ]
            ).view(torch.float8_e4m3fn)
            assert torch.equal(
                actual_scale.view(torch.uint8), expected_scale.view(torch.uint8)
            )

            quant_scale = expected_scale.float() / sf_scale_value
            expected_codes = _reference_codes(blocks, quant_scale).flatten()
            differing = (output_codes[global_row] != expected_codes).sum().item()
            assert differing <= hidden_size * 0.05
        row_offset += m_size


def test_cutedsl_nvfp4_group_quant_zero_input():
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("CuTe-DSL NVFP4 group quant requires an SM10x GPU")

    inputs = _make_inputs([64], 128, torch.bfloat16)
    x, problem_sizes, expert_offsets, blockscale_offsets, output, output_scale = inputs
    x.zero_()
    sf_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    cutedsl_scaled_fp4_experts_quant(
        x,
        sf_scale,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        output,
        output_scale,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(output_scale) == 0
