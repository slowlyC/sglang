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

import sglang.kernels.ops.moe.deepgemm_contiguous_preprocess as preprocess
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _run_preprocess(use_cutedsl: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        preprocess,
        "_should_use_nvfp4_cutedsl",
        lambda _all_tokens, _hidden_size: use_cutedsl,
    )

    topk_ids = torch.tensor(
        [
            [4, 5, -1],
            [6, 9, 4],
            [3, 5, 6],
            [5, 4, 6],
            [-1, 8, 4],
            [6, 5, 3],
            [4, 6, 5],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    torch.manual_seed(20260928)
    hidden_states = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16) * 3.0
    input_global_scale = torch.tensor([128.0], device="cuda", dtype=torch.float32)

    result = preprocess.moe_tp_deepgemm_nvfp4_preprocess(
        topk_ids,
        num_local_experts=3,
        hidden_states=hidden_states,
        top_k=3,
        input_global_scale=input_global_scale,
        block_shape=(1, 16),
        expert_alignment=128,
        all_tokens=384,
        expert_start=4,
    )
    return topk_ids, result


def test_nvfp4_cutedsl_matches_triton_fallback(monkeypatch: pytest.MonkeyPatch):
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("CuTe-DSL NVFP4 scatter requires an SM10x GPU")

    topk_ids, triton_result = _run_preprocess(False, monkeypatch)
    _, cutedsl_result = _run_preprocess(True, monkeypatch)
    triton_fp4, triton_scale, triton_m_indices, triton_src2dst = triton_result
    cutedsl_fp4, cutedsl_scale, cutedsl_m_indices, cutedsl_src2dst = cutedsl_result

    assert triton_fp4.shape == cutedsl_fp4.shape == (384, 256)
    assert triton_scale.shape == cutedsl_scale.shape == (384, 8)
    assert triton_scale.dtype == cutedsl_scale.dtype == torch.int32
    assert triton_scale.stride() == cutedsl_scale.stride() == (1, 384)
    torch.testing.assert_close(triton_m_indices, cutedsl_m_indices, rtol=0, atol=0)

    local_route_mask = (topk_ids >= 4) & (topk_ids < 7)
    assert torch.all(triton_src2dst[~local_route_mask] == -1)
    assert torch.all(cutedsl_src2dst[~local_route_mask] == -1)

    for token_idx, route_idx in local_route_mask.nonzero().tolist():
        triton_row = int(triton_src2dst[token_idx, route_idx])
        cutedsl_row = int(cutedsl_src2dst[token_idx, route_idx])
        assert torch.equal(
            triton_fp4[triton_row].view(torch.uint8),
            cutedsl_fp4[cutedsl_row].view(torch.uint8),
        )
        assert torch.equal(
            triton_scale[triton_row].contiguous().view(torch.uint8),
            cutedsl_scale[cutedsl_row].contiguous().view(torch.uint8),
        )


def test_nvfp4_empty_routes_return_packed_scale(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        preprocess,
        "_should_use_nvfp4_cutedsl",
        lambda _all_tokens, _hidden_size: True,
    )
    hidden_states = torch.empty((0, 512), device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.empty((0, 2), device="cuda", dtype=torch.int32)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

    output_fp4, output_scale, m_indices, src2dst = (
        preprocess.moe_tp_deepgemm_nvfp4_preprocess(
            topk_ids,
            num_local_experts=2,
            hidden_states=hidden_states,
            top_k=2,
            input_global_scale=global_scale,
            expert_alignment=128,
            all_tokens=0,
        )
    )

    assert output_fp4.shape == (0, 256)
    assert output_scale.shape == (0, 8)
    assert output_scale.dtype == torch.int32
    assert m_indices.shape == (0,)
    assert src2dst.shape == (0, 2)
