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

"""Small CuTe-DSL helpers shared by NVFP4 MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.ops.moe.cutedsl_fp8_quant_scatter import (
    bfloat2_to_float2_scaled,
)

E2M1_MAX = 6.0
NVFP4_GROUP_SIZE = 16
THREADS_PER_BLOCK = 128


@dsl_user_op
def abs_f32(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f32_to_e4m3(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert FP32 to one E4M3 value in the result's low byte."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
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
def e4m3_to_f32_and_rcp(value: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert a positive normal E4M3 value to FP32 and return its reciprocal."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_zero;
                .reg .u32 exponent, mantissa;
                .reg .s32 unbiased_exponent;
                .reg .f32 exponent_f32, mantissa_f32, decoded, reciprocal;

                setp.eq.u32 is_zero, $1, 0;
                and.b32 mantissa, $1, 7;
                shr.b32 exponent, $1, 3;
                and.b32 exponent, exponent, 15;
                sub.s32 unbiased_exponent, exponent, 7;
                cvt.rn.f32.s32 exponent_f32, unbiased_exponent;
                ex2.approx.f32 exponent_f32, exponent_f32;
                cvt.rn.f32.u32 mantissa_f32, mantissa;
                fma.rn.f32 mantissa_f32, mantissa_f32, 0f3E000000, 0f3F800000;
                mul.f32 decoded, exponent_f32, mantissa_f32;
                rcp.approx.ftz.f32 reciprocal, decoded;
                selp.f32 $0, 0f00000000, reciprocal, is_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def quantize_e2m1_f32(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Match the Triton fallback's explicit E2M1 midpoint thresholds."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred above, negative, nonzero;
                .reg .u32 code, increment, sign;
                .reg .f32 magnitude;
                abs.f32 magnitude, $1;
                mov.u32 code, 0;
                setp.gt.f32 above, magnitude, 0f3E800000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f3F400000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f3FA00000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f3FE00000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f40200000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f40600000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.gt.f32 above, magnitude, 0f40A00000;
                selp.u32 increment, 1, 0, above;
                add.u32 code, code, increment;
                setp.lt.f32 negative, $1, 0f00000000;
                setp.ne.u32 nonzero, code, 0;
                and.pred negative, negative, nonzero;
                selp.u32 sign, 8, 0, negative;
                or.b32 $0, code, sign;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def quantize_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
) -> Uint32:
    result = quantize_e2m1_f32(v0)
    result = result | (quantize_e2m1_f32(v1) << Uint32(4))
    result = result | (quantize_e2m1_f32(v2) << Uint32(8))
    result = result | (quantize_e2m1_f32(v3) << Uint32(12))
    result = result | (quantize_e2m1_f32(v4) << Uint32(16))
    result = result | (quantize_e2m1_f32(v5) << Uint32(20))
    result = result | (quantize_e2m1_f32(v6) << Uint32(24))
    return result | (quantize_e2m1_f32(v7) << Uint32(28))


@dsl_user_op
def st_global_u64(base_ptr: cutlass.Int64, value: Uint64, *, loc=None, ip=None) -> None:
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


@cute.jit
def quant_16_bf16_to_fp4_u64(values: cute.Tensor, inv_scale: Float32) -> Uint64:
    scaled = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        scaled[i * 2], scaled[i * 2 + 1] = bfloat2_to_float2_scaled(
            values[i], inv_scale
        )

    packed_lo = quantize_e2m1x8_f32(
        scaled[0],
        scaled[1],
        scaled[2],
        scaled[3],
        scaled[4],
        scaled[5],
        scaled[6],
        scaled[7],
    )
    packed_hi = quantize_e2m1x8_f32(
        scaled[8],
        scaled[9],
        scaled[10],
        scaled[11],
        scaled[12],
        scaled[13],
        scaled[14],
        scaled[15],
    )
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
