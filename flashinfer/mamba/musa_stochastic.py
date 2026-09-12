"""Device primitives matching FlashInfer's software stochastic FP16 path."""

import triton
import triton.language as tl


@triton.jit
def philox4x32(seed, offset, rounds: tl.constexpr):
    # The MUSA Triton 3.6 implementation preserves the high counter word for
    # a 64-bit offset. Keeping this cast explicit prevents implicit narrowing.
    return tl.randint4x(seed.to(tl.uint64), offset.to(tl.uint64), rounds)


@triton.jit
def cvt_rs_f16(value, random_word):
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 16) & 0x8000
    magnitude = (bits & 0x7FFFFFFF) + (random_word & 0x1FFF)
    exponent = (magnitude >> 23) & 0xFF
    mantissa = magnitude & 0x7FFFFF
    half_bits = ((exponent - 112) << 10) | (mantissa >> 13)
    half_bits = tl.where(exponent < 113, 0, half_bits)
    half_bits = tl.where(exponent > 142, 0x7C00, half_bits)
    half_bits = tl.where(
        exponent == 255, tl.where(mantissa != 0, 0x7E00, 0x7C00), half_bits
    )
    return (half_bits | sign).to(tl.uint16).to(tl.float16, bitcast=True)
