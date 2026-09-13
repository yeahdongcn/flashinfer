"""Lazy loader for the S5000 native Simple-STP MUSA kernel.

The extension is deliberately opt-in: importing FlashInfer must not trigger a
MUSA compiler invocation. Set ``FLASHINFER_MUSA_SIMPLE_STP_NATIVE=1`` at the
serving process boundary to select this provider; callers can fall back to the
MUSA Triton provider when the extension is unavailable.
"""
from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import Any

_EXT: Any | None = None


def _load_extension() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT
    if os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1":
        raise RuntimeError("native Simple STP is opt-in")
    source = Path(__file__).resolve().parents[2] / "csrc" / "mamba" / "musa_simple_stp.mu"
    if not source.is_file():
        raise FileNotFoundError(source)
    loader = None
    with suppress(ImportError):
        from torch_musa.utils.cpp_extension import load as loader  # type: ignore[import-not-found]
    if loader is None:
        raise RuntimeError("torch_musa.utils.cpp_extension.load is unavailable")
    _EXT = loader(
        name="flashinfer_musa_simple_stp",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("FLASHINFER_MUSA_BUILD_VERBOSE") == "1",
    )
    return _EXT


def musa_ssu_one_token_native(*args: Any, **kwargs: Any) -> Any:
    """Invoke the native S5000 Simple-STP extension lazily."""
    return _load_extension().musa_ssu_simple(*args, **kwargs)


__all__ = ["musa_ssu_one_token_native"]
