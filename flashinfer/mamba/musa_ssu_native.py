"""Lazy loader for the S5000 native Simple-STP MUSA kernel.

The extension is deliberately opt-in: importing FlashInfer must not trigger a
MUSA compiler invocation. Set ``FLASHINFER_MUSA_SIMPLE_STP_NATIVE=1`` at the
serving process boundary to select this provider; callers can fall back to the
MUSA Triton provider when the extension is unavailable.
"""
from __future__ import annotations

import os
import importlib.machinery
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_EXT: Any | None = None


def _load_extension() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT
    if os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1":
        raise RuntimeError("native Simple STP is opt-in")
    import torch_musa  # noqa: F401  # activates the MUSA PyTorch extension shim
    source = Path(__file__).resolve().parents[2] / "csrc" / "mamba" / "musa_simple_stp.mu"
    if not source.is_file():
        # Non-editable wheels install repository sources under flashinfer.data.
        from importlib.resources import files

        source = Path(files("flashinfer.data").joinpath("csrc/mamba/musa_simple_stp.mu"))
    if not source.is_file():
        raise FileNotFoundError(source)
    # vLLM-MUSA builds `.mu` sources through PyTorch's CUDAExtension shim;
    # torch_musa patches that class in the runtime. Keep MUSAExtension as a
    # fallback for images exposing the newer dedicated helper.
    if importlib.util.find_spec("torch.utils.cpp_extension") is not None:
        extension_import = "from torch.utils.cpp_extension import CUDAExtension as Extension, BuildExtension"
    elif importlib.util.find_spec("torch_musa.utils.musa_extension") is not None:
        extension_import = "from torch_musa.utils.musa_extension import MUSAExtension as Extension, BuildExtension"
    else:
        raise RuntimeError("no PyTorch MUSA extension builder is available")
    build_root = Path(tempfile.gettempdir()) / "flashinfer_musa_simple_stp"
    build_root.mkdir(parents=True, exist_ok=True)
    setup_py = build_root / "setup.py"
    setup_py.write_text(
        "from setuptools import setup\n"
        "import torch_musa\n"
        f"{extension_import}\n"
        f"setup(name='flashinfer_musa_simple_stp', ext_modules=[Extension('flashinfer_musa_simple_stp', [{str(source)!r}], extra_compile_args=['-O3', '-std=c++17'])], cmdclass={{'build_ext': BuildExtension}})\n"
    )
    subprocess.run(
        [sys.executable, str(setup_py), "build_ext", "--inplace"],
        cwd=build_root,
        check=True,
        env=os.environ.copy(),
    )
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    candidates = [p for suffix in suffixes for p in build_root.glob(f"flashinfer_musa_simple_stp*{suffix}")]
    if not candidates:
        raise RuntimeError(f"MUSAExtension build produced no module in {build_root}")
    spec = importlib.util.spec_from_file_location("flashinfer_musa_simple_stp", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native module {candidates[0]}")
    _EXT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_EXT)
    return _EXT


def musa_ssu_one_token_native(*args: Any, **kwargs: Any) -> Any:
    """Invoke the native S5000 Simple-STP extension lazily."""
    return _load_extension().musa_ssu_simple(*args, **kwargs)


__all__ = ["musa_ssu_one_token_native"]
