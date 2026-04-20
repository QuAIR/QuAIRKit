#!/usr/bin/env python3
# Copyright (c) 2026 QuAIR team.
#
# This setup.py exists to build the optional PyTorch C++ extension (quairkit._C).
# Project metadata and runtime dependencies are defined in pyproject.toml.

from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup


def _get_ext_modules():
    # Import torch lazily so that build tools can still inspect the project
    # without importing torch, unless we are actually building extensions.
    from torch.utils.cpp_extension import CppExtension

    here = Path(__file__).parent.resolve()
    # Build all translation units under quairkit/cpp using paths relative to setup.py.
    sources = sorted(
        p.relative_to(here).as_posix()
        for p in (here / "quairkit" / "cpp").rglob("*.cpp")
        if p.is_file()
    )

    debug_mode = os.getenv("DEBUG", "0") == "1"
    extra_compile_args = {
        "cxx": [
            "/O2" if os.name == "nt" and not debug_mode else "",
            "-O3" if os.name != "nt" and not debug_mode else "",
            "/std:c++17" if os.name == "nt" else "-std=c++17",
        ]
    }
    # Filter empty strings introduced above.
    extra_compile_args["cxx"] = [arg for arg in extra_compile_args["cxx"] if arg]

    return [
        CppExtension(
            name="quairkit._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]


if __name__ == "__main__":
    from torch.utils.cpp_extension import BuildExtension

    setup(
        name="quairkit",
        version="0.5.1",
        packages=find_packages(include=["quairkit", "quairkit.*"]),
        ext_modules=_get_ext_modules(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )