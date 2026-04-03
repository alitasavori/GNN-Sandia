"""Shared helpers for daily compare scripts: optional torch.compile on inference."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


def _default_gnn_torch_compile() -> str:
    """Windows + CPU Inductor often needs MSVC ``cl``; default compile off unless user opts in."""
    return "0" if sys.platform == "win32" else "1"


class _CompileOrEager(nn.Module):
    """
    ``torch.compile`` can fail lazily on the first real forward (e.g. Windows without MSVC
    for Inductor). On failure, fall back to eager ``net`` for that call and all later calls.

    On **Windows**, the default Inductor backend invokes ``cl.exe`` for CPU codegen; we use
    ``backend="eager"`` (Dynamo + eager ops, no Inductor C++ build) so compare runs without
    Visual Studio Build Tools.
    """

    def __init__(self, net: nn.Module, label: str):
        super().__init__()
        self.net = net
        self._label = label
        if sys.platform == "win32":
            self._compiled = torch.compile(net, backend="eager")  # type: ignore[assignment]
        else:
            self._compiled = torch.compile(net)  # type: ignore[assignment]
        self._use_compile = True

    def forward(self, *args, **kwargs):
        if not self._use_compile:
            return self.net(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)  # type: ignore[misc]
        except Exception as e:
            self._use_compile = False
            print(
                f"[{self._label}] torch.compile failed on first forward; using eager mode. "
                f"({type(e).__name__}: {e})  "
                "Set GNN_TORCH_COMPILE=0 to skip compile attempts.",
                flush=True,
            )
            return self.net(*args, **kwargs)


def maybe_torch_compile(model: nn.Module, *, label: str = "GNN") -> nn.Module:
    """
    Wrap ``model`` with ``torch.compile`` when PyTorch supports it and
    ``GNN_TORCH_COMPILE`` is not ``0``/``false``.

    Default: **on** on Linux/macOS, **off** on Windows (CPU Inductor there typically needs
    MSVC ``cl``; set ``GNN_TORCH_COMPILE=1`` to try anyway — failures fall back to eager).

    First few forwards after compile can be slow (graph capture); timing prints
    still include that overhead unless you add a separate warmup loop.
    """
    v = os.environ.get("GNN_TORCH_COMPILE", _default_gnn_torch_compile()).strip().lower()
    if v in ("0", "false", "no", "off"):
        print(f"[{label}] torch.compile disabled (GNN_TORCH_COMPILE={v!r})")
        return model
    if not hasattr(torch, "compile"):
        print(f"[{label}] torch.compile not available (PyTorch version)")
        return model

    try:
        out = _CompileOrEager(model, label)
        print(
            f"[{label}] torch.compile enabled ("
            + ("Windows: Dynamo+eager backend, no MSVC; " if sys.platform == "win32" else "Inductor; ")
            + "falls back to plain eager on failure) — set GNN_TORCH_COMPILE=0 to disable",
            flush=True,
        )
        return out
    except Exception as e:
        print(f"[{label}] torch.compile skipped: {e}")
        return model
