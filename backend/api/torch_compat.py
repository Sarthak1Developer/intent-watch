from __future__ import annotations

"""Torch compatibility helpers.

Some environments (including some Colab images) may default `torch.load()` to
weights-only loading, which breaks Ultralytics YOLO `.pt` checkpoints.

We patch `torch.load` to default `weights_only=False` when the argument exists
and the caller didn't explicitly set it.

This is a pragmatic demo/deploy safeguard.
"""

from typing import Any, Callable
import inspect


def apply_torch_load_weights_only_default_false() -> None:
    try:
        import torch
    except Exception:
        return

    try:
        sig = inspect.signature(torch.load)
    except Exception:
        return

    if "weights_only" not in sig.parameters:
        return

    original: Callable[..., Any] = torch.load

    # Avoid double-patching.
    if getattr(original, "__intentwatch_patched__", False):
        return

    def patched(*args: Any, **kwargs: Any):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    setattr(patched, "__intentwatch_patched__", True)
    torch.load = patched  # type: ignore[attr-defined]
