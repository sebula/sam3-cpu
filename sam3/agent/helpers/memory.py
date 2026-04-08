# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from functools import wraps

__all__ = ["retry_if_oom"]


def retry_if_oom(func):
    """Thin wrapper; upstream retried on accelerator OOM — not used on CPU-only builds."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped
