"""Isolated inference forks for ablation baselines."""

__all__ = ["BaselineLLM", "FovealLLM"]


def __getattr__(name):
    if name == "BaselineLLM":
        from .engine import BaselineLLM

        return BaselineLLM
    if name == "FovealLLM":
        from .foveal_engine import FovealLLM

        return FovealLLM
    raise AttributeError(name)
