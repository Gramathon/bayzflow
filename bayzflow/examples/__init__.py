"""Runnable examples with optional dependencies imported on demand."""

__all__ = ["forecast_paths_primary_only"]


def __getattr__(name):
    if name == "forecast_paths_primary_only":
        from .fx.fx_utils.forecast import forecast_paths_primary_only

        return forecast_paths_primary_only
    raise AttributeError(name)
