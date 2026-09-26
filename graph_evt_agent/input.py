"""Input inspection and normalization for one- and multi-channel series."""

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from .config import InputConfig
from .models import InputProfile


@dataclass(frozen=True)
class TimeSeriesInput:
    """Explicit input container; values use ``[time, channel]`` by default."""

    values: Any
    timestamps: Any | None = None
    channel_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PreparedInput:
    values: np.ndarray
    timestamps: np.ndarray | None
    profile: InputProfile


class InputInspector:
    """Deterministically identify, validate, and prepare the input modality."""

    def __init__(self, config: InputConfig | None = None):
        self.config = config or InputConfig()

    def _unpack(self, source: Any) -> tuple[Any, Any | None, tuple[str, ...] | None]:
        if isinstance(source, TimeSeriesInput):
            return source.values, source.timestamps, source.channel_names
        if isinstance(source, Mapping):
            if not source:
                raise ValueError("channel mapping cannot be empty")
            names = tuple(str(name) for name in source)
            return np.column_stack(list(source.values())), None, names
        # Pandas-like objects are supported without making pandas a dependency.
        if hasattr(source, "to_numpy") and hasattr(source, "columns"):
            names = tuple(str(name) for name in source.columns)
            timestamps = source.index.to_numpy() if hasattr(source, "index") else None
            return source.to_numpy(), timestamps, names
        return source, None, None

    @staticmethod
    def _numeric_timestamps(timestamps: np.ndarray) -> np.ndarray:
        if np.issubdtype(timestamps.dtype, np.datetime64):
            return timestamps.astype("datetime64[ns]").astype(np.int64).astype(float)
        try:
            return timestamps.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("timestamps must be numeric or datetime64") from error

    @staticmethod
    def _interpolate(data: np.ndarray) -> np.ndarray:
        result = data.copy()
        index = np.arange(len(data))
        for node in range(data.shape[1]):
            valid = np.isfinite(data[:, node])
            if not valid.any():
                raise ValueError(f"channel {node} contains no finite observations")
            result[:, node] = np.interp(index, index[valid], data[valid, node])
        return result

    def inspect(self, source: Any) -> PreparedInput:
        raw, raw_timestamps, names = self._unpack(source)
        data = np.asarray(raw, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim != 2:
            raise ValueError("input must be a 1-D series or a 2-D time/channel array")
        if self.config.time_axis == 1:
            data = data.T
        if not len(data) or not data.shape[1]:
            raise ValueError("input cannot be empty")

        channels = data.shape[1]
        if names is None:
            names = tuple(f"channel_{node}" for node in range(channels))
        if len(names) != channels or len(set(names)) != channels:
            raise ValueError("channel_names must be unique and match the channel count")
        missing = (~np.isfinite(data)).sum(axis=0)
        actions: list[str] = []

        timestamps = None if raw_timestamps is None else np.asarray(raw_timestamps)
        if timestamps is not None and (timestamps.ndim != 1 or len(timestamps) != len(data)):
            raise ValueError("timestamps must be one-dimensional and match the time axis")
        if missing.any():
            if self.config.missing == "error":
                raise ValueError(f"input contains {int(missing.sum())} missing/non-finite values")
            if self.config.missing == "drop_rows":
                keep = np.isfinite(data).all(axis=1)
                data = data[keep]
                if timestamps is not None:
                    timestamps = timestamps[keep]
                actions.append("dropped_rows_with_missing_values")
            else:
                data = self._interpolate(data)
                actions.append("interpolated_missing_values")

        regular: bool | None = None
        interval: float | None = None
        if timestamps is not None:
            numeric_time = self._numeric_timestamps(timestamps)
            differences = np.diff(numeric_time)
            if not np.isfinite(numeric_time).all() or np.any(differences <= 0):
                raise ValueError("timestamps must be finite, unique, and strictly increasing")
            interval = float(np.median(differences)) if len(differences) else None
            regular = bool(len(differences) < 2 or np.allclose(differences, interval, rtol=1e-5))
            if self.config.require_regular_time and not regular:
                raise ValueError("irregular timestamps require resampling before EVT")

        profile = InputProfile(
            kind="univariate" if channels == 1 else "multivariate",
            n_time=len(data), n_channels=channels, channel_names=names,
            missing_per_channel=missing, has_timestamps=timestamps is not None,
            regular_time=regular, sampling_interval=interval, actions=tuple(actions),
        )
        return PreparedInput(data, timestamps, profile)
