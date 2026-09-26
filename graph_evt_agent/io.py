"""File readers for the documented CSV and EDF/BDF interchange formats."""

import csv
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .input import TimeSeriesInput


_TIME_NAMES = {"time", "timestamp", "datetime", "date_time"}


def _parse_timestamps(values: list[str]) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except ValueError:
        try:
            timestamps = np.asarray(values, dtype="datetime64[ns]")
        except ValueError as error:
            raise ValueError("CSV timestamps must be numeric or ISO-8601 datetime values") from error
        if np.isnat(timestamps).any():
            raise ValueError("CSV timestamp column contains an invalid datetime")
        return timestamps


def load_csv(path: str | Path, *, time_column: str | None = None,
             delimiter: str = ",", encoding: str = "utf-8-sig") -> TimeSeriesInput:
    """Read a wide CSV: one row per time point and one column per channel."""
    file_path = Path(path)
    with file_path.open("r", newline="", encoding=encoding) as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header row")
        headers = tuple(str(name).strip() for name in reader.fieldnames)
        if not all(headers) or len(set(headers)) != len(headers):
            raise ValueError("CSV header names must be non-empty and unique")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain at least one data row")

    if time_column is None:
        matches = [name for name in headers if name.lower() in _TIME_NAMES]
        time_column = matches[0] if len(matches) == 1 else None
    if time_column is not None and time_column not in headers:
        raise ValueError(f"time column {time_column!r} is not present in CSV")
    channels = tuple(name for name in headers if name != time_column)
    if not channels:
        raise ValueError("CSV must contain at least one signal channel")

    values = np.empty((len(rows), len(channels)), dtype=float)
    for row_index, row in enumerate(rows):
        for channel_index, channel in enumerate(channels):
            text = (row.get(channel) or "").strip()
            values[row_index, channel_index] = float(text) if text else np.nan
    timestamps = None
    if time_column is not None:
        raw_time = [(row.get(time_column) or "").strip() for row in rows]
        if any(not value for value in raw_time):
            raise ValueError("CSV timestamp cells cannot be empty")
        timestamps = _parse_timestamps(raw_time)
    return TimeSeriesInput(values, timestamps, channels)


def load_edf(path: str | Path) -> TimeSeriesInput:
    """Read equally sampled EDF/EDF+/BDF channels through optional pyEDFlib."""
    if importlib.util.find_spec("pyedflib") is None:
        raise ImportError("EDF support requires: python -m pip install 'graph-evt-agent[edf]'")
    pyedflib: Any = importlib.import_module("pyedflib")
    reader = pyedflib.EdfReader(str(path))
    try:
        channel_count = int(reader.signals_in_file)
        if channel_count < 1:
            raise ValueError("EDF file contains no signals")
        names = tuple(str(name).strip() or f"channel_{index}"
                      for index, name in enumerate(reader.getSignalLabels()))
        frequencies = np.asarray(reader.getSampleFrequencies(), dtype=float)
        sample_counts = np.asarray(reader.getNSamples(), dtype=int)
        if not np.allclose(frequencies, frequencies[0]) or np.any(sample_counts != sample_counts[0]):
            raise ValueError("all selected EDF channels must have equal sample rates and lengths")
        values = np.column_stack([reader.readSignal(index) for index in range(channel_count)])
    finally:
        reader.close()
    timestamps = np.arange(len(values), dtype=float) / frequencies[0]
    return TimeSeriesInput(values, timestamps, names)


def load_timeseries(path: str | Path, **kwargs: Any) -> TimeSeriesInput:
    """Dispatch to CSV or EDF/BDF reader from the filename extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path, **kwargs)
    if suffix in {".edf", ".bdf"}:
        if kwargs:
            raise TypeError("EDF/BDF loader does not accept CSV options")
        return load_edf(path)
    raise ValueError("supported input file extensions are .csv, .edf, and .bdf")
