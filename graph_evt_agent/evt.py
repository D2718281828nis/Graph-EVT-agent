"""Leakage-safe multichannel POT/GPD detection."""

import numpy as np
from scipy.stats import genpareto

from .config import EVTConfig
from .models import EVTDetection


def _as_2d(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("values must be a finite [time, channel] array")
    return data


def detect(values: np.ndarray, config: EVTConfig) -> EVTDetection:
    """Fit the baseline tail and find the first persistent post-baseline alarm."""
    data = _as_2d(values)
    if config.baseline_size >= len(data):
        raise ValueError("baseline_size must leave post-baseline observations")
    base = data[: config.baseline_size]
    location = np.median(base, axis=0)
    mad = np.median(np.abs(base - location), axis=0)
    scale = np.maximum(1.4826 * mad, config.epsilon)
    z = np.abs(data - location) / scale
    k = min(config.top_k, data.shape[1])
    indicator = np.mean(np.partition(z, -k, axis=1)[:, -k:], axis=1)

    baseline_indicator = indicator[: config.baseline_size]
    u = float(np.quantile(baseline_indicator, config.tail_quantile))
    excess = baseline_indicator[baseline_indicator > u] - u
    if len(excess) >= 5 and np.any(excess > 0):
        shape, _, gpd_scale = genpareto.fit(excess, floc=0)
        conditional = (config.alarm_probability - config.tail_quantile) / (
            1 - config.tail_quantile
        )
        threshold = u + float(genpareto.ppf(conditional, shape, loc=0, scale=gpd_scale))
    else:
        threshold = float(np.quantile(baseline_indicator, config.alarm_probability))

    hits = indicator > threshold
    found = None
    for start in range(config.baseline_size, len(data) - config.persistence + 1):
        if bool(np.all(hits[start : start + config.persistence])):
            found = start
            break
    return EVTDetection(found is not None, found, threshold, indicator, location, scale)

