from typing import Optional

import numpy as np

from cppquant.dclass import QuantResult

def razor_rollup(ratios: np.ndarray) -> np.ndarray:
    """
    If the majority of values are real numbers, return a list of real numbers.
    If the majority of values are positive infinity, return a list of positive infinity.
    If the majority of values are negative infinity, return a list of negative infinity.
    """
    non_inf_nan_count = np.count_nonzero(~np.isnan(ratios) & ~np.isinf(ratios))
    pos_inf_count = np.count_nonzero(np.isposinf(ratios))
    neg_inf_count = np.count_nonzero(np.isneginf(ratios))

    total_counts = non_inf_nan_count + pos_inf_count + neg_inf_count

    # Determine the majority
    if pos_inf_count > max(neg_inf_count, non_inf_nan_count):
        return np.full(ratios.shape, np.inf)  # Fill with positive infinity
    elif neg_inf_count > max(pos_inf_count, non_inf_nan_count):
        return np.full(ratios.shape, -np.inf)  # Fill with negative infinity
    else:
        # Return non-NaN, non-inf values or empty array if no such values exist
        return ratios[~np.isnan(ratios) & ~np.isinf(ratios)]


def mean_ratio_rollup(results: list[QuantResult], inf_replacement: Optional[float], razor: bool) -> (float, float, int):
    """
    Calculate the mean of the log2 ratios, and the standard deviation.
    """

    log2_ratios = np.array([result.log2_ratio for result in results], dtype=np.float32)

    if razor:
        log2_ratios = razor_rollup(log2_ratios)

    # replace pos inf with inf_replacement
    pos_inf_idx = np.isposinf(log2_ratios)
    log2_ratios[pos_inf_idx] = inf_replacement

    # replace neg inf with -inf_replacement
    neg_inf_idx = np.isneginf(log2_ratios)
    log2_ratios[neg_inf_idx] = -inf_replacement

    mean = np.nanmean(log2_ratios)
    std = np.nanstd(log2_ratios, ddof=1)
    non_nan_count = np.count_nonzero(~np.isnan(log2_ratios))
    return mean, std, non_nan_count


def median_ratio_rollup(results: list[QuantResult], inf_replacement: Optional[float], razor: bool) -> (float, float, int):
    """
    Calculate the mean of the log2 ratios, and the standard deviation.

    inf_replacement: (inf, np.nan, float)
    """

    log2_ratios = np.array([result.log2_ratio for result in results], dtype=np.float32)

    if razor:
        log2_ratios = razor_rollup(log2_ratios)

    # replace pos inf with inf_replacement
    pos_inf_idx = np.isposinf(log2_ratios)
    log2_ratios[pos_inf_idx] = inf_replacement

    # replace neg inf with -inf_replacement
    neg_inf_idx = np.isneginf(log2_ratios)
    log2_ratios[neg_inf_idx] = -inf_replacement

    mean = np.nanmedian(log2_ratios)
    std = np.nanstd(log2_ratios, ddof=1)
    non_nan_count = np.count_nonzero(~np.isnan(log2_ratios))
    return mean, std, non_nan_count


def intensity_sum_ratio_rollup(results: list[QuantResult], inf_replacement: Optional[float]) -> (float, float, int):
    """
    Calculate the mean of the log2 ratios, and the standard deviation.
    """

    # List of light and heavy intensities
    light_intensities = np.array([result.light for result in results], dtype=np.float32)
    heavy_intensities = np.array([result.heavy for result in results], dtype=np.float32)

    # Sum of light and heavy intensities
    light_total = np.sum(light_intensities, axis=1)
    heavy_total = np.sum(heavy_intensities, axis=1)

    # Calculate the ratio of the sums
    ratio = np.divide(light_total, heavy_total)
    log2_ratio = np.log2(ratio)

    if inf_replacement is not None and np.isposinf(log2_ratio):
        log2_ratio = inf_replacement

    elif inf_replacement is not None and np.isneginf(log2_ratio):
        log2_ratio = -inf_replacement

    return log2_ratio, np.nan, len(light_intensities)
