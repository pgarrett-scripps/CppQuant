from typing import List, Callable

import numpy as np

from cppquant.dclass import QuantResult


def simple_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool,
                  imput_func: Callable) -> None:
    if separate_single_double is True:
        mean_impute([quant_result for quant_result in quant_results if quant_result.is_single],
                    False, separate_light_heavy)
        mean_impute([quant_result for quant_result in quant_results if quant_result.is_double],
                    False, separate_light_heavy)

    lights = np.array([quant_result.light for quant_result in quant_results])
    light_mean = imput_func(lights)
    heavys = np.array([quant_result.heavy for quant_result in quant_results])
    heavy_mean = imput_func(heavys)
    if separate_light_heavy is True:
        for quant_result in quant_results:
            if quant_result.light == 0.0:
                quant_result.imputed_light = light_mean
            if quant_result.heavy == 0.0:
                quant_result.imputed_heavy = heavy_mean

    else:
        mean = imput_func(np.hstack((lights, heavys)))
        for quant_result in quant_results:
            if quant_result.light == 0.0:
                quant_result.imputed_light = mean
            if quant_result.heavy == 0.0:
                quant_result.imputed_heavy = mean


def mean_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool) -> None:
    simple_impute(quant_results, separate_single_double, separate_light_heavy, np.nanmean)


def median_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool) -> None:
    simple_impute(quant_results, separate_single_double, separate_light_heavy, np.nanmean)


def min_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool) -> None:
    simple_impute(quant_results, separate_single_double, separate_light_heavy, np.nanmin)


def max_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool) -> None:
    simple_impute(quant_results, separate_single_double, separate_light_heavy, np.nanmin)


def constant_impute(quant_results: List[QuantResult], separate_single_double: bool, separate_light_heavy: bool,
                    constant: float) -> None:
    simple_impute(quant_results, separate_single_double, separate_light_heavy, lambda x: constant)
