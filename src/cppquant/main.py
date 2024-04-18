"""
Given a list of census files, and bio/tech replicates, this script will output a parquet file compatible with sagequant

1) Get a list of files, bio replicate labels, and tech replicate labels
2) Read the Census files and format the dataframe
"""
from __future__ import annotations

import os
import pprint
from functools import partial
from itertools import groupby
from typing import List, Dict, Generator, Any, Iterable, IO, Optional, Callable

import datetime
import numpy as np
import pandas as pd
import peptacular as pt
from statsmodels.stats.multitest import multipletests

from cppquant.census_util import build_census_results
from cppquant.args_handler import parse_args
from cppquant.fasta_util import map_ip2_fasta_id_to_sequence, read_fasta
from cppquant.dclass import CensusResult, CPPResult, QuantResult, QuantResult, Pair, RatioResult, CompareRatio
from cppquant.ratio_rollup import mean_ratio_rollup, median_ratio_rollup, intensity_sum_ratio_rollup


def build_cpp_results(census_results: Iterable[CensusResult]) -> Generator[CPPResult, None, None]:
    for census_result in census_results:
        for sline in census_result.s_lines:
            yield CPPResult(census_result.p_lines, sline)

def build_quant_ratios(cpp_results: Iterable[CPPResult], fasta_map: Dict[str, str], regex_str: str, group: Any) -> \
        Generator[QuantResult, None, None]:
    for cpp_result in cpp_results:
        peptide_indices = []
        for loci in cpp_result.loci:
            protein_sequence = fasta_map.get(loci)

            if protein_sequence is None:
                raise ValueError(f'Protein {loci} not found in fasta')

            unmodified_sequence = cpp_result.unmodified_sequence
            indices = pt.find_subsequence_indices(protein_sequence, unmodified_sequence, ignore_mods=True)
            peptide_indices.append(indices)

            if cpp_result.is_single:
                yield QuantResult(cpp_result, 0, group, peptide_indices, regex_str)
            elif cpp_result.is_double:
                yield QuantResult(cpp_result, 0, group, peptide_indices, regex_str)
                yield QuantResult(cpp_result, 1, group, peptide_indices, regex_str)


def to_results_file(quant_results: List[QuantResult], output_file: str):
    cols = ['ip2_sequence', 'proforma_sequence', 'unmodified_sequence', 'charge', 'filename', 'scannr', 'regex_site',
            'group', 'loci_str', 'is_decoy', 'is_single', 'is_double', 'light', 'heavy', 'ratio', 'log2_ratio',
            'log10_ratio',
            'peptide_site_str', 'protein_site_str']

    datas = []

    for qr in quant_results:

        data = []
        for col in cols:
            data.append(getattr(qr, col))
        datas.append(data)

    df = pd.DataFrame(datas, columns=cols)
    df.to_csv(output_file, index=False, float_format='%.5f')


def to_ratio_results(quant_results: List[QuantResult], ratio_rollup: Callable, groupby_cols: List[str]) -> List[
    RatioResult]:
    group_by_func = lambda x: tuple([getattr(x, col) for col in groupby_cols])

    # sort
    quant_results.sort(key=group_by_func)

    results = []
    for key, group in groupby(quant_results, group_by_func):
        results.append(RatioResult(list(group), ratio_rollup, groupby_cols, key))

    return results


def to_group_file(results: List[RatioResult], output_file: str):
    if len(results) == 0:
        return

    data = [res.to_dict() for res in results]
    pd.DataFrame(data).to_csv(output_file, index=False, float_format='%.5f')


def _assign_qvalues(pvalues: np.ndarray) -> np.ndarray:
    # TODO: Check if this is the correct way to handle QValues
    """
    . code-block:: python

        >>> pvalues = np.array([0.1, 0.2, 0.3, np.nan, 0.4, 0.5])
        >>> _assign_qvalues(pvalues)
        array([0.5, 0.5, 0.5, nan, 0.5, 0.5])

        >>> pvalues = np.array([0.01, 0.5, 0.6, np.nan, 0.3, 0.1])
        >>> _assign_qvalues(pvalues)
        array([0.05, 0.6 , 0.6 ,  nan, 0.5 , 0.25])

    """
    # Filter out NaN values and keep track of their original indices
    non_nan_indices = np.array([i for i, pv in enumerate(pvalues) if not np.isnan(pv)])
    non_nan_pvalues = np.array([pv for pv in pvalues if not np.isnan(pv)])

    # Perform multipletests on non-NaN p-values
    qvalues = np.full(len(pvalues), np.nan)  # Initialize full array with NaNs

    if len(non_nan_pvalues) > 0:
        qvalues_non_nan = multipletests(non_nan_pvalues, method='fdr_bh')[1]
        qvalues[non_nan_indices] = qvalues_non_nan  # Update only the non-NaN positions

    return qvalues


def to_compare_file(results: List[RatioResult], output_file: str, pairs: List[Pair]):
    if len(results) == 0:
        return

    # separate groups
    group_map = {}
    non_group_key_labels = None
    for res in results:
        non_group_key = res.non_group_key

        if non_group_key_labels is None:
            non_group_key_labels = res.non_group_key_labels

        group = res.group

        if non_group_key not in group_map:
            group_map[non_group_key] = {}

        if group in group_map[non_group_key]:
            raise ValueError(f'Group {group} already exists in {non_group_key}')

        group_map[non_group_key][group] = res

    compare_results = []
    for group_key, groups in group_map.items():

        # get non_group_key_labels
        non_group_key = None
        non_group_key_labels = None
        for key in groups:
            non_group_key = groups[key].non_group_key
            non_group_key_labels = groups[key].non_group_key_labels
            break

        for pair in pairs:
            group1 = groups.get(pair.group1, None)
            group2 = groups.get(pair.group2, None)

            compare = CompareRatio(pair, group1, group2, non_group_key_labels, non_group_key)
            compare_results.append(compare)

    pvalues = np.array([res.pvalue for res in compare_results])
    qvalues = _assign_qvalues(pvalues)
    for compare, qvalue in zip(compare_results, qvalues):
        compare.qvalue = qvalue

    data = [res.to_dict() for res in compare_results]
    pd.DataFrame(data).to_csv(output_file, index=False, float_format='%.5f')


def run():
    args = parse_args()

    print("=" * 30)
    print("Cpp Ratio Calculator")
    print("=" * 30)
    print()

    print('Arguments:')
    pprint.pprint(vars(args))
    print()

    # Create output folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

    # save the arguments to a file
    with open(os.path.join(args.output_folder, args.args_file_name +'.txt'), 'w') as f:
        f.write(f'\n\nDate: {datetime.datetime.now()}')

        f.write(pprint.pformat(vars(args)))

    print('Generating Fasta Map...')
    with open(args.fasta_file, 'r') as f:
        fasta_map = map_ip2_fasta_id_to_sequence(read_fasta(f))

    quant_results = []
    for census_file in args.census_files:
        file_path = census_file.file
        group = census_file.group

        # TODO: Add this filename somewhere since the filenames within each census file can have H/L/M prefixes
        file_base_name = os.path.basename(file_path).split('.')[0]

        if args.input_folder:
            file_path = os.path.join(args.input_folder, file_path)

        with open(file_path, 'r') as f:
            census_results = list(build_census_results(f))
            quant_results.extend(
                list(build_quant_ratios(build_cpp_results(census_results), fasta_map, args.site_regex, group)))

    num_quant_results = len(quant_results)
    quant_results = [qr for qr in quant_results if qr.is_valid]
    print(f'Filtered {num_quant_results - len(quant_results)} invalid values')

    num_quant_results = len(quant_results)
    if not args.keep_missing_values:
        quant_results = [qr for qr in quant_results if not qr.is_missing]
    print(f'Filtered {num_quant_results - len(quant_results)} missing values')

    num_quant_results = len(quant_results)
    if not args.keep_duplicates:
        # sort based on total intensity
        quant_results.sort(key=lambda x: x.total_intensity, reverse=True)
        duplicate_keys = set()
        for i, qr in enumerate(quant_results):
            key = qr.duplicate_key
            if key in duplicate_keys:
                qr.is_duplicate = True
            else:
                duplicate_keys.add(key)
        quant_results = [qr for qr in quant_results if not qr.is_duplicate]
    print(f'Filtered {num_quant_results - len(quant_results)} duplicate values')

    num_quant_results = len(quant_results)
    if args.remove_double:
        quant_results = [qr for qr in quant_results if not qr.is_double]
    print(f'Filtered {num_quant_results - len(quant_results)} double sites')

    num_quant_results = len(quant_results)
    if args.remove_single:
        quant_results = [qr for qr in quant_results if not qr.is_single]
    print(f'Filtered {num_quant_results - len(quant_results)} single sites')

    # remaining quant results
    print(f'Number of quant results: {len(quant_results)}')

    if args.rollup_method == 'mean':
        ratio_rollup = partial(mean_ratio_rollup, inf_replacement=args.inf_replacement, razor=args.razor)
    elif args.rollup_method == 'median':
        ratio_rollup = partial(median_ratio_rollup, inf_replacement=args.inf_replacement, razor=args.razor)
    elif args.rollup_method == 'sum':
        ratio_rollup = partial(intensity_sum_ratio_rollup, inf_replacement=args.inf_replacement)
    else:
        raise ValueError(f'Invalid rollup method {args.rollup_method}')

    quant_results.sort(key=lambda x: (x.ip2_sequence, group))

    to_results_file(quant_results,  str(os.path.join(args.output_folder, args.results_file_name + '.csv')))

    psm_ratios = to_ratio_results(quant_results, ratio_rollup, args.psm_groupby)
    peptide_ratios = to_ratio_results(quant_results, ratio_rollup, args.peptide_groupby)
    protein_ratios = to_ratio_results(quant_results, ratio_rollup, args.protein_groupby)

    to_group_file(psm_ratios, str(os.path.join(args.output_folder, args.psm_file_name + '.csv')))
    to_group_file(peptide_ratios, str(os.path.join(args.output_folder, args.peptide_file_name + '.csv')))
    to_group_file(protein_ratios, str(os.path.join(args.output_folder, args.protein_file_name + '.csv')))

    to_compare_file(psm_ratios, str(os.path.join(args.output_folder, args.psm_file_name + '_compare_.csv')), args.pairs)
    to_compare_file(peptide_ratios, str(os.path.join(args.output_folder, args.peptide_file_name + '_compare.csv')),
                    args.pairs)
    to_compare_file(protein_ratios, str(os.path.join(args.output_folder, args.protein_file_name + '_compare.csv')),
                    args.pairs)


if __name__ == '__main__':
    run()

