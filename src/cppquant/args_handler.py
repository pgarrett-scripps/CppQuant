import argparse
from typing import List

from cppquant.dclass import Group, Pair


def parse_census_files(arg) -> Group:
    # Should be in the following format: filename:groupname

    try:
        # Fix for files that start with C:\\
        filename = ':'.join(arg.split(':')[:-1])
        groupname = arg.split(':')[-1]
        return Group(filename, groupname)

    except ValueError:
        raise argparse.ArgumentTypeError("Census files must be in format 'filename:groupname;filename:groupname;...'")


def parse_groupby_columns(arg) -> List[str]:
    """
    Parse the groupby columns argument into a list of strings.
    """
    try:
        return arg.split(';')
    except ValueError:
        raise argparse.ArgumentTypeError("Groupby columns must be in format 'column1;column2;...'")


def parse_pairs(arg) -> List[Pair]:
    """
    Parse the pairs argument into a list of tuples.
    """
    try:
        return [Pair(*map(str, pair.split(','))) for pair in arg.split(';')]
    except ValueError:
        raise argparse.ArgumentTypeError("Pairs must be in format '1,2;3,4;...'")


def parse_inf_replacement(arg) -> float:
    """
    Parse the inf_replacement argument into a float.
    """
    try:
        if arg == 'inf':
            return float('inf')
        elif arg == 'nan':
            return float('nan')
        else:
            return float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("inf_replacement must be a float, 'inf', or 'nan'.")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Census files to parquet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--census_files', nargs='+', type=parse_census_files, required=True,
                        help='The input CPP census files, can be full paths for relative paths to --input_folder argument.'
                             ' Each file should be in the format filename:groupname.',
                        )
    parser.add_argument('--input_folder',  default='', required=False,
                        help='Input folder containing census files. Used to resolve relative paths.',
                        )
    parser.add_argument('--fasta_file', required=True,
                        help='Fasta file used in the original search. Used to get protein sequence from their loci.')
    parser.add_argument('--pairs', type=parse_pairs, required=True,
                        help='Pairs of files to compare. Should be in the format "groupN:groupM;...;groupX:groupY".'
                             'The group names should match the names in the groupname provided in the --census_files'
                             'argument.')
    parser.add_argument('--site_regex', default='K', help='Regex to match site names. '
                                                          'Default is "K" for lysine.')

    filter_options = parser.add_argument_group('Filter Options')
    filter_options.add_argument('--remove_double', action='store_true',
                                help='Remove peptides with regex sites.')
    filter_options.add_argument('--remove_single', action='store_true',
                                help='Remove peptides with regex site')
    filter_options.add_argument('--keep_missing_values', action='store_true',
                                help='Keep missing datapoints (Its recommended to keep this off). '
                                     'A datapoint is considered missing if it is missing both'
                                     'the light and heavy measurements. The light and heavy values are calculated '
                                     'according to the light/medium/heavy peak areas, for a given single/double site '
                                     'peptide.')
    filter_options.add_argument('--keep_duplicates', action='store_true',
                                help='Keep duplicate datapoints. A datapoint is considered a duplicate if it has the '
                                'same filename and scan number. The data is first sorted based on intensity, so that only '
                                'the highest intensity datapoint is kept.')


    output_options = parser.add_argument_group('Output Options')
    output_options.add_argument('--output_folder', default='.',
                                help='Output folder. Default is the current directory.')
    output_options.add_argument('--args_file_name', type=str, default='args',
                                help='Output file for arguments used in the analysis.')
    output_options.add_argument('--results_file_name', type=str, default='cpp_results',
                                help='Output file for cpp results.')
    output_options.add_argument('--psm_file_name', type=str, default='psm_cpp_groups',
                                help='Output file for psm cpp results. Group files will have this name while group '
                                     'compare files will have this name + _compare')
    output_options.add_argument('--peptide_file_name', type=str, default='peptide_cpp_groups',
                                help='Output file for peptide cpp results. Group files will have this name while group '
                                     'compare files will have this name + _compare')
    output_options.add_argument('--protein_file_name', type=str, default='protein_cpp_groups',
                                help='Output file for protein cpp results. Group files will have this name while group '
                                     'compare files will have this name + _compare')

    grouping_options = parser.add_argument_group('Grouping Options')
    grouping_options.add_argument('--psm_groupby',
                                  default='peptide_site_str;charge;group',
                                  type=parse_groupby_columns,
                                  help='The grouping criteria for PSMs, for group comparison the group column is '
                                       'required')
    grouping_options.add_argument('--peptide_groupby',
                                  default='peptide_site_str;group',
                                  type=parse_groupby_columns,
                                  help='The grouping criteria for Peptides, for group comparison the group column is '
                                       'required')
    grouping_options.add_argument('--protein_groupby',
                                  default='protein_site_str;charge;group',
                                  type=parse_groupby_columns,
                                  help='The grouping criteria for Proteins, for group comparison the group column is '
                                       'required')

    rollup_options = parser.add_argument_group('Ratio Rollup')
    rollup_options.add_argument('--rollup_method', choices=['mean', 'median', 'sum'], default='median',
                                help='Method to rollup the log2 ratios. The mean / median approaches are very '
                                     'similar, they first group all of the datapoints based on the grouping criteria. '
                                     'Then the log2_ratios are calculated for each datapoint. If razor is enabled, '
                                     'these log2_ratios are filtered according to the razor method (winner takes all)'
                                     '. Next the inf_replacement value is applied to all inf values. The mean method '
                                     'then takes the mean of the log2_ratios, while the median method takes the '
                                     'median of the log2_ratios. The sum method simply takes the sum of the light and '
                                     'heavy intensities and then calculates the log2_ratio (razor method is not '
                                     'compatible with sum since it only produced a single value). Like the mean/median'
                                     'approaches the inf_replacement is applied at the end. Default is median.')
    rollup_options.add_argument('--inf_replacement', type=parse_inf_replacement,
                                default='100',
                                help='Value to replace infinities log2_ratios with (Options: float, inf, nan). '
                                     'If float, replace with the given float value. '
                                     'If inf, this will effectively keep infinity values in place. '
                                     'If nan, replace with np.nan, which will be ignored in the rollup calculation.')
    rollup_options.add_argument('--razor', action='store_true',
                                help='Use razor method to rollup ratios. Counts the number of real, pos inf and neg '
                                     'inf ratios, and keeps only the majority type. If there is a tie, '
                                     'the real ratio wins.')

    return parser.parse_args()
