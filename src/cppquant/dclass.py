import math
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, Any, Callable, Tuple

import numpy as np
import peptacular as pt
import regex
from scipy.stats import ttest_ind_from_stats

from cppquant.uncertainty_estimation import sum_uncertainty
from cppquant.util import process_value


@dataclass
class Group:
    file: str
    group: Any


@dataclass
class Pair:
    group1: Any
    group2: Any


@dataclass
class Line(ABC):
    """
    This class stores only a single reference to the column values and the column index. This is preferable to storing
    the entire row as a dictionary, as it is more memory efficient and faster.
    """
    _column_index: Dict[str, int]
    _values: List[Any]

    def __getitem__(self, item: str) -> Any:
        return process_value(self._values[self._column_index[item]])

    def __setitem__(self, key: str, value: Any):
        self._values[self._column_index[key]] = value

    @property
    def columns(self) -> List[str]:
        return list(self._column_index.keys())

    @property
    def values(self) -> List[Any]:
        return [self[column] for column in self.columns]


@dataclass
class SequenceLine(Line):

    @property
    def light_peak_area(self) -> float:
        if self['PEAK_AREA_L'] is None or math.isnan(self['PEAK_AREA_L']):
            return 0.0
        return float(self['PEAK_AREA_L'])

    @light_peak_area.setter
    def light_peak_area(self, value: float):
        self['PEAK_AREA_L'] = value

    @property
    def medium_peak_area(self) -> float:
        if self['PEAK_AREA_M'] is None or math.isnan(self['PEAK_AREA_M']):
            return 0.0
        return float(self['PEAK_AREA_M'])

    @medium_peak_area.setter
    def medium_peak_area(self, value: float):
        self['PEAK_AREA_M'] = value

    @property
    def heavy_peak_area(self) -> float:
        if self['PEAK_AREA_H'] is None or math.isnan(self['PEAK_AREA_H']):
            return 0.0
        return float(self['PEAK_AREA_H'])

    @heavy_peak_area.setter
    def heavy_peak_area(self, value: float):
        self['PEAK_AREA_H'] = value

    @property
    def peptide_sequence(self) -> str | None:
        if self['SEQUENCE'] is None:
            return None
        return str(self['SEQUENCE'])

    @peptide_sequence.setter
    def peptide_sequence(self, value: str):
        self['SEQUENCE'] = value

    @property
    def peptide_charge(self) -> int | None:
        if self['CS'] is None:
            return None
        return int(self['CS'])

    @peptide_charge.setter
    def peptide_charge(self, value: int):
        self['CS'] = value

    @property
    def scannr(self) -> int | None:
        if self['SCAN'] is None:
            return None
        return int(self['SCAN'])

    @scannr.setter
    def scannr(self, value: int):
        self['SCAN'] = value

    @property
    def filename(self) -> str | None:
        if self['FILE_NAME'] is None:
            return None
        return str(self['FILE_NAME'])

    @filename.setter
    def filename(self, value: str):
        self['FILE_NAME'] = value

    @property
    def xcorr(self) -> float | None:
        if self['XCorr'] is None:
            return None
        return float(self['XCorr'])

    @xcorr.setter
    def xcorr(self, value: float):
        self['XCorr'] = value


@dataclass
class DLine(SequenceLine):
    pass


@dataclass
class SLine(SequenceLine):
    d_lines: List[DLine] = field(default_factory=list)


@dataclass
class PLine(Line):
    @property
    def locus(self) -> str | None:
        return str(self['LOCUS'])

    @locus.setter
    def locus(self, value: str):
        self['LOCUS'] = value


@dataclass
class CensusResult:
    p_lines: List[PLine] = field(default_factory=list)
    s_lines: List[SLine] = field(default_factory=list)


@dataclass
class CPPResult:
    p_lines: List[PLine]
    s_line: SLine

    @property
    def ip2_sequence(self) -> str:
        return self.s_line.peptide_sequence

    @cached_property
    def proforma_sequence(self) -> str:
        return pt.convert_ip2_sequence(self.ip2_sequence)

    @cached_property
    def unmodified_sequence(self) -> str:
        return pt.strip_mods(self.proforma_sequence)

    @property
    def is_double(self):
        return len(self.s_line.d_lines) == 2

    @property
    def is_single(self):
        return len(self.s_line.d_lines) == 0

    @property
    def light(self) -> List[float]:

        if self.is_single:
            return [self.s_line.light_peak_area]

        elif self.is_double:
            site1 = self.s_line.light_peak_area + self.s_line.d_lines[0].medium_peak_area
            site2 = self.s_line.light_peak_area + self.s_line.d_lines[1].medium_peak_area
            return [site1, site2]

        else:
            raise ValueError('Invalid census site')

    @property
    def heavy(self) -> List[float]:

        if self.is_single:
            return [self.s_line.medium_peak_area]

        elif self.is_double:
            site1 = self.s_line.heavy_peak_area + self.s_line.d_lines[1].medium_peak_area
            site2 = self.s_line.heavy_peak_area + self.s_line.d_lines[0].medium_peak_area
            return [site1, site2]

        else:
            raise ValueError('Invalid census site')

    @property
    def loci(self) -> List[str]:

        loci = set()
        for pline in self.p_lines:
            for locus in pline.locus.split(';'):
                loci.add(locus)
        loci = sorted(list(loci))

        return loci

    @property
    def scannr(self) -> int:
        return self.s_line.scannr

    @property
    def filename(self) -> str:
        return self.s_line.filename

    @filename.setter
    def filename(self, value: str):
        self.s_line.filename = value

    @property
    def charge(self) -> int:
        return self.s_line.peptide_charge


@dataclass
class QuantResult:
    cpp_result: CPPResult
    cpp_result_index: int
    group: Any
    peptide_indices: List[List[int]]
    regex_str: str

    # Afterwards, since it is not known at the time of creation
    is_duplicate: bool = None
    imputed_light: float = None
    imputed_heavy: float = None

    @property
    def total_intensity(self) -> float:
        return self.light + self.heavy

    @property
    def ip2_sequence(self) -> str:
        return self.cpp_result.ip2_sequence

    @cached_property
    def proforma_sequence(self) -> str:
        return self.cpp_result.proforma_sequence

    @cached_property
    def unmodified_sequence(self) -> str:
        return self.cpp_result.unmodified_sequence

    @property
    def charge(self) -> int:
        return self.cpp_result.charge

    @property
    def filename(self) -> str:
        return self.cpp_result.filename

    @filename.setter
    def filename(self, value: str):
        self.cpp_result.filename = value

    @property
    def scannr(self) -> int:
        return self.cpp_result.s_line.scannr

    @cached_property
    def regex_sites(self) -> List[int]:
        regex_sites = [int(match.start()) for match in
                       regex.finditer(self.regex_str, self.unmodified_sequence, overlapped=True)]
        return regex_sites

    @cached_property
    def regex_site(self) -> int:
        if self.is_single:
            assert len(self.regex_sites) == 1
        elif self.is_double:
            assert len(self.regex_sites) == 2

        return self.regex_sites[self.cpp_result_index]

    @property
    def is_valid(self) -> bool:
        if self.is_single and len(self.regex_sites) == 1:
            return True
        elif self.is_double and len(self.regex_sites) == 2:
            return True
        else:
            return False

    @property
    def is_missing(self) -> bool:
        return self.light == 0.0 and self.heavy == 0.0

    @property
    def duplicate_key(self) -> Tuple[str, int]:
        return self.filename, self.scannr

    @property
    def loci(self) -> List[str]:
        return self.cpp_result.loci

    @property
    def is_decoy(self) -> bool:
        return all('reverse' in protein.lower() for protein in self.loci)

    @property
    def is_contaminant(self) -> bool:
        return any('contaminant' in protein for protein in self.loci)

    @property
    def is_double(self) -> bool:
        return self.cpp_result.is_double

    @property
    def is_single(self) -> bool:
        return self.cpp_result.is_single

    @property
    def light(self) -> float:

        if self.imputed_light is not None:
            return self.imputed_light

        return self.cpp_result.light[self.cpp_result_index]

    @property
    def heavy(self) -> float:
        if self.imputed_heavy is not None:
            return self.imputed_heavy

        return self.cpp_result.heavy[self.cpp_result_index]

    @cached_property
    def ratio(self) -> float:

        if self.heavy == 0.0 and self.light == 0.0:
            return np.nan

        if self.heavy == 0.0:
            return np.inf

        if self.light == 0.0:
            return 0.0

        return np.divide(self.light, self.heavy)

    @cached_property
    def log2_ratio(self) -> float:

        if self.ratio == 0.0:
            return -np.inf

        if self.ratio == np.inf:
            return np.inf

        return np.log2(self.ratio)

    @cached_property
    def log10_ratio(self) -> float:

        if self.ratio == 0.0:
            return -np.inf

        if self.ratio == np.inf:
            return np.inf

        return np.log10(self.ratio)

    @cached_property
    def peptide_site_str(self) -> str:
        return f'{self.proforma_sequence}@{self.regex_str}{self.regex_site + 1}'

    @cached_property
    def protein_site_str(self) -> str:
        protein_site_str = ''
        for locus, peptide_indices in zip(self.loci, self.peptide_indices):
            protein_site_str += locus
            for i in peptide_indices:
                protein_site_str += f'@{self.regex_str}{i + self.regex_site + 1}'
            protein_site_str += ';'
        return protein_site_str

    @cached_property
    def loci_str(self) -> str:
        return ';'.join(self.loci)


@dataclass
class RatioResult:
    quant_results: List[QuantResult]
    ratio_rollup: Callable
    grouping: List[str]  # attribute names of QuantResult2
    grouping_vals: List[Any]  # values of the grouping attributes

    @property
    def is_valid(self) -> bool:
        if self.quant_results is None:
            return False
        if len(self.quant_results) == 0:
            return False
        return True

    @property
    def ratio_type(self) -> str:

        if not self.is_valid:
            return 'invalid'

        single_count = sum(qr.is_single for qr in self.quant_results)
        double_count = sum(qr.is_double for qr in self.quant_results)

        if single_count > 0 and double_count == 0:
            return 'single'

        if single_count == 0 and double_count > 0:
            return 'double'

        return 'mixed'

    @cached_property
    def _rollup(self):
        if not self.is_valid:
            return np.nan, np.nan, 0
        return self.ratio_rollup(self.quant_results)

    @property
    def log2_ratio(self) -> float:
        return self._rollup[0]

    @property
    def log2_ratio_std(self) -> float:
        return self._rollup[1]

    @property
    def cnt(self) -> int:
        """
        Returns the number of results used in the rollup calculation.
        """
        return int(self._rollup[2])

    def to_dict(self, add_peptides: False, add_proteins: False) -> Dict[str, Any]:
        d = {c: self._get_quant_result_attribute(c) for c in self.grouping}
        d2 = self.to_value_dict(add_peptides, add_proteins)

        for k, v in d2.items():
            d[k] = v

        return d

    def to_value_dict(self, add_peptides: False, add_proteins: False) -> Dict[str, Any]:
        d = {}
        d['log2_ratio'] = self.log2_ratio
        d['log2_ratio_std'] = self.log2_ratio_std
        d['cnt'] = self.cnt
        d['type'] = self.ratio_type

        if add_peptides:
            d['peptide_site_str'] = ';'.join(self.peptide_site_strs)

        if add_proteins:
            d['protein_site_str'] = ';'.join(self.protein_site_strs)

        return d

    def _get_quant_result_attribute(self, attribute: str) -> Any:
        if not self.is_valid:
            return None

        assert all(qr.__getattribute__(attribute) == self.quant_results[0].__getattribute__(attribute) for qr in
                   self.quant_results)
        return self.quant_results[0].__getattribute__(attribute)

    @property
    def group(self):
        for c, v in zip(self.grouping, self.grouping_vals):
            if c == 'group':
                return v

    @property
    def non_group_key(self) -> tuple[Any, ...]:
        return tuple([v for l, v in zip(self.grouping, self.grouping_vals) if l != 'group'])

    @property
    def non_group_key_labels(self) -> tuple[Any, ...]:
        return tuple([l for l, v in zip(self.grouping, self.grouping_vals) if l != 'group'])

    @property
    def ip2_sequences(self) -> List[str]:
        if not self.is_valid:
            return []
        return list(set([qr.ip2_sequence for qr in self.quant_results]))

    @property
    def proforma_sequences(self) -> List[str]:
        if not self.is_valid:
            return []
        return list(set([qr.proforma_sequence for qr in self.quant_results]))

    @property
    def unmodified_sequences(self) -> List[str]:
        if not self.is_valid:
            return []
        return list(set([qr.unmodified_sequence for qr in self.quant_results]))

    @property
    def peptide_site_strs(self) -> List[str]:
        if not self.is_valid:
            return []
        return list(set([qr.peptide_site_str for qr in self.quant_results]))

    @property
    def protein_site_strs(self) -> List[str]:
        if not self.is_valid:
            return []
        return list(set([qr.protein_site_str for qr in self.quant_results]))


@dataclass
class CompareRatio:
    pair: Pair
    group1_ratio: RatioResult
    group2_ratio: RatioResult
    grouping: List[str]  # attribute names of QuantResult2
    grouping_vals: List[Any]  # values of the grouping attributes

    # Afterwards
    qvalue: float = None

    @property
    def group1(self) -> Any:
        return self.pair.group1

    @property
    def group1_log2_ratio(self) -> float:
        if self.group1_ratio is None:
            return np.nan
        return self.group1_ratio.log2_ratio

    @property
    def group1_std(self) -> float:
        if self.group1_ratio is None:
            return np.nan
        return self.group1_ratio.log2_ratio_std

    @property
    def group1_cnt(self) -> int:
        if self.group1_ratio is None:
            return 0
        return self.group1_ratio.cnt

    @property
    def group2(self) -> Any:
        return self.pair.group2

    @property
    def group2_log2_ratio(self) -> float:
        if self.group2_ratio is None:
            return np.nan
        return self.group2_ratio.log2_ratio

    @property
    def group2_std(self) -> float:
        if self.group2_ratio is None:
            return np.nan
        return self.group2_ratio.log2_ratio_std

    @property
    def group2_cnt(self) -> int:
        if self.group2_ratio is None:
            return 0
        return self.group2_ratio.cnt

    @property
    def is_valid(self) -> bool:
        if self.group1_ratio is None or self.group2_ratio is None:
            return False
        return self.group1_ratio.cnt > 0 and self.group2_ratio.cnt > 0

    @cached_property
    def _ttest(self) -> Tuple[float, float]:
        if not self.is_valid:
            return np.nan, np.nan

        t_stat, p_value = ttest_ind_from_stats(self.group1_ratio.log2_ratio,
                                               self.group1_ratio.log2_ratio_std,
                                               self.group1_ratio.cnt,
                                               self.group2_ratio.log2_ratio,
                                               self.group2_ratio.log2_ratio_std,
                                               self.group2_ratio.cnt,
                                               equal_var=False)

        return t_stat, p_value

    @property
    def test_statistic(self) -> float:
        return self._ttest[0]

    @property
    def pvalue(self) -> float:
        return self._ttest[1]

    @property
    def log2_ratio_diff(self) -> float:
        if not self.is_valid:
            return np.nan

        return self.group1_ratio.log2_ratio - self.group2_ratio.log2_ratio

    @property
    def log2_ratio_diff_std(self) -> float:
        if not self.is_valid:
            return np.nan

        return float(sum_uncertainty(self.group1_ratio.log2_ratio_std, self.group2_ratio.log2_ratio_std))

    @property
    def cnt(self) -> int:
        if not self.is_valid:
            return 0

        return int(self.group1_ratio.cnt + self.group2_ratio.cnt)

    def to_dict(self, add_peptides: False, add_proteins: False) -> Dict[str, Any]:

        d1 = {}
        for l, v in zip(self.grouping, self.grouping_vals):
            d1[l] = v

        d2 = self.to_value_dict(add_peptides, add_proteins)

        if add_peptides:
            d2['peptide_site_str'] = ';'.join(self.peptide_site_strs)

        if add_proteins:
            d2['protein_site_str'] = ';'.join(self.protein_site_strs)

        d = {**d1, **d2}
        return d

    def to_value_dict(self, add_peptides: False, add_proteins: False) -> Dict[str, Any]:
        d = {
            'group1': self.group1,
            'group2': self.group2,
            'group1_log2_ratio': self.group1_log2_ratio,
            'group1_std': self.group1_std,
            'group1_cnt': self.group1_cnt,
            'group2_log2_ratio': self.group2_log2_ratio,
            'group2_std': self.group2_std,
            'group2_cnt': self.group2_cnt,
            'log2_ratio_diff': self.log2_ratio_diff,
            'diff_std': self.log2_ratio_diff_std,
            'total_cnt': self.cnt,
            'test_statistic': self.test_statistic,
            'pvalue': self.pvalue,
            'qvalue': self.qvalue
        }

        if add_peptides:
            d['peptide_site_str'] = ';'.join(self.peptide_site_strs)

        if add_proteins:
            d['protein_site_str'] = ';'.join(self.protein_site_strs)

        return d

    @property
    def ip2_sequences(self) -> List[str]:
        ip2_sequences = set()

        if self.group1_ratio is not None:
            ip2_sequences.update(self.group1_ratio.ip2_sequences)

        if self.group2_ratio is not None:
            ip2_sequences.update(self.group2_ratio.ip2_sequences)

        return list(ip2_sequences)

    @property
    def proforma_sequences(self) -> List[str]:
        proforma_sequences = set()

        if self.group1_ratio is not None:
            proforma_sequences.update(self.group1_ratio.proforma_sequences)

        if self.group2_ratio is not None:
            proforma_sequences.update(self.group2_ratio.proforma_sequences)

        return list(proforma_sequences)


    @property
    def unmodified_sequences(self) -> List[str]:
        unmodified_sequences = set()
        if self.group1_ratio is not None:
            unmodified_sequences.update(self.group1_ratio.unmodified_sequences)

        if self.group2_ratio is not None:
            unmodified_sequences.update(self.group2_ratio.unmodified_sequences)

        return list(unmodified_sequences)

    @property
    def peptide_site_strs(self) -> List[str]:
        peptide_site_strs = set()
        if self.group1_ratio is not None:
            peptide_site_strs.update(self.group1_ratio.peptide_site_strs)

        if self.group2_ratio is not None:
            peptide_site_strs.update(self.group2_ratio.peptide_site_strs)

        return list(peptide_site_strs)

    @property
    def protein_site_strs(self) -> List[str]:
        protein_site_strs = set()

        if self.group1_ratio is not None:
            protein_site_strs.update(self.group1_ratio.protein_site_strs)

        if self.group2_ratio is not None:
            protein_site_strs.update(self.group2_ratio.protein_site_strs)

        return list(protein_site_strs)

