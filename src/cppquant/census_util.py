from typing import Iterable, Generator

from cppquant.dclass import CensusResult, PLine, SLine, DLine


def build_census_results(census_file: Iterable) -> Generator[CensusResult, None, None]:
    cpp_result = None
    new_protein_flag = True

    p_header_map, s_header_map, d_header_map = None, None, None

    for i, line in enumerate(census_file):

        try:
            line = line.decode()
        except AttributeError:
            pass

        elements = line.strip().split('\t')

        match elements[0]:

            case 'H':

                match elements[1]:
                    case 'PLINE':
                        p_header = elements[2:]
                        p_header_map = {header: i for i, header in enumerate(p_header)}
                    case 'SLINE':
                        s_header = elements[2:]
                        s_header_map = {header: i for i, header in enumerate(s_header)}
                    case 'DLINE':
                        d_header = elements[2:]
                        d_header_map = {header: i for i, header in enumerate(d_header)}
                    case _:
                        continue

            case 'P':

                if new_protein_flag is True:

                    if cpp_result is not None:
                        yield cpp_result

                    cpp_result = CensusResult()
                    new_protein_flag = False

                cpp_result.p_lines.append(PLine(p_header_map, list(elements[1:])))

            case '&S' | 'S':
                new_protein_flag = True

                cpp_result.s_lines.append(SLine(s_header_map, elements[1:]))

            case '&D' | 'D':
                new_protein_flag = True

                cpp_result.s_lines[-1].d_lines.append(DLine(d_header_map, elements[1:]))

            case '':
                continue

            case _:
                raise ValueError(f'Invalid census file format: line {i}')

    if cpp_result is not None:
        yield cpp_result