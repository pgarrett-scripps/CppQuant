from dataclasses import dataclass
from typing import Dict, IO, List


@dataclass
class Fasta:
    headers: List[str]
    sequences: List[str]
    max_sequence_length: int


def read_fasta(fasta: IO[str]) -> Fasta:
    """
    Read a FASTA file and return the headers and sequences.
    :param fasta: The FASTA file.
    :type fasta: IO[str]

    :return: The headers and sequences.
    :rtype: Fasta
    """

    fasta.seek(0)

    headers, sequences, max_sequence_length = [], [], 0
    for line in fasta:
        line = line.rstrip()
        if line.startswith('>'):
            headers.append(line)
            sequences.append('')
        else:
            sequences[-1] += line
            max_sequence_length = max(max_sequence_length, len(line))

    return Fasta(headers, sequences, max_sequence_length)


def map_ip2_fasta_id_to_sequence(fasta: Fasta) -> Dict[str, str]:
    d = {}
    for header, sequence in zip(fasta.headers, fasta.sequences):

        ip2_id = None
        if header.startswith('>sp|') or header.startswith('>tr|'):
            ip2_id = header.split('|')[1].strip()

        elif header.startswith('>Reverse_sp|') or header.startswith('>Reverse_tr|'):
            ip2_id = header[1:].split()[0]
            #continue

        elif header.startswith('>Reverse_contaminant_'):
            ip2_id = header[1:].strip().split()[0]
            #continue

        elif header.startswith('>contaminant_'):
            ip2_id = header[1:].strip().split()[0]

        elif header.startswith('>Reverse_'):
            ip2_id = header[1:]
            #continue

        elif header.startswith('>'):
            ip2_id = header[1:].strip()

        else:
            raise ValueError(f"Invalid header: {header}")

        if ip2_id:
            d[ip2_id] = sequence

    return d