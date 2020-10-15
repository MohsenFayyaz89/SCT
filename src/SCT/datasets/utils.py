from typing import Tuple, List

from SCT.datasets import breakfast

KINETICS_FEAT_NAME = "i3d"
IDT_FEAT_NAME = "idt"
FEAT_DIM_MAPPING = {KINETICS_FEAT_NAME: 2048, IDT_FEAT_NAME: 64}
MAPPING_FILE_NAME = "mapping.txt"
LIST_OF_DATASETS = ["breakfast"]

BOS_I = 0
BOS_S = "_bos_"
EOS_I = 1
EOS_S = "_eos_"
BASE_STOI = {BOS_S: BOS_I, EOS_S: EOS_I}
LEN_EXTRA_WORDS = len(BASE_STOI)



def create_tf_input(transcript: List[int]) -> List[int]:
    return [BOS_I] + [x + LEN_EXTRA_WORDS for x in transcript]


def create_tf_target(transcript: List[int]) -> List[int]:
    return [x + LEN_EXTRA_WORDS for x in transcript] + [EOS_I]


def create_tf_set_target(transcript: List[int], num_classes: int) -> List[int]:
    set_target = [0]*num_classes
    for t in transcript:
    	set_target[t] = 1
    return set_target


def summarize_list(the_list: List[int]) -> Tuple[List[int], List[int]]:
    """
    Given a list of items, it summarizes them in a way that no two neighboring values are the same.
    It also returns the size of each section.
    e.g. [4, 5, 5, 6] -> [4, 5, 6], [1, 2, 1]
    """
    summary = []
    lens = []
    if len(the_list) > 0:
        current = the_list[0]
        summary.append(current)
        lens.append(1)
        for item in the_list[1:]:
            if item != current:
                current = item
                summary.append(item)
                lens.append(1)
            else:
                lens[-1] += 1
    return summary, lens


def unsummarize_list(labels: List[int], lengths: List[int]) -> List[int]:
    """
    Does the reverse of summarize list. You give it a list of segment labels and their lengths and it returns the full
    labels for the full sequence.
    e.g. ([4, 5, 6], [1, 2, 1]) -> [4, 5, 5, 6]
    """
    assert len(labels) == len(lengths)

    the_sequence = []
    for label, length in zip(labels, lengths):
        the_sequence.extend([label] * length)

    return the_sequence
