from difflib import SequenceMatcher
from typing import List, Set

import numpy as np
import torch
from torch.nn.functional import interpolate

from SCT.utils import tensor_to_numpy


def make_same_size(
    prediction: np.ndarray, target: np.ndarray, background: int = 0
) -> np.ndarray:
    """
    Tries to use some heuristic to make the prediction the same size as the target.
    If the prediction is shorter, it will add background class at the end.
    If the prediction is longer, it will crop to the size of the target.
    :returns predictions. It will return the updated predictions file.
    """

    t_len = len(target)
    p_len = len(prediction)

    if p_len == t_len:
        return prediction
    elif p_len > t_len:
        new_predictions = prediction.copy()
        extra_len = p_len - t_len
        new_predictions = new_predictions[:-extra_len]
    else:  # p_len < t_len
        new_predictions = prediction.copy()
        remaining_len = t_len - p_len
        bg = np.full(remaining_len, fill_value=background)
        new_predictions = np.concatenate((new_predictions, bg), axis=0)
    return new_predictions


def make_same_size_interpolate(
    prediction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    It will use nearest neighbor interpolation to make the prediction the same size as the target.
    """
    t_len = len(target)

    prediction_tensor = torch.tensor(prediction).float()
    prediction_tensor_resized = interpolate(
        prediction_tensor, size=t_len, mode="nearest"
    )

    return tensor_to_numpy(prediction_tensor_resized.long())


# noinspection PyPep8Naming
def MoF(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    ignore_ids: List[int] = None,
) -> float:
    """
    Calculates the Mean over frames segmentation metric.
    :param predictions: List of numpy arrays. Each array is assumed to do 1D. It should contain the id of the predicted
    frame label.
    :param targets: List of numpy arrays.
    :param ignore_ids: The list of ids that have to be ignored during evaluation.
    :return: the mean over frame metric. It is between 0 and 1.
    """
    if ignore_ids is None:
        ignore_ids = []
    if type(ignore_ids) == int:
        ignore_ids = [ignore_ids]

    assert len(predictions) == len(targets)
    total = 0
    correct = 0
    for i in range(len(predictions)):
        p = predictions[i]
        t = targets[i]

        assert len(p) == len(t)

        where_to_consider = np.ones(len(p))
        for iid in ignore_ids:
            where_to_consider[np.where(t == iid)] = 0

        where_to_consider = np.where(where_to_consider)

        total += len(p[where_to_consider])
        correct += (p[where_to_consider] == t[where_to_consider]).sum()

    return float(correct) / total

def matching_score(gt_transcript: List[int], predicted_transcript: List[int]) -> float:
    return SequenceMatcher(a=gt_transcript, b=predicted_transcript).ratio()