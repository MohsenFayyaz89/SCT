import datetime
import random
import subprocess
from typing import Tuple, Union, Set, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler


def set_seed(seed: int, fully_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def send_to_device(
    items: Union[Tensor, Tuple[Tensor, ...], List[Tensor]], device
) -> Union[Tensor, Tuple[Tensor, ...], List[Tensor]]:
    if type(items) is tuple:
        return tuple(map(lambda x: x.to(device), items))
    elif type(items) is list:
        return list(map(lambda x: x.to(device), items))
    else:
        return items.to(device)


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        # this is probably not in a git repo or git is not installed.
        return ""


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def tensors_to_numpys(
    x: Union[Tensor, Tuple[Tensor, ...]]
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if type(x) is not tuple:
        return tensor_to_numpy(x)
    else:
        return tuple(map(lambda i: tensor_to_numpy(i), x))


def change_multiprocess_strategy():
    torch.multiprocessing.set_sharing_strategy("file_system")


def print_with_time(the_thing: str):
    print("[{}] {}".format(str(datetime.datetime.now()), the_thing))


class OverfitSampler(Sampler):
    # TODO: Mohsen: Better to make different dataset objects for overfitting
    #  to have right value for len() in tqdm (used in train and evaluate functions)
    def __init__(self, main_source, indices, num_iter=0):
        """

        :param main_source:
        :param indices:
        :param num_iter:
          0: how_many=main_source_len/len(self.indices),
          otherwise: how_many=1
        """
        super().__init__(main_source)
        self.main_source = main_source
        self.indices = indices

        if num_iter == 0:
            main_source_len = len(self.main_source)
            how_many = int(round(main_source_len / len(self.indices)))
        else:
            how_many = 1

        self.to_iter_from = []
        for _ in range(how_many):
            self.to_iter_from.extend(self.indices)

    def __iter__(self):
        return iter(self.to_iter_from)

    def __len__(self):
        return len(self.main_source)


def tensor_to_set(input_tensor: Tensor, thr: float) -> Set[int]:
    """
    converts predicted tensor of sets to python built-in set
    :param input_tensor: [C]
    :param thr: 0.0 <= thr <= 1.0
    :return: \set\=M'
    """
    s = input_tensor.detach().cpu().numpy()
    s = np.argwhere(s >= thr)
    return set(s.flatten())


def set_to_tensor(s: set) -> Tensor:
    """

    :param s:
    :return:
    """
    return torch.Tensor(sorted(list(s))).long()
