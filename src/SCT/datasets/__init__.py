from yacs.config import CfgNode

from SCT.datasets.breakfast import create_breakfast_dataset
from SCT.datasets.general_dataset import GeneralDataset


def make_db(cfg: CfgNode, train: bool):
    dataset_name = cfg.dataset.name

    if dataset_name == breakfast.DATASET_NAME:
        db = create_breakfast_dataset(cfg, train)
    else:
        raise Exception("dataset not found. (name: %s)" % dataset_name)

    return db
