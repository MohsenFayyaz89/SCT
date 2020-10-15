from pathlib import Path

from yacs.config import CfgNode

from SCT.datasets.general_dataset import GeneralDataset
from SCT.datasets.utils import MAPPING_FILE_NAME, FEAT_DIM_MAPPING

POSSIBLE_SPLITS = [0, 1, 2, 3, 4]
MAX_TRANSCRIPT_LENGTH = 25
DATASET_NAME = "breakfast"
NUM_CLASSES = 48


def create_breakfast_dataset(cfg: CfgNode, train: bool = True) -> GeneralDataset:
    split = cfg.dataset.split
    feat_name = cfg.dataset.feat_name
    root = Path(cfg.dataset.root)
    assert split in POSSIBLE_SPLITS
    db_path = root / "datasets" / DATASET_NAME
    rnd_flip = False
    rnd_cat = False
    rnd_cat_n_vid = 0
    if train:
        rnd_flip = cfg.training.random_flip
        rnd_cat = cfg.training.random_concat.active
        rnd_cat_n_vid = cfg.training.random_concat.num_videos
    feat_list = db_path / "split{sn}_{tt}_feats_{fn}.txt".format(
        sn=split, tt="train" if train else "test", fn=feat_name
    )

    gt_list = db_path / "split{sn}_{tt}_labels.txt".format(
        sn=split, tt="train" if train else "test"
    )

    mapping = db_path / MAPPING_FILE_NAME

    db = GeneralDataset(
        root=root,
        feat_list=feat_list,
        gt_list=gt_list,
        mapping_file=mapping,
        feat_dim=FEAT_DIM_MAPPING[feat_name],
        num_classes= NUM_CLASSES,
        rnd_flip=rnd_flip,
        rnd_cat=rnd_cat,
        rnd_cat_n_vid=rnd_cat_n_vid
    )
    db.end_class_id = 0
    db.mof_eval_ignore_classes = []
    db.background_class_ids = [0]
    db.convenient_name = DATASET_NAME
    db.split = split
    db.max_transcript_length = MAX_TRANSCRIPT_LENGTH

    return db
