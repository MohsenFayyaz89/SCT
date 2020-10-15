import os
from dataclasses import dataclass
from random import randint
from typing import List, Set, Dict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from SCT.utils import send_to_device, set_to_tensor
from .utils import (
    summarize_list,
    create_tf_input,
    create_tf_target,
    create_tf_set_target,
)


@dataclass
class BatchItem:
    """
    T: the video length
    D: the feat dim
    M: number of actions in the video.
    """

    video_name: str
    feats: Tensor  # [1 x D x T] float
    gt_label: Tensor  # [T] long
    transcript: Tensor  # [M] long
    action_lengths: Tensor  # [M] int
    tf_input: Tensor  # [M + 1] long: equal to BOS + transcript
    tf_target: Tensor  # [M + 1] long: equal to transcript + EOS
    tf_set_target: Tensor  # [C] long
    gt_set: Set[int]  # |gt_set|=M
    gt_set_list: List[Set[int]]  # |set(gt_set)|=M
    A_hat: Tensor
    target_indices_list: List[Tensor]
    not_target_indices: Tensor
    not_target_indices_list: List[Tensor]
    not_target_set: Set[int]  # |not_target_set|=|U| - M
    not_target_set_list: List[Set[int]]  # |not_target_set|=|U| - M
    target_indices_mapping: Dict
    t_videos: List[int]  # lengths of the input videos
    T: int  # video length

    def to(self, device):
        self.feats = send_to_device(self.feats, device)
        self.gt_label = send_to_device(self.gt_label, device)
        self.transcript = send_to_device(self.transcript, device)
        self.action_lengths = send_to_device(self.action_lengths, device)
        self.tf_input = send_to_device(self.tf_input, device)
        self.tf_target = send_to_device(self.tf_target, device)
        self.tf_set_target = send_to_device(self.tf_set_target, device)
        self.A_hat = send_to_device(self.A_hat, device)
        self.target_indices_list = send_to_device(self.target_indices_list, device)
        self.not_target_indices = send_to_device(self.not_target_indices, device)
        self.not_target_indices_list = send_to_device(
            self.not_target_indices_list, device
        )


class GeneralDataset(Dataset):
    def __init__(
        self,
        root: str,
        feat_list: str = None,
        gt_list: str = None,
        mapping_file: str = None,
        feat_dim: int = -1,
        num_classes: int = -1,
        rnd_flip: bool = False,
        rnd_cat: bool = False,
        rnd_cat_n_vid: int = 0,
    ):
        """
        feat_list: a file containing the relative path to numpy files with video features in them, separated by new line
                   video features should have the shape: (feat_dim, n_frames)
        gt_list: a file containing the relative path to txt files with framewise labels in them, separated by new line.
        mapping_file: a file containing the mapping from integers to gt_labels.
        """
        self.root = root
        self.feat_list = feat_list
        self.gt_list = gt_list
        self.mapping_file = mapping_file
        self.end_class_id = 0
        self.mof_eval_ignore_classes = []
        self.n_classes = num_classes
        self.background_class_ids = [0]
        # following are defaults, should be set
        self.feat_dim = feat_dim
        self.convenient_name = None
        self.split = -1
        self.max_transcript_length = 100
        self.rnd_flip = rnd_flip
        self.rnd_cat = rnd_cat
        self.rnd_cat_n_vid = rnd_cat_n_vid

        if self.feat_list is not None:
            with open(self.feat_list) as f:
                self.feat_file_paths = [x.strip() for x in f]
        else:
            self.feat_file_paths = []

        if self.gt_list is not None:
            with open(self.gt_list) as f:
                self.gt_file_paths = [x.strip() for x in f]
        else:
            self.gt_file_paths = []

        self.mapping = {}
        self.inverse_mapping = {}
        if self.mapping_file is not None:
            with open(self.mapping_file) as f:
                the_mapping = [tuple(x.strip().split()) for x in f]

                for (i, l) in the_mapping:
                    self.mapping[int(i)] = l
                    self.inverse_mapping[l] = int(i)

        assert len(self.feat_file_paths) == len(self.gt_file_paths)

    @property
    def num_classes(self) -> int:
        return len(self.mapping)

    def __len__(self) -> int:
        return len(self.feat_file_paths)

    def __getitem__(self, item: int, no_features: bool = False) -> BatchItem:
        """
        parameters:
            no_features: if set to True, this method will *not* actually load_model the features from disk.
            This is useful if we want to quickly run some code which doesn't need the features.
        """
        feat_file_path = os.path.join(self.root, self.feat_file_paths[item])
        gt_file_path = os.path.join(self.root, self.gt_file_paths[item])

        if not no_features:
            vid_feats = torch.tensor(torch.load(feat_file_path)).float()
        else:
            vid_feats = torch.tensor([0])

        # vid_feats.t_()

        with open(gt_file_path) as f:
            gt_label_names = [
                x.strip() for x in f.read().split("\n") if len(x.strip()) > 0
            ]

        gt_label_ids = [self.inverse_mapping[x] for x in gt_label_names]
        weak_label_ids, weak_label_lens = summarize_list(gt_label_ids)

        gt_labels = torch.tensor(gt_label_ids).long()
        gt_action_lengths = torch.tensor(weak_label_lens).int()
        weak_labels = torch.tensor(weak_label_ids).long()
        weak_label_tf = torch.tensor(create_tf_input(weak_label_ids)).long()
        weak_label_target = torch.tensor(create_tf_target(weak_label_ids)).long()
        set_label_target = torch.tensor(
            create_tf_set_target(weak_label_ids, self.n_classes)
        ).float()
        gt_set_list = [set(weak_label_ids)]
        gt_set = set(weak_label_ids)
        target_indices_list = [set_to_tensor(gt_set)]
        not_target_set_list = [set(range(self.num_classes)).difference(set(gt_set))]
        not_target_indices_list = [set_to_tensor(not_target_set_list[0])]
        # fixme: make it object oriented
        if self.rnd_flip:
            if randint(0, 1) == 1:
                vid_feats = vid_feats.flip(1)
        t_vids = [vid_feats.shape[1]]

        # fixme: make it object oriented
        feats = [vid_feats]
        if self.rnd_cat:
            if randint(0, 1) == 1:
                for i in range(self.rnd_cat_n_vid):
                    item = randint(0, self.__len__() - 1)
                    feat_file_path = os.path.join(self.root, self.feat_file_paths[item])
                    gt_file_path = os.path.join(self.root, self.gt_file_paths[item])
                    f = torch.tensor(np.load(feat_file_path)).float().t_()
                    # fixme: make it object oriented
                    if self.rnd_flip:
                        if randint(0, 1) == 1:
                            f = f.flip(1)
                    t_vids.append(f.shape[1])
                    feats.append(f)
                    with open(gt_file_path) as f:
                        gt_label_names = [
                            x.strip()
                            for x in f.read().split("\n")
                            if len(x.strip()) > 0
                        ]
                    gt_label_ids = [self.inverse_mapping[x] for x in gt_label_names]
                    weak_label_ids, weak_label_lens = summarize_list(gt_label_ids)
                    gt_set_list.append(set(weak_label_ids))
                    target_indices_list.append(set_to_tensor(gt_set_list[-1]))
                    not_target_set_list.append(
                        set(range(self.num_classes)).difference(set(weak_label_ids))
                    )
                    not_target_indices_list.append(
                        set_to_tensor(not_target_set_list[-1])
                    )
                    gt_set = gt_set.union(set(weak_label_ids))

                set_label_target = torch.zeros(self.n_classes)
                set_label_target[list(gt_set)] = 1.0
                vid_feats = torch.cat(feats, dim=1)

        vid_feats.unsqueeze_(0)
        not_target_set = set(range(self.num_classes)).difference(set(gt_set))
        not_target_indices = set_to_tensor(not_target_set)
        target_indices = set_to_tensor(gt_set)
        target_indices_mapping = {
            target_indices[i].item(): i for i in range(0, target_indices.shape[0])
        }
        T = vid_feats.shape[2]

        return BatchItem(
            video_name=feat_file_path.split("/")[-1][:-4],
            feats=vid_feats,
            gt_label=gt_labels,
            transcript=weak_labels,
            action_lengths=gt_action_lengths,
            tf_input=weak_label_tf,
            tf_target=weak_label_target,
            tf_set_target=set_label_target,
            gt_set=gt_set,
            gt_set_list=gt_set_list,
            A_hat=target_indices,
            target_indices_list=target_indices_list,
            not_target_indices=not_target_indices,
            not_target_indices_list=not_target_indices_list,
            not_target_set=not_target_set,
            not_target_set_list=not_target_set_list,
            target_indices_mapping=target_indices_mapping,
            t_videos=t_vids,
            T=T,
        )

    @staticmethod
    def concat_videos(items: List[BatchItem]) -> BatchItem:
        output = BatchItem
        features = []
        sets = set()
        for batch in items:
            features.append(batch.feats)
            sets = sets.union(batch.gt_set)
        num_classes = items[0].tf_set_target.shape[0]
        set_target = torch.zeros(num_classes)
        set_target[list(sets)] = 1.0
        output.feats = torch.cat(features, dim=2)
        output.gt_set = sets
        output.tf_set_target = set_target
        return output

    @staticmethod
    def collate_fn(items: List[BatchItem]) -> BatchItem:
        """
        We assume batch_size = 1
        """
        return items[0]
