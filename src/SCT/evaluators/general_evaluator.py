from dataclasses import dataclass
from os import mkdir
from os.path import exists
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode
import edit_distance

from SCT.datasets import GeneralDataset
from SCT.datasets.general_dataset import BatchItem
from SCT.datasets.utils import unsummarize_list, summarize_list
from SCT.evaluators.metrics import (
    make_same_size_interpolate,
    matching_score,
    MoF,
)
from SCT.models import GeneralModel
from SCT.utils import print_with_time, tensor_to_numpy, OverfitSampler


@dataclass
class EvalResult:
    mof: float


@dataclass
class FinalEvalResult(EvalResult):
    viterbi_mof: float
    viterbi_mof_rnn: float


class GeneralEvaluator(object):
    def __init__(
        self,
        cfg: CfgNode,
        model: GeneralModel,
        dataset: GeneralDataset,
        teacher_forcing: bool = False,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.dataloader = self.create_eval_dataloader(
            cfg=self.cfg, dataset=self.dataset
        )
        self.device = torch.device(self.cfg.system.device)
        self.ignore_classes = cfg.training.evaluators.ignore_classes

        self.predicted_sets = []
        self.target_sets = []
        self.target_segmentations = []
        self.predicted_segmentations = []
        self.masks_prediction_segmentations = []
        self.lengths_prediction_segmentations = []

        self.target_sentences = []
        self.predicted_sentences = []

        self.IoU_values = []

    def convert_seg_to_text(self, seg_pred: torch.Tensor) -> str:
        return "\n".join([self.dataset.mapping[x] for x in seg_pred])

    def log_inference(
        self, video_name, predictions, target_segmentation, target_transcript
    ):
        pred_trn = summarize_list(predictions)[0]
        ms = matching_score(
            gt_transcript=target_transcript.tolist(), predicted_transcript=pred_trn
        )
        if (
            matching_score(
                gt_transcript=target_transcript.tolist(),
                predicted_transcript=[0, 24, 2, 25, 0],
            )
            == 1
        ):
            print(video_name)
        ms *= 100
        ms = str(ms)  # [:2]

        if not exists("./log/" + ms):
            mkdir("./log/" + ms)

        file = open("./log/" + ms + "/" + video_name + ".txt", "w")
        text = self.convert_seg_to_text(predictions)
        file.write(text)
        file.close()

        file = open("./log/" + ms + "/" + video_name + "_target.txt", "w")
        text = self.convert_seg_to_text(target_segmentation)
        file.write(text)
        file.close()

    @staticmethod
    def create_eval_dataloader(cfg: CfgNode, dataset: GeneralDataset) -> DataLoader:
        """

        :param cfg:
        :param dataset:
        :return:
        """
        if cfg.training.overfit:
            sampler = OverfitSampler(
                main_source=dataset, indices=cfg.training.overfit_indices, num_iter=1
            )

            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=sampler,
                num_workers=cfg.system.num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=1,
                num_workers=cfg.system.num_workers,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                pin_memory=True,
            )

    # noinspection PyPep8Naming
    def eval_1_batch(self, batch: BatchItem):
        batch.to(self.device)
        forward_out = self.model.forward(batch)

        _, Y_pred = forward_out.Y.topk(1, dim=1)
        _, A_pred = forward_out.A.topk(1, dim=1)
        target_segmentation = tensor_to_numpy(batch.gt_label)
        predicted_segmentation = make_same_size_interpolate(
            tensor_to_numpy(Y_pred), target=target_segmentation
        ).reshape(-1)
        masks_prediction_segmentation = make_same_size_interpolate(
            tensor_to_numpy(A_pred), target=target_segmentation
        ).reshape(-1)

        self.target_segmentations.append(target_segmentation)
        self.predicted_segmentations.append(predicted_segmentation)
        self.masks_prediction_segmentations.append(masks_prediction_segmentation)
        if self.cfg.experiment.log_inference_output:
            self.log_inference(batch.video_name, target_segmentation, batch.transcript)

    def reset(self):
        self.predicted_sets = []
        self.target_sets = []
        self.target_segmentations = []
        self.predicted_segmentations = []
        self.masks_prediction_segmentations = []
        self.lengths_prediction_segmentations = []

        self.target_sentences = []
        self.predicted_sentences = []

        self.IoU_values = []

    def compute_metrics(self, ignore_classes: List[int] = None) -> EvalResult:
        mof = MoF(
            predictions=self.predicted_segmentations,
            targets=self.target_segmentations,
            ignore_ids=ignore_classes,
        )
        return EvalResult(
            mof=mof,
        )

    def evaluate(self) -> Dict[str, EvalResult]:
        print_with_time("Evaluating ...")
        self.reset()
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                self.eval_1_batch(batch)

        result = {"All": self.compute_metrics()}

        if not self.ignore_classes == []:
            result["W/O Ignored Classes"] = self.compute_metrics(self.ignore_classes)

        return result
