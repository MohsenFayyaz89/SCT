from typing import Optional

from torch import Tensor
from yacs.config import CfgNode

from SCT.datasets import GeneralDataset
from SCT.evaluators import GeneralEvaluator
from SCT.experiment.general_experiment import GeneralExperiment
from SCT.models import GeneralModel


def make_experiment(
    cfg: CfgNode,
    dataset: GeneralDataset,
    model: GeneralModel,
    loss_weights: Tensor,
    val_evaluator: Optional[GeneralEvaluator],
    train_evaluator: Optional[GeneralEvaluator],
) -> GeneralExperiment:
    training_name = cfg.training.name
    if training_name == "normal":
        return GeneralExperiment(
            cfg,
            dataset,
            model,
            loss_weights,
            val_evaluator=val_evaluator,
            train_evaluator=train_evaluator,
        )
    else:
        raise Exception("Invalid training name (%s)" % training_name)
