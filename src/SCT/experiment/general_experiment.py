import datetime
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional, Union, Dict, Iterable, List

import torch
import torch.optim as optim
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode

from SCT.datasets import GeneralDataset
from SCT.datasets.general_dataset import BatchItem
from SCT.evaluators import GeneralEvaluator
from SCT.evaluators.general_evaluator import EvalResult
from SCT.models import GeneralModel
from SCT.models.data_classes import LossOut
from SCT.models.losses import loss_func
from SCT.utils import get_git_commit_hash, print_with_time, OverfitSampler

Scheduler = Union[ReduceLROnPlateau, MultiStepLR]


RUN_INFO_FORMAT = """Time: {time}
Command: {command}
Git hash: {hash}
-----------------------------------------
{config}
"""


class ScalarMetric:
    def __init__(self, writer: SummaryWriter, name: str, report_average: bool = True):
        self.writer = writer
        self.name = name
        self.report_average = report_average
        self.values = []
        self.average_tag = "training_average/%s" % self.name

    def add_value(self, value: float, step: int, add_to_writer: bool = True):
        if add_to_writer:
            self.writer.add_scalar(tag=self.name, scalar_value=value, global_step=step)
        self.values.append(value)

    def epoch_finished(self, epoch_num):
        average_value = self.average_value()
        if self.report_average:
            self.writer.add_scalar(
                tag=self.average_tag,
                scalar_value=average_value,
                global_step=epoch_num + 1,
            )
            print_with_time("%s: %f" % (self.average_tag, average_value))
        self.reset_values()

    def reset_values(self):
        self.values.clear()

    def average_value(self) -> float:
        return sum(self.values) / len(self.values)


def create_optimizer(cfg: CfgNode, parameters: Iterable[Parameter]) -> Optimizer:
    learning_rate = cfg.training.learning_rate
    momentum = cfg.training.momentum
    optimizer_name = cfg.training.optimizer
    weight_decay = cfg.training.weight_decay

    if optimizer_name == "SGD":
        return optim.SGD(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer_name == "Adam":
        return optim.Adam(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise Exception("Invalid optimizer name (%s)" % optimizer_name)


def create_scheduler(cfg: CfgNode, optimizer: Optimizer) -> Optional[Scheduler]:
    scheduler_name = cfg.training.scheduler.name
    if scheduler_name == "none":
        return None
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.scheduler.plateau.mode,
            factor=cfg.training.scheduler.plateau.factor,
            verbose=cfg.training.scheduler.plateau.verbose,
            patience=cfg.training.scheduler.plateau.patience,
        )
    elif scheduler_name == "step":
        steps = cfg.training.scheduler.multi_step.steps
        return MultiStepLR(optimizer=optimizer, milestones=steps)
    else:
        raise Exception("Invalid scheduler name (%s)" % scheduler_name)


def create_metrics(cfg: CfgNode, writer: SummaryWriter) -> Dict[str, ScalarMetric]:
    metric_names_with_average = GeneralExperiment.metric_names_with_average
    metric_names_each_epoch = GeneralExperiment.metric_names_each_epoch_testing

    if cfg.training.evaluators.eval_train:
        metric_names_each_epoch.extend(
            GeneralExperiment.metric_names_each_epoch_training
        )

    metrics = {}
    for mn in metric_names_with_average:
        metrics[mn] = ScalarMetric(writer, mn, report_average=True)

    for mn in metric_names_each_epoch:
        metrics[mn] = ScalarMetric(writer, mn, report_average=False)

    return metrics


def create_train_dataloader(cfg: CfgNode, dataset: GeneralDataset) -> DataLoader:
    if cfg.training.overfit:
        sampler = OverfitSampler(
            main_source=dataset, indices=cfg.training.overfit_indices
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
            shuffle=True,
            num_workers=cfg.system.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )


class GeneralExperiment(object):
    metric_names_with_average = [
        "loss/Total",
        "loss/Set",
        "loss/Region",
        "loss/SCT",
        "loss/Temporal_Consistency",
        "loss/Length",
        "loss/Inverse_Sparsity",
        "optimization/learning_rate",
    ]
    metric_names_each_epoch_testing = [
        "average_testing/mof",
    ]
    metric_names_each_epoch_training = [
        "average_training/mof",
    ]

    model_filename = "model.pkl"
    optimizer_filename = "optimizer.pkl"
    scheduler_filename = "scheduler.pkl"

    def __init__(
        self,
        cfg: CfgNode,
        dataset: GeneralDataset,
        model: GeneralModel,
        loss_weights: torch.Tensor,
        val_evaluator: Optional[GeneralEvaluator],
        train_evaluator: Optional[GeneralEvaluator],
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.loss_weights = loss_weights
        self.val_evaluator = val_evaluator
        self.train_evaluator = train_evaluator

        self.device = torch.device(self.cfg.system.device)

        if not cfg.training.evaluators.ignore_classes == []:
            self.metric_names_each_epoch_training.extend(
                [
                    "average_training/mof_w/o_ignored_cls",
                ]
            )
            self.metric_names_each_epoch_testing.extend(
                [
                    "average_testing/mof_w/o_ignored_cls",
                ]
            )

        self.experiment_name = (
            Path(self.cfg.experiment.name)
            / self.cfg.dataset.name
            / str(self.cfg.dataset.split)
        )

        self.experiment_folder = Path(self.cfg.experiment.root) / self.experiment_name
        self.experiment_folder.mkdir(exist_ok=True, parents=True)
        self.iter_number = 0
        self.run_number = self.cfg.experiment.run_number
        if self.cfg.experiment.run_number == -1:
            self.run_number = self._figure_run_number()

        self.run_folder = self.experiment_folder / str(self.run_number)

        self.tb_folder = (
            Path(self.cfg.experiment.tb_root)
            / self.experiment_name
            / Path(str(self.run_number))
        )
        self.tb_writer = SummaryWriter(self.tb_folder)

        self.metrics = create_metrics(self.cfg, self.tb_writer)

        self.clip_grad_norm = self.cfg.training.clip_grad_norm
        self.clip_grad_norm_value = self.cfg.training.clip_grad_norm_value

        self.epoch_number = 0
        if self.cfg.training.only_test and self.cfg.training.resume_from == -1:
            self.epoch_number = self._figure_epoch_number()
        elif self.cfg.training.only_test and not self.cfg.training.resume_from == -1:
            self.epoch_number = self.cfg.training.resume_from
        self.iter_num = 0
        self.epoch_losses = []

        self.optimizer = create_optimizer(self.cfg, self.model.parameters())
        self.scheduler = create_scheduler(self.cfg, self.optimizer)
        self.scheduler_type_plateau = (
            True if self.cfg.training.scheduler.name == "plateau" else False
        )
        self.loss = loss_func

    def _figure_run_number(self) -> int:
        # fixme: this is not thread safe!
        max_run = 0
        for f in self.experiment_folder.iterdir():
            if f.is_dir():
                try:
                    f = int(str(f.name))
                except ValueError:
                    continue
                if f > max_run:
                    max_run = f

        if self.cfg.training.only_test:
            return max_run
        return max_run + 1

    def _figure_epoch_number(self) -> int:
        max_epoch = 0
        for f in self.run_folder.iterdir():
            if f.is_dir():
                try:
                    f = int(str(f.name))
                except ValueError:
                    continue
                if f > max_epoch:
                    max_epoch = f

        if self.cfg.training.only_test:
            return max_epoch
        return max_epoch

    def generate_run_info(self) -> str:
        config_dump = self.cfg.dump()

        return RUN_INFO_FORMAT.format(
            time=str(datetime.datetime.now()),
            command=" ".join(sys.argv),
            hash=get_git_commit_hash(),
            config=config_dump,
        )

    def _mark_the_run(self):
        self.run_folder.mkdir(exist_ok=True, parents=True)
        run_info = self.generate_run_info()
        with open(self.run_folder / "info.txt", "w") as f:
            f.write(run_info)

    # noinspection PyUnusedLocal
    def train_1_batch(self, iter_number: int, batch: BatchItem) -> LossOut:
        self.optimizer.zero_grad()
        batch.to(self.device)
        prediction = self.model.forward(batch)
        loss = self.loss(
            A=prediction.A,
            L=prediction.L,
            V=prediction.V,
            A_hat=batch.A_hat,
            cfg=self.cfg.loss,
            weight=self.loss_weights,
        )
        loss.total_loss.backward()
        if self.clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm_value)
            # clip_grad_norm_(self.model.fl.parameters(), max_norm=self.clip_grad_norm_value)
        self.optimizer.step()
        return loss

    def train_1_epoch(self, epoch_number: int, dataloader: DataLoader):
        print_with_time("Training epoch %d ...." % (epoch_number + 1))
        self.model.train()
        for batch in tqdm(dataloader):
            self.on_start_batch(self.iter_num)

            batch_loss = self.train_1_batch(self.iter_num, batch)
            self.track_training_metrics(batch, batch_loss, self.iter_num)

            self.on_end_batch(self.iter_num)
            self.iter_num += 1

        for n in self.metric_names_with_average:
            self.metrics[n].epoch_finished(epoch_number)

    def train(self):
        self._mark_the_run()
        num_epochs = self.cfg.training.num_epochs
        print_with_time(self.cfg.dump())
        print_with_time("Training for run number: {:d}".format(self.run_number))
        epoch_range = range(0, num_epochs)
        train_dataloader = create_train_dataloader(self.cfg, self.dataset)
        self.model.to(self.device)
        self.loss_weights = self.loss_weights.to(self.device)

        self.on_start_training()
        for epoch_num in epoch_range:
            self.epoch_number = epoch_num

            # resetting metrics
            for n, m in self.metrics.items():
                m.reset_values()

            # callback
            self.on_start_epoch(epoch_num)

            # train for 1 epoch
            # with torch.autograd.set_detect_anomaly(True):
            self.train_1_epoch(epoch_num, train_dataloader)

            # save
            if (epoch_num + 1) % self.cfg.training.save_every == 0:
                self.save()

            # end of epoch evaluations
            if self.train_evaluator is not None:
                train_eval_result = self.train_evaluator.evaluate()
                print_with_time("Evaluation result on train set ...")
                print(train_eval_result)
                self.update_epoch_metrics_train_eval(train_eval_result, epoch_num)
            val_eval_result = self.val_evaluator.evaluate()
            print_with_time("Evaluation result on test set ...")
            print(val_eval_result)
            self.update_epoch_metrics_val_eval(val_eval_result, epoch_num)

            if self.scheduler is not None:
                # plateau scheduler
                if self.scheduler_type_plateau:
                    self.scheduler.step(
                        metrics=self._prepare_plateau_scheduler_input(val_eval_result),
                        epoch=epoch_num,
                    )
                # step scheduler
                else:
                    self.scheduler.step()

            # callback
            self.on_end_epoch(epoch_num)

    @staticmethod
    def _prepare_plateau_scheduler_input(eval_result: EvalResult) -> float:
        return eval_result["All"].mof

    def current_lr(self):
        if self.scheduler is not None:
            return list(self.scheduler.optimizer.param_groups)[0]["lr"]
        else:
            return self.optimizer.defaults["lr"]

    def on_start_epoch(self, epoch_num: int):

        print_with_time("Epoch {}, LR: {}".format(epoch_num + 1, self.current_lr()))

    def on_end_epoch(self, epoch_num: int):
        pass

    def on_start_batch(self, iter_num: int):
        pass

    def on_end_batch(self, iter_num: int):
        pass

    def on_start_training(self):
        pass

    def on_end_training(self):
        pass

    def init_from_pretrain(self):
        print_with_time("Initializing from pretrained weights...")
        model_file = self.cfg.training.pretrained_weight
        self.load_model(model_file, self.cfg.training.skip_modules)

    def resume(self):
        print_with_time("Resuming the experiment...")
        # TODO
        raise NotImplementedError("I am lazy!")

    def load_model_for_test(self):
        epoch_folder = self.run_folder / str(self.epoch_number)
        model_file = epoch_folder / self.model_filename
        self.load_model(model_file)

    def load_model(self, model_file: str, skip_modules: List[str] = []):
        print_with_time("Loading Model: {}".format(model_file))
        input_model_dict = torch.load(model_file, map_location=torch.device("cpu"))
        filtered_model_dict = OrderedDict()
        for key, val in input_model_dict.items():
            if key.split(".")[0] not in skip_modules:
                filtered_model_dict[key] = val
            else:
                print("Skipping: {}".format(key))

        self.model.load_state_dict(filtered_model_dict, strict=False)

    def load_optimizer(self):
        epoch_folder = self.run_folder / str(self.epoch_number)
        optimizer_file = epoch_folder / self.model_filename
        print_with_time("Loading Optimizer: {}".format(optimizer_file))
        self.model.load_state_dict(torch.load(optimizer_file))

    def load_scheduler(self):
        epoch_folder = self.run_folder / str(self.epoch_number)
        scheduler_file = epoch_folder / self.model_filename
        print_with_time("Loading Scheduler: {}".format(scheduler_file))
        self.model.load_state_dict(torch.load(scheduler_file))

    def save(self):
        epoch_folder = self.run_folder / str(self.epoch_number + 1)
        epoch_folder.mkdir(exist_ok=True, parents=True)

        model_file = epoch_folder / self.model_filename
        optimizer_file = epoch_folder / self.optimizer_filename
        scheduler_file = epoch_folder / self.scheduler_filename

        print_with_time("Saving model ...")
        torch.save(self.model.cpu().state_dict(), model_file)
        print_with_time("Saving Optimizer ...")
        torch.save(self.optimizer.state_dict(), optimizer_file)
        if self.scheduler is not None:
            print_with_time("Saving Scheduler ...")
            torch.save(self.scheduler, scheduler_file)

    def update_epoch_metrics_train_eval(
        self, train_eval_result: EvalResult, epoch_num: int
    ):
        names = self.metric_names_each_epoch_training
        values = [
            train_eval_result["All"].mof,
        ]
        if not self.cfg.training.evaluators.ignore_classes == []:
            values.extend(
                [
                    train_eval_result["W/O Ignored Classes"].mof,
                ]
            )

        for n, v in zip(names, values):
            self.metrics[n].add_value(v, step=epoch_num + 1)

    def update_epoch_metrics_val_eval(
        self, val_eval_result: EvalResult, epoch_num: int
    ):
        names = self.metric_names_each_epoch_testing
        values = [
            val_eval_result["All"].mof,
        ]
        if not self.cfg.training.evaluators.ignore_classes == []:
            values.extend(
                [
                    val_eval_result["W/O Ignored Classes"].mof,
                ]
            )

        for n, v in zip(names, values):
            self.metrics[n].add_value(v, step=epoch_num + 1)

    # noinspection PyUnusedLocal
    def track_training_metrics(
        self, batch: BatchItem, batch_loss: LossOut, iter_num: int
    ):
        metric_names = self.metric_names_with_average
        values = [
            batch_loss.total_loss.item(),
            batch_loss.set_loss.item(),
            batch_loss.region_loss.item(),
            batch_loss.sct_loss.item(),
            batch_loss.temporal_consistency_loss.item(),
            batch_loss.length_loss.item(),
            batch_loss.inv_sparsity_loss.item(),
            self.current_lr(),
        ]
        add_to_writer = self.cfg.experiment.track_training_metrics_per_iter
        for n, v in zip(metric_names, values):
            self.metrics[n].add_value(v, step=iter_num, add_to_writer=add_to_writer)
