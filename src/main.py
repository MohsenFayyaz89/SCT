import argparse

from SCT.config import get_cfg_defaults
from SCT.datasets import make_db
from SCT.evaluators import make_evaluators, make_evaluator_final
from SCT.experiment import make_experiment
from SCT.models import make_model, make_loss_weights
from SCT.utils import set_seed, change_multiprocess_strategy, print_with_time
from yacs.config import CfgNode


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def create_cfg() -> CfgNode:
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg.merge_from_list(args.set_cfgs)
    cfg.freeze()
    return cfg


def main():
    cfg = create_cfg()
    # set_seed(cfg.system.seed)
    change_multiprocess_strategy()

    train_db = make_db(cfg, train=True)

    if cfg.training.overfit:
        test_db = train_db
    else:
        test_db = make_db(cfg, train=False)
    model = make_model(
        cfg,
        num_classes=train_db.num_classes,
    )
    loss_weights = make_loss_weights(
        num_classes=train_db.num_classes, weights=cfg.loss.class_weight
    )
    train_evaluator, val_evaluator = make_evaluators(cfg, train_db, test_db, model)
    experiment = make_experiment(
        cfg,
        train_db,
        model,
        loss_weights,
        val_evaluator=val_evaluator,
        train_evaluator=train_evaluator,
    )

    if not cfg.training.only_test:
        if cfg.training.pretrained and cfg.training.resume:
            raise ValueError(
                "training.pretrained and training.resume"
                " flags cannot be True at the same time"
            )
        elif cfg.training.pretrained:
            experiment.init_from_pretrain()
        elif cfg.training.resume:
            experiment.resume()
        experiment.train()
    else:
        experiment.load_model_for_test()

    final_evaluator = make_evaluator_final(cfg, test_db, model)
    final_eval_result = final_evaluator.evaluate()

    print_with_time("Final Evaluation Result ...")
    print(final_eval_result)
    if not cfg.training.only_test:
        print_with_time("Saving final model ...")
        experiment.save()


if __name__ == "__main__":
    main()
