from typing import Optional, Tuple

from .general_evaluator import GeneralEvaluator


def make_evaluators(
    cfg, train_db, test_db, model
) -> Tuple[Optional[GeneralEvaluator], Optional[GeneralEvaluator]]:

    train_evaluator = make_train_evaluator(cfg, train_db, model)
    val_evaluator = make_val_evaluator(cfg, test_db, model)

    return train_evaluator, val_evaluator


def make_train_evaluator(cfg, train_db, model) -> Optional[GeneralEvaluator]:
    if cfg.training.evaluators.eval_train:
        return GeneralEvaluator(cfg=cfg, model=model, dataset=train_db)
    else:
        return None


def make_val_evaluator(cfg, val_db, model) -> Optional[GeneralEvaluator]:
    return GeneralEvaluator(cfg=cfg, model=model, dataset=val_db)


def make_evaluator_final(cfg, test_db, model) -> GeneralEvaluator:
    return GeneralEvaluator(cfg=cfg, model=model, dataset=test_db)
