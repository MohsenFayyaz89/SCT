from yacs.config import CfgNode as CN

_C = CN()

_C.system = CN()
_C.system.device = "cuda"
_C.system.num_workers = 2
_C.system.seed = 1

_C.dataset = CN()
_C.dataset.root = "/Data/SCT"
_C.dataset.name = "breakfast"
_C.dataset.feat_name = "i3d"
_C.dataset.split = 1

_C.experiment = CN()
_C.experiment.name = "default"
_C.experiment.root = "/Data/ActionSet"
_C.experiment.tb_root = "/Data/ActionSet"
_C.experiment.run_number = -1  # -1 is the default
_C.experiment.track_training_metrics_per_iter = False
_C.experiment.log_inference_output = False

_C.training = CN()
_C.training.name = "normal"
_C.training.save_every = 1
_C.training.overfit = False
_C.training.overfit_indices = [0]
_C.training.clip_grad_norm = True
_C.training.clip_grad_norm_value = 10
_C.training.num_epochs = 100
_C.training.optimizer = "SGD"
_C.training.learning_rate = 1e-2
_C.training.momentum = 0.009
_C.training.weight_decay = 0.000
_C.training.random_flip = False
_C.training.random_concat = CN()
_C.training.random_concat.active = False
_C.training.random_concat.num_videos = 1
_C.training.scheduler = CN()
_C.training.scheduler.name = "step"  # can be 'none', 'plateau', 'step'

_C.training.scheduler.multi_step = CN()
_C.training.scheduler.multi_step.steps = [80]

# below are the settings for plateau lr scheduler.
_C.training.scheduler.plateau = CN()
_C.training.scheduler.plateau.mode = "max"
_C.training.scheduler.plateau.factor = 0.9
_C.training.scheduler.plateau.verbose = True
_C.training.scheduler.plateau.patience = 40

_C.training.pretrained = False
_C.training.pretrained_weight = "None"
_C.training.skip_modules = []

_C.training.only_test = False
_C.training.resume = False
_C.training.resume_from = -1  # resume from the latest

_C.training.evaluators = CN()
_C.training.evaluators.eval_train = False
_C.training.evaluators.ignore_classes = [0]

_C.model = CN()
_C.model.fer = CN()
_C.model.fer.name = "wavenet"
_C.model.fer.hidden_size = 128
_C.model.fer.input_size = 2048
_C.model.fer.dropout_on_x = 0.05
_C.model.fer.gn_num_groups = 32
_C.model.fer.wavenet_pooling_levels = [1, 2, 4, 8, 10]
_C.model.fer.output_levels = [4]

_C.model.fs = CN()
_C.model.fs.name = "conv"
_C.model.fs.hidden_size = 128
_C.model.fs.dropout = 0.2
_C.model.fs.set_pred_thr = 0.5
_C.model.fs.length_loss_width = 1.0

_C.model.fc = CN()
_C.model.fc.name = "conv"
_C.model.fc.softmax_temp = 0.1

_C.model.fl = CN()
_C.model.fl.name = "conv"

_C.model.fu = CN()
_C.model.fu.name = "TemporalSampling"


_C.loss = CN()
_C.loss.sct_loss_mul = 1.0
_C.loss.region_loss_mul = 0.1
_C.loss.set_loss_mul = 1.0
_C.loss.length_loss_mul = 0.1
_C.loss.temporal_consistency_loss_mul = 0.1
_C.loss.inv_sparsity_loss_mul = 0.1
_C.loss.inv_sparsity_loss_activation = 1.0
_C.loss.inv_sparsity_loss_type = "L1"
_C.loss.length_loss_width = 1.0
_C.loss.interpolate_before_sct = True
_C.loss.class_weight = [(0, 0.001)]

def get_cfg_defaults():
    return _C.clone()
