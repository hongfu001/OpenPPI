from dataclasses import dataclass

@dataclass
class Config:
    n: int = 2000
    reps: int = 100
    rho: float = .2
    alpha: float = .05
    seed: int = 20260927
    model: str = 'linear'
    regime: str = 'informative'
    beta0: float = 0.
    beta_h: float = 1.
    signal: float = 1.
    bias: float = 1.5
    common_sd: float = .3
    noise_sd: float = .7
    judges: int = 5
    judge_features: str = 'prob'
    ref_size: int = 4000
    truth_size: int = 262144
    update_every: int = 100
    polarized: float = 0.
    kappa_easy: float = 4.
    kappa_hard: float = .3
    explore: int = 0
    learner: str = 'rf'
    svm_c: float = .3
    svm_epsilon: float = .2
    svm_gamma: str = '0.02'
    mlp_hidden: str = '32,16'
    mlp_alpha: float = 10.
    judge_temp: float = 4.
    asi_tau: float = .5
    asi_f: str = 'soft'
    asi_budget_cap: int = 1
    greedy_steps: int = 100
    update_on: str = 'labels'
    tune_tau: int = 0

    pretrain_size: int = 200


def add_config_arguments(parser):
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument('--' + name.replace('_', '-'),
                            type=type(field.default), default=field.default)
    return parser


def config_from_args(args):
    return Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
