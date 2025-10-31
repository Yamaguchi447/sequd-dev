import numpy as np
import optuna

from sampler.sequd import SeqUD
from sampler.dvesequd import DVESeqUD
from sampler.disequd import DISeqUD
from sampler.divesequd import DIVESeqUD

def suggest_sampler(cfg):
    param_space = {
        f"x{i}": {
            "Type": "continuous",
            "Range": [cfg.objective.min_bound, cfg.objective.max_bound],
        }
        for i in range(cfg.objective.dim)
    }
    if cfg.sampler.name == "SeqUD": # オリジナルSeqUD
        sampler = SeqUD(
            param_space,
            n_runs_per_stage=cfg.sampler.pop_size,
            random_state=cfg.default.r_seed,
        )
    elif cfg.sampler.name == "DVESeqUD": # 次元によらず体積の収縮度合いを一定にしたSeqUD
        sampler = DVESeqUD(
            param_space,
            n_runs_per_stage=cfg.sampler.pop_size,
            random_state=cfg.default.r_seed,
        )
    elif cfg.sampler.name == "DISeqUD": # 重要度の逆数に基づいてZooming後の探索空間の比率を補正するSeqUD
        sampler = DISeqUD(
            param_space,
            n_runs_per_stage=cfg.sampler.pop_size,
            random_state=cfg.default.r_seed,
        )
    elif cfg.sampler.name == "DIVESeqUD": # 重要度の逆数に基づいてZooming比率補正し，かつ体積の収縮度合いを一定にしたSeqUD
        sampler = DIVESeqUD(
            param_space,
            n_runs_per_stage=cfg.sampler.pop_size,
            random_state=cfg.default.r_seed,
        )

    elif cfg.sampler.name == "CMAES":
        sampler = optuna.samplers.CmaEsSampler(
            popsize=cfg.sampler.pop_size, sigma0=0.5, seed=cfg.default.r_seed
        )
        

    return sampler
