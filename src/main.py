import os
import numpy as np
import optuna
import time
from omegaconf import OmegaConf

from bbob.run_bbob import eval
from common import setup_config
from sampler.master import suggest_sampler
from sampler.plot_log import plot_number_of_solutions_per_stage


class OptunaObjective(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, trial):
        X = np.array(
            [
                trial.suggest_float(
                    f"x{i}", self.cfg.objective.min_bound, self.cfg.objective.max_bound
                )
                for i in range(self.cfg.objective.dim)
            ]
        )
        objective_value = self.calc_objective_value(X)

        return objective_value

    def calc_objective_value(self, X):
        return eval(
            X,
            self.cfg.objective.func_id,
            self.cfg.default.r_seed,
            self.cfg.objective.X_opt,
            self.cfg.objective.F_opt,
        )


def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    if not os.path.exists(cfg.out_dir + "finish.txt"):
        s = time.time()
        sampler = suggest_sampler(cfg)

        optuna_objective = OptunaObjective(cfg)

        study = optuna.create_study(sampler=sampler, direction=cfg.objective.direction)
        study.optimize(optuna_objective, n_trials=cfg.sampler.n_trials)
        study_df = study.trials_dataframe()

        study_df.to_csv(cfg.out_dir + "results.csv", index=False)

        if "SeqUD" in cfg.sampler.name:
            plot_number_of_solutions_per_stage(cfg)

        with open(cfg.out_dir + "finish.txt", "w") as f:
            f.write(str(time.time() - s))


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)
