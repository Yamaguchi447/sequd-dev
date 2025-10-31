# 収縮率を次元数によらず一定にしたSeqUD

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
import optuna
import pyunidoe as pydoe
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from sklearn.neighbors import KDTree


class DVESeqUD(BaseSampler):

    def __init__(self, param_space, n_runs_per_stage, random_state=None):
        super().__init__()
        np.random.seed(random_state)
        self.param_space = param_space
        self.n_runs_per_stage = n_runs_per_stage
        self.random_state = random_state
        self.current_stage = 0
        self.lower_bounds = {k: v["Range"][0] for k, v in param_space.items()}
        self.upper_bounds = {k: v["Range"][1] for k, v in param_space.items()}
        self.study = None
        self.current_design = self._generate_initial_design()
        self.pop_id = 0
        self.threshold = 1e-8
        # self.threshold = 1e-8

    def _generate_initial_design(self):
        n_params = len(self.param_space)

        stat = pydoe.gen_ud(
            n=self.n_runs_per_stage,
            s=n_params,
            q=self.n_runs_per_stage,
            init="rand",
            crit="CD2",
            maxiter=100,
            vis=False,
        )
        base_ud = stat["final_design"]

        if base_ud.shape[0] != self.n_runs_per_stage:
            raise ValueError(
                f"Unexpected base_ud size: {base_ud.shape[0]} rows, expected {self.n_runs_per_stage}"
            )

        ud_space_scaled = np.zeros((self.n_runs_per_stage, n_params))
        for i, (k, v) in enumerate(self.param_space.items()):
            ud_space = np.linspace(
                self.lower_bounds[k], self.upper_bounds[k], self.n_runs_per_stage
            )
            design_column = np.clip(base_ud[:, i], 0, self.n_runs_per_stage - 1).astype(
                int
            )
            ud_space_scaled[:, i] = ud_space[design_column]

        return ud_space_scaled

    def _shrink_search_space(self):
        best_trial = self.study.best_trial
        best_params = best_trial.params

        # 次元数に基づいて収縮率を計算
        n_params = len(self.param_space)
        shrink_ratio = 0.25 ** (1 / n_params)  # 各次元の収縮率（今は0.25倍にしているので，適当に変えてみてください）

        new_lower_bounds = {}
        new_upper_bounds = {}
        for k in self.param_space.keys():
            range_span = (self.upper_bounds[k] - self.lower_bounds[k]) * shrink_ratio
            new_lower_bounds[k] = max(
                self.lower_bounds[k], best_params[k] - range_span / 2
            )
            new_upper_bounds[k] = min(
                self.upper_bounds[k], best_params[k] + range_span / 2
            )
        self.lower_bounds = new_lower_bounds
        self.upper_bounds = new_upper_bounds

        stat = pydoe.gen_ud(
            n=self.n_runs_per_stage,
            s=n_params,
            q=self.n_runs_per_stage,
            init="rand",
            crit="CD2",
            maxiter=100,
            vis=False,
        )
        additional_design = stat["final_design"]

        if additional_design.shape[0] != self.n_runs_per_stage:
            raise ValueError(
                f"Unexpected additional_design size: {additional_design.shape[0]} rows"
                + f"expected {self.n_runs_per_stage}"
            )

        additional_design_scaled = np.zeros((self.n_runs_per_stage, n_params))
        for i, (k, v) in enumerate(self.param_space.items()):
            ud_space = np.linspace(
                self.lower_bounds[k], self.upper_bounds[k], self.n_runs_per_stage
            )
            design_column = np.clip(
                additional_design[:, i], 0, self.n_runs_per_stage - 1
            ).astype(int)
            additional_design_scaled[:, i] = ud_space[design_column]

        existing_points = self.study.trials_dataframe()[:-1]
        existing_points = existing_points[
            [f"params_{k}" for k in self.param_space.keys()]
        ].values

        tree = KDTree(existing_points)
        new_points = []
        for point in additional_design_scaled:
            distances, _ = tree.query([point], k=1)
            if distances[0][0] > self.threshold:
                new_points.append(point)

        if len(new_points) == 0:
            warnings.warn("No valid points found. Keeping at least one point.")
            best_trial = self.study.best_trial
            new_points = [np.array([best_trial.params[k] for k in self.param_space.keys()])]

        return new_points

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if self.study is None:
            self.study = study

        if len(self.current_design) == self.pop_id:
            self.current_stage += 1
            self.pop_id = 0
            self.current_design = self._shrink_search_space()
        # pop_idをgrid_idにして，1増やしておく
        trial.system_attrs["grid_id"] = self.pop_id
        self.pop_id += 1

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        pass

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {param_name: dist for param_name, dist in self.param_space.items()}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # 評価する解がどのステージ（世代）で生成されたかを保存
        trial.system_attrs["stage"] = self.current_stage
        # 評価する点が生成された時の探索空間の範囲（Zoomingがあるから世代毎に変わるから保存）
        trial.system_attrs["bounds"] = [
            [self.lower_bounds[k], self.upper_bounds[k]]
            for k in self.param_space.keys()
        ]
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        grid_id = trial.system_attrs.get("grid_id")
        if grid_id is None:
            return param_distribution.sample()

        if param_name not in self.param_space:
            raise ValueError(
                f"The parameter name, {param_name}, is not found in the given space."
            )

        param_value = self.current_design[grid_id][
            list(self.param_space.keys()).index(param_name)
        ]
        contains = param_distribution._contains(
            param_distribution.to_internal_repr(param_value)
        )

        return param_value