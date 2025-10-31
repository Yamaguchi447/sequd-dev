# 重要度の逆数に基づいてZooming後の比率を補正するSeqUD

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
from sampler.sequd import SeqUD

class DISeqUD(SeqUD):
    """重要度の逆数を正規化し，その比率通りにZooming（シグモイド関数による補正あり）"""
    def __init__(self, param_space, n_runs_per_stage, random_state=None):
        super().__init__(param_space, n_runs_per_stage, random_state=None)
    def calc_parameter_importance(self):
        """
        重要度を計算
        """
        # Anovaの設定
        importance_evaluator = optuna.importance.PedAnovaImportanceEvaluator()

        if self.current_stage >= 3:
            study_df = self.study.trials_dataframe()
            for i, (k, v) in enumerate(self.param_space.items()):
                study_df = study_df[study_df[f"params_{k}"] >= self.lower_bounds[k]]
                study_df = study_df[study_df[f"params_{k}"] <= self.upper_bounds[k]]
            # Studyを作り直して，Zooming後の探索空間に該当する解をStudyに追加
            zooming_study = optuna.create_study()
            zooming_study.add_trials(
                [self.study.get_trials()[INDEX] for INDEX in list(study_df.index)]
            )
            input_study = zooming_study
        else:
            input_study = self.study
        # 重要度を計算
        importance = importance_evaluator.evaluate(input_study)
        # 合計がゼロの場合、前のステージの重要度を使用
        SUM = sum(importance[k] for k in importance.keys())
        if SUM == 0:
            if hasattr(self, 'prev_importance') and self.prev_importance is not None:
                importance = self.prev_importance
            else:
                n_params = len(self.param_space)
                importance = {k: 1.0 / n_params for k in self.param_space.keys()}
        else:
            importance = {k: (1.0 / SUM) * importance[k] for k in importance.keys()}
        # 前のステージの重要度を保存
        self.prev_importance = importance
        return importance

    def convert_scale(self, w, k=1):
        n_params = len(self.param_space)
        # 0.5が基準になるように変換
        base_importance = 1 / n_params
        w *= 0.5 / base_importance
        return 1 / (1 + np.exp(-1 * k * (w - 0.5)))
    
    def _shrink_search_space(self):
        # 最良の試行を取得
        best_trial = self.study.best_trial
        best_params = best_trial.params
        # 重要度を取得
        # importance_evaluator = optuna.importance.FanovaImportanceEvaluator()
        # importance = importance_evaluator.evaluate(self.study)
        self.importance = self.calc_parameter_importance()
        # 重要度の逆数を計算
        inverse_importance = {
            k: 1.0 / v if v > 0 else float("inf") for k, v in self.importance.items()
        }
        inverse_importance_sum = sum(inverse_importance.values())
        new_lower_bounds = {}
        new_upper_bounds = {}
        self.coef_dict = {}
        for k in self.param_space.keys():
            # 重要度の逆数を正規化した値をシグモイド関数でスケーリング
            scaled_inverse_importance = self.convert_scale(
                inverse_importance[k] / inverse_importance_sum
            )
            self.coef_dict[k] = scaled_inverse_importance
            range_span = (
                self.upper_bounds[k] - self.lower_bounds[k]
            ) * scaled_inverse_importance
            new_lower_bounds[k] = max(
                self.lower_bounds[k], best_params[k] - range_span * 0.5
            )
            new_upper_bounds[k] = min(
                self.upper_bounds[k], best_params[k] + range_span * 0.5
            )
        self.lower_bounds = new_lower_bounds
        self.upper_bounds = new_upper_bounds
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
        new_points = np.array(new_points)
        return new_points
    
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
        if self.current_stage >= 2:
            trial.system_attrs["importance"] = [
                self.importance[k] for k in self.param_space.keys()
            ]
            trial.system_attrs["importance_correction"] = [
                self.coef_dict[k] for k in self.param_space.keys()
            ]
        return {}