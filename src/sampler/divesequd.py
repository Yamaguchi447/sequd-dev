# 重要度の逆数に基づいてZooming後の比率を補正し，かつ体積の収縮度合いを一定にするSeqUD

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


class DIVESeqUD(SeqUD):
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
            # 前のステージの重要度を使用
            if hasattr(self, 'prev_importance') and self.prev_importance is not None:
                importance = self.prev_importance
            else:
                n_params = len(self.param_space)
                importance = {k: 1.0 / n_params for k in self.param_space.keys()}
        else:
            # 合計がゼロでない場合、重要度を正規化
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
        self.importance = self.calc_parameter_importance()
        # 重要度の逆数を計算
        inverse_importance = {
            k: 1.0 / v if v > 0 else float("inf") for k, v in self.importance.items()
        }
        inverse_importance_sum = sum(inverse_importance.values())

        # 各次元のスケール係数を計算（シグモイド関数による補正あり）
        self.coef_dict = {}
        scaled_inverse_importance = {}
        for k in self.param_space.keys():
            scaled_inverse_importance[k] = self.convert_scale(
                inverse_importance[k] / inverse_importance_sum
            )
            self.coef_dict[k] = scaled_inverse_importance[k]


        # 元の探索空間の体積を計算
        original_volume = np.prod([
            self.upper_bounds[k] - self.lower_bounds[k] for k in self.param_space.keys()
        ])

        # 目標体積（前のステージの25%）
        target_volume = original_volume * 0.25


        # 各パラメータにおける収縮率を計算
        scaling_factors = {}
        adjusted_volume = 1.0
        for k in self.param_space.keys():
            scaling_factors[k] = scaled_inverse_importance[k]
            adjusted_volume *= (self.upper_bounds[k] - self.lower_bounds[k]) * scaling_factors[k]

        # 収縮した体積が目標体積に一致するように調整
        scaling_ratio = (target_volume / adjusted_volume) ** (1 / len(self.param_space))

        # スケーリング係数を再計算
        for k in scaling_factors.keys():
            scaling_factors[k] *= scaling_ratio

        # 新しい探索空間の下限と上限を計算
        new_lower_bounds = {}
        new_upper_bounds = {}
        for k in self.param_space.keys():
            range_span = (self.upper_bounds[k] - self.lower_bounds[k]) * scaling_factors[k]
            # print(scaling_factors)
            new_lower_bounds[k] = max(self.lower_bounds[k], best_params[k] - range_span / 2)
            new_upper_bounds[k] = min(self.upper_bounds[k], best_params[k] + range_span / 2)

        # 新しい探索空間の下限と上限を設定
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

        # スケーリングを反映させた追加設計点を生成
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