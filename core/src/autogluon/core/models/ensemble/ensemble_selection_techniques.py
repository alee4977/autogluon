from __future__ import annotations

import logging
from typing import List


import numpy as np

from ..greedy_ensemble.ensemble_selection import EnsembleSelection

from ....core.utils.phem_methods import ( 
    base_models_creation, 
    score_metric_creation,
    get_top_N_predictions,
    create_and_fit_ensemble_greedy_ensemble_selection,
    create_and_fit_ensemble_cmaes, 
    create_and_fit_ensemble_qdo, 
)

logger = logging.getLogger(__name__)


class EnsembleSelectionMethod(EnsembleSelection):
    def __init__(
        self,
        ensemble_size: int,
        problem_type: str,
        metric,
        sorted_initialization: bool = False,
        bagging: bool = False,
        tie_breaker: str = "random",
        subsample_size: int | None = None,
        random_state: np.random.RandomState = None,
        **kwargs,
    ) -> None:
        
        super().__init__(
            ensemble_size=ensemble_size,
            problem_type=problem_type,
            metric=metric,
            sorted_initialization=sorted_initialization,
            bagging=bagging,
            tie_breaker=tie_breaker,
            subsample_size=subsample_size,
            random_state=random_state,
            **kwargs,
        )
        self.method_name = kwargs.get("ensemble_selection_method")


    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, sample_weight=None):

        predictions = get_top_N_predictions(predictions, labels, self.metric, 50)
        func_args = [predictions, labels, self.ensemble_size, self.method_name]

        base_models_data = base_models_creation(predictions)
        func_args.append(base_models_data)

        score_metric = score_metric_creation(self.metric.name, labels) 
        func_args.append(score_metric)


        if (self.method_name in ["greedy_ensemble_selection", "phem_greedy_ensemble_selection"]):
            es = create_and_fit_ensemble_greedy_ensemble_selection(*func_args, problem_type=self.problem_type,metric=self.metric)
            # es = EnsembleSelection(ensemble_size=self.ensemble_size, problem_type=self.problem_type,metric=self.metric)
            # es.fit(predictions, labels) 
        elif (self.method_name in ["single_best", "cmaes", "cmaes_with_normalization"]):
            es = create_and_fit_ensemble_cmaes(*func_args)
        elif (self.method_name in ["quality_optimization", "quality_diversity_optimization"]):
            es = create_and_fit_ensemble_qdo(*func_args)

        self.weights_ = es.weights_

        return self.weights_

    def _calculate_weights(self):
        pass

    