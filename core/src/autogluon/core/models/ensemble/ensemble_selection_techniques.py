from __future__ import annotations

import logging
from typing import List

from functools import partial

import numpy as np

from phem.application_utils.supported_metrics import msc
from phem.base_utils.metrics import make_metric
from sklearn.metrics import mean_squared_error

from ..greedy_ensemble.ensemble_selection import EnsembleSelection
from phem.methods.ensemble_weighting import CMAES
from phem.methods.ensemble_selection.qdo import (
    QDOEnsembleSelection,
    get_bs_ensemble_size_and_loss_correlation,
)
from phem.examples.simulate_with_existing_data_example.simulate_based_on_sklearn_data import FakedFittedAndValidatedClassificationBaseModel

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

        base_models_data = base_models_creation(predictions)

        score_metric = score_metric_creation(self.metric.name)

        if (self.method_name == "greedy ensemble selection"):
            es = EnsembleSelection(
                ensemble_size=self.ensemble_size,
                problem_type=self.problem_type,
                metric=self.metric
            )
        elif (self.method_name == "cmaes"):
            es = CMAES(
                base_models=base_models_data,
                n_iterations=self.ensemble_size,  # Number of iterations determined by self.ensemble_size
                score_metric=score_metric,
                random_state=1,
            )
        elif (self.method_name == "cmaes with normalization"):
            es = CMAES(
                base_models=base_models_data,
                n_iterations=self.ensemble_size, # Number of iterations determined by self.ensemble_size
                score_metric=score_metric,
                random_state=1,
                normalize_weights="softmax",
                trim_weights="ges-like",
            )
        elif (self.method_name == "quality optimization"):
            es = QDOEnsembleSelection(
                base_models=base_models_data,
                n_iterations=self.ensemble_size, # Number of iterations determined by self.ensemble_size
                archive_type="quality",
                score_metric=score_metric,
                random_state=1,
            )
        elif (self.method_name == "quality diversity optimization"):
            es = QDOEnsembleSelection(
                base_models=base_models_data,
                n_iterations=self.ensemble_size, # Number of iterations determined by self.ensemble_size
                score_metric=score_metric,
                behavior_space=get_bs_ensemble_size_and_loss_correlation(),
                random_state=1,
            )

        # Fit the ensemble on the validation data
        if (self.method_name == "greedy ensemble selection"):
            es.fit(predictions, labels)
        else:
            es.fit(predictions.T, labels)


        # Return ensemble weights
        if ("cmaes" in self.method_name):    
            self.weights_ = es.opt_best_stats[1]
        else:
            self.weights_ = es.weights_

        return self.weights_

    def _calculate_weights(self):
        pass

def base_models_creation(predictions: List[np.ndarray]):

    # Create FakedFittedAndValidatedClassificationBaseModel instances
    base_models_data = []
    for i in range(predictions.shape[0]):
        bm_data = FakedFittedAndValidatedClassificationBaseModel(
            name=f"model_{i}",
            val_probabilities=predictions[i],  # Using the same probabilities for validation and test
            test_probabilities=predictions[i]  # Setting test probability because it is a required positional argument for the class
        )
        base_models_data.append(bm_data)
    
    return base_models_data
    
def score_metric_creation(metric_type: str):
    if (metric_type == "root_mean_squared_error"):
        score_metric = make_metric(
            partial(mean_squared_error, squared=False),
            metric_name="rmse",
            maximize=False,
            classification=False,
            always_transform_conf_to_pred=False,
            optimum_value=0,
            requires_confidences=False
        )
    elif (metric_type == "roc_auc"):
        score_metric = msc(metric_name="roc_auc", is_binary=True, labels=[0, 1])
    elif (metric_type == "log_loss"):
        score_metric = msc(metric_name="log_loss", is_binary=True, labels=[0, 1]) # Multiclass but becomes binary

    return score_metric