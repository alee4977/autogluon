from __future__ import annotations

import logging
from typing import List

from functools import partial

import numpy as np

from phem.application_utils.supported_metrics import msc
from phem.base_utils.metrics import make_metric
from sklearn.metrics import mean_squared_error, log_loss

from phem.methods.ensemble_weighting import CMAES
from phem.methods.ensemble_selection import EnsembleSelection as PHEMEnsembleSelection
from ...core.models.greedy_ensemble.ensemble_selection import EnsembleSelection as AutoGluonEnsembleSelection

from phem.framework.abstract_numerical_solvers import evaluate_single_solution

from phem.methods.ensemble_selection.qdo import (
    QDOEnsembleSelection,
    get_bs_ensemble_size_and_loss_correlation,
)
from phem.examples.simulate_with_existing_data_example.simulate_based_on_sklearn_data import FakedFittedAndValidatedClassificationBaseModel

logger = logging.getLogger(__name__)

def base_models_creation(predictions: List[np.ndarray]):

    # Create FakedFittedAndValidatedClassificationBaseModel instances
    base_models_data = []
    for i in range(len(predictions)):
        bm_data = FakedFittedAndValidatedClassificationBaseModel(
            name=f"model_{i}",
            val_probabilities=predictions[i],  # Using the same probabilities for validation and test
            test_probabilities=predictions[i]  # Setting test probability because it is a required positional argument for the class
        )
        base_models_data.append(bm_data)
    
    return base_models_data

def score_metric_creation(metric_type: str, labels: np.ndarray):

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
        score_metric = make_metric(
            partial(log_loss, labels=[0,1]),
            metric_name="log_loss",
            maximize=False,
            classification=True,
            always_transform_conf_to_pred=False,
            optimum_value=0,
            requires_confidences=False
        )

    return score_metric

def get_top_N_predictions(predictions, labels, metric, num_of_base_models):

        if (predictions.shape[0] < num_of_base_models):
            return predictions
        else:
            num_models = predictions.shape[0]
            scores = []
            pred = []

            for i in range(num_models):
                score = metric(y_pred=predictions[i], y_true=labels)
                pred.append(predictions[i])
                scores.append(score)
  
            sorted_scores_with_indices = sorted((num, idx) for idx, num in enumerate(scores))

            threshold_value = sorted_scores_with_indices[num_of_base_models - 1][0]

            scores = [num if num <= threshold_value else 0 for num in scores]

            predictions_new = [
                prediction if score != 0 else [0] * len(prediction)
                for score, prediction in zip(scores, predictions)
            ]

            predictions_new = np.array(predictions_new)

            return predictions_new

def create_and_fit_ensemble_greedy_ensemble_selection(predictions, labels, ensemble_size, method_name, base_models, score_metric, problem_type, metric):

    if (method_name == "greedy_ensemble_selection"):
        es = AutoGluonEnsembleSelection(ensemble_size=ensemble_size, problem_type=problem_type,metric=metric)
        es.fit(predictions, labels) 
    elif (method_name == "phem_greedy_ensemble_selection"):
        es = PHEMEnsembleSelection(
            base_models=base_models,
            n_iterations=ensemble_size,
            metric=score_metric,
            random_state=1,
        )
        es.fit(predictions.T, labels)

    return es

def create_and_fit_ensemble_cmaes(predictions, labels, ensemble_size, method_name, base_models, score_metric):

    normalize_weights = "softmax" if (method_name == "cmaes_with_normalization") else "no"
    trim_weights = "ges-like" if (method_name == "cmaes_with_normalization") else "no"

    es = CMAES(
            base_models=base_models,
            n_iterations=ensemble_size,  # Number of iterations determined by self.ensemble_size
            score_metric=score_metric,
            random_state=1,
            normalize_weights=normalize_weights,
            trim_weights=trim_weights,
        )

    es.fit(predictions.T, labels)

    if (method_name == 'single_best'):
        es.weights_ = es.single_best_stats[1]
    
    return es

def create_and_fit_ensemble_qdo(predictions, labels, ensemble_size, method_name, base_models, score_metric):

    archive_type = "quality" if (method_name == "quality_optimization") else "sliding"
    behavior_space = get_bs_ensemble_size_and_loss_correlation() if (method_name == "quality_diversity_optimization") else None

    es = QDOEnsembleSelection(
            base_models=base_models,
            n_iterations=3, # Number of iterations determined by self.ensemble_size
            score_metric=score_metric,
            archive_type=archive_type,
            behavior_space=behavior_space,
            random_state=1,
        )

    es.fit(predictions.T, labels)
    return es