from __future__ import annotations

import logging
from typing import List

import numpy as np

from phem.application_utils.supported_metrics import msc
from .ensemble_selection import EnsembleSelection
from phem.methods.ensemble_weighting import CMAES
from phem.examples.simulate_with_existing_data_example.simulate_based_on_sklearn_data import FakedFittedAndValidatedClassificationBaseModel, SimulationData

logger = logging.getLogger(__name__)


class CMAESSimulate(EnsembleSelection):

    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, sample_weight=None):

        # Create FakedFittedAndValidatedClassificationBaseModel instances
        base_models_data = []
        for i in range(predictions.shape[0]):
            bm_data = FakedFittedAndValidatedClassificationBaseModel(
                name=f"model_{i}",
                val_probabilities=predictions[i],  # Using the same probabilities for validation and test
                test_probabilities=predictions[i]
            )
            base_models_data.append(bm_data)

        # Create a SimulationData instance (only test data is actually used)
        simulation_data = SimulationData(
            X_train=None,  # Not used in this context
            y_train=None,  # Not used in this context
            X_val=None,    # Not used in this context
            y_val=labels,  # Validation labels
            X_test=None,   # Not used in this context
            y_test=labels, # Test labels
            base_models_data=base_models_data
        )

        # CMAES 
        if (self.metric.name == "root_mean_squared_error"):
            cmaes_metric=msc(metric_name="rmse", is_binary=False, labels=list(range(2)))
        elif (self.metric.name == "roc_auc"):
            cmaes_metric=msc(metric_name="roc_auc", is_binary=True, labels=list(range(2)))
        elif (self.metric.name == "log_loss"):
            cmaes_metric=msc(metric_name="log_loss", is_binary=False, labels=list(range(2)))

        cmaes = CMAES(
            base_models=base_models_data,
            n_iterations=self.ensemble_size,  # Number of iterations for CMAES algorithm
            score_metric=cmaes_metric,
            random_state=1,
        )


        # Switch to simulating predictions on validation data
        for bm in cmaes.base_models:
            bm.switch_to_val_simulation()

        # Fit the ensemble on the validation data
        cmaes.fit(predictions.T, labels)

        # Return ensemble weights
        self.weights_ = cmaes.opt_best_stats[1]

    def _calculate_weights(self):
        pass