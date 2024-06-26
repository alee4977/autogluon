from __future__ import annotations

import logging
import time
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

from ...constants import PROBLEM_TYPES
from ...metrics import log_loss
from ...utils import compute_weighted_metric, get_pred_from_proba

# The following import statements are specifically for the CMAESWeights class 
import cma
from functools import partial
from math import isclose
# End of the import statements 


logger = logging.getLogger(__name__)


class AbstractWeightedEnsemble:
    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)

    def predict_proba(self, X):
        return self.weight_pred_probas(X, weights=self.weights_)

    @staticmethod
    def weight_pred_probas(pred_probas, weights):
        """
        This method takes predicted probabilities from multiple models and a list of weights corresponding to those models, 
        and then calculates a weighted ensemble prediction. 

        Parameters
        ----------
        pred_probas: np.ndarray
            Array of prediction probabilities from multiple models
        weights: np.ndarray
            Weightages corresponding to different models 

        Returns
        -------
        preds_ensemble: np.ndarray
            Weighted ensemble prediction given the passed in predictions probabilities and weights
        """

        preds_norm = [pred * weight for pred, weight in zip(pred_probas, weights)]
        preds_ensemble = np.sum(preds_norm, axis=0)
        return preds_ensemble
    
    @staticmethod
    # This method includes the code to normalize the predictions probabilities in addition
    # to calculating the weight. This is the main difference between this method and the 
    # weight_pred_probas() method
    def _ensemble_predict(
        predictions: list[np.ndarray], weights: np.ndarray, normalize_predict_proba: bool = False
    ) -> np.ndarray:
        """Blanket (not-the-most-efficient) ensemble predict for a weighted ensemble.

        Parameters
        ----------
        weights: np.ndarray
            Can be of any numeric range (but some metric might require normalization).
        normalize_predict_proba: bool, default=False
            If True, normalize the prediction probabilities such that they sum up to 1 and are in [0,1].
            Only needed if the weights are not in [0,1] or do not sum up to 1.
            We apply the softmax but only if negative weights are present.
            We apply simple normalization if weights are positive but do not sum to 1.
        """
        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        for pred, weight in zip(predictions, weights, strict=False):
            np.multiply(pred, weight, out=tmp_predictions)
            np.add(average, tmp_predictions, out=average)

        if normalize_predict_proba:
            if any(weights < 0):
                exp = np.nan_to_num(
                    np.exp(np.clip(average, -88.72, 88.72))
                )  # Clip to avoid overflow
                average = exp / exp.sum(axis=1)[:, None]
                average = average / average.sum(axis=1)[:, None]
            elif not isclose(weights.sum(), 1):
                average = average / average.sum(axis=1)[:, None]

        return average


class CMAESWeights(AbstractWeightedEnsemble):
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
    ):
        self.ensemble_size = ensemble_size
        self.problem_type = problem_type
        self.metric = metric
        self.sorted_initialization = sorted_initialization
        self.bagging = bagging
        self.use_best = True
        if tie_breaker not in ["random", "second_metric"]:
            raise ValueError(f"Unknown tie_breaker value: {tie_breaker}. Must be one of: ['random', 'second_metric']")
        self.tie_breaker = tie_breaker
        self.subsample_size = subsample_size
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(seed=0)
        self.quantile_levels = kwargs.get("quantile_levels", None)

    def fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, identifiers=None, sample_weight=None):
        """
        This method ensures the ensemble size and problem types are valid, fits the ensemble model using the 
        provided probability predictions and ground truth labels, logs the ensemble weights, and returns the 
        instance for method chaining

        Parameters
        ----------
        predictions: np.ndarray
            Prediction probabilities from multiple models
        labels: np.ndarray
            Ground truth labels 
        time_limit: float, default = None
            If specified, will place a time limit on the fit method
        identifiers: default = None

        sample_weight: default = None

        Returns
        -------
        self: CMAESWeights object
            
        """
        self.ensemble_size = int(self.ensemble_size)

        # Validation step to ensure that the ensemble size is at least 1
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        
        # Validation step to ensure that the problem type is recognized
        if not self.problem_type in PROBLEM_TYPES:
            raise ValueError("Unknown problem type %s." % self.problem_type)
        # if not isinstance(self.metric, Scorer):
        #     raise ValueError('Metric must be of type scorer')

        # this method is responsible for the actual fitting process of the ensemble model
        self._fit(predictions=predictions, labels=labels, time_limit=time_limit, sample_weight=sample_weight)

        # This method is responsible for calculating the different weights for the models in the ensemble
        #self._calculate_weights()

        # This logs the ensemble weights
        logger.log(15, "Ensemble weights: ")
        logger.log(15, self.weights_)

        # Returning self is useful for method chaining, which is where you can call another method on the returned instance immediately
        return self

    # TODO: Consider having a removal stage, remove each model and see if score is affected, if improves or not effected, remove it.
    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray, time_limit=None, sample_weight=None):
        """
        This method combines model predictions into an ensemble by optimizing the weighted ensemble vector using 
        the CMA-ES optimization algorithm (CMA-ES) to find the best combination of model weights. The initial weight
        vector passed into the CMA-ES algorithm represents the single best model, ie. the single best model has a weight 
        of 1 and all other models are zero. 

        Parameters
        ----------
        predictions: np.ndarray
            Prediction probabilities from multiple models
        labels: np.ndarray
            Ground truth labels
        time_limit: float, default = None
            If specified, will place a time limit on the fit method
        sample_weight: default = None

        sample_weight: default = None
            
        """
        ensemble_size = self.ensemble_size

        # Convert the pd.Series "labels" to a numpy array, which is necessary for continued processing
        if isinstance(labels, pd.Series):
            labels = labels.values

        # This variable keeps track of how many input models are used
        # For the context in run_quickstart.py, there are 1530 models that are used (ie. configurations)
        self.num_input_models_ = len(predictions)

        # This list keeps track of the models used in the ensemble
        ensemble = []

        # This takes the length of the labels array and assigns it to the number of samples
        # For the run_quickstart.py file the total number of samples is 621
        num_samples_total = len(labels)

        # This block of code basically checks if there is a subsample_size passed in and if it is less than the total number of samples
        # It then randomly selects subsample_size of all the num_samples_total samples in label, and selects those indices from
        # Both the predictions and labels array
        # This speeds up the ensemble selection process because there is less data to apply computations on
        if self.subsample_size is not None and self.subsample_size < num_samples_total:
            logger.log(15, f"Subsampling to {self.subsample_size} samples to speedup ensemble selection...")
            subsample_indices = self.random_state.choice(num_samples_total, self.subsample_size, replace=False)
            labels = labels[subsample_indices]
            for i in range(self.num_input_models_):
                predictions[i] = predictions[i][subsample_indices]

        # This records the time since January 1, 1970, probably set to record the time taken to build the ensemble
        time_start = time.time()
        # This sets a flag round_scores to False initially, that determines if the score should be rounded or not
        round_scores = False
        # This sets the number of digits after the decimal point to round at
        round_decimals = 6

        # score is set to an array of zeros of the length of the number of input models (len(predictions))
        scores = np.zeros((len(predictions)))

        # Setting s to the length of the ensemble
        s = len(ensemble)

        weighted_ensemble_prediction = np.zeros(predictions[0].shape)
        fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            # This for loop iterates through all the predictions, where j is the index and pred is the predictions
        for j, pred in enumerate(predictions):
                # The fant_ensemble_prediction vector simulates the weight vector that would result if the new model was added to the ensemble
                fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1.0 / float(s + 1)) * pred
                # Renormalizes ensuring that the predictions for each sample sum up to one
                if self.problem_type in ["multiclass", "softclass"]:
                    # Renormalize
                    fant_ensemble_prediction[:] = fant_ensemble_prediction / fant_ensemble_prediction.sum(axis=1)[:, np.newaxis]
                # self._calculate_regret calculates the regret between the ground truth labels and predicted labels
                scores[j] = self._calculate_regret(y_true=labels, y_pred_proba=fant_ensemble_prediction, metric=self.metric, sample_weight=sample_weight)
                # if round_scores flag is set the scores are rounded
                if round_scores:
                    scores[j] = scores[j].round(round_decimals)

        # Finds the indicies of the models with the lowest scores and flattens them to a 1D array
        all_best = np.argwhere(scores == np.nanmin(scores)).flatten()

        # Checks if there still is a tie to break
        if len(all_best) > 1:
            # Check if a second metric should be used
            if self.tie_breaker == "second_metric":
                # Checks the problem type
                if self.problem_type in ["binary", "multiclass"]:
                    # Tiebreak with log_loss
                    scores_tiebreak = np.zeros((len(all_best)))
                    secondary_metric = log_loss
                    fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
                    # Maps the scores in all_best to their predictions
                    index_map = {}
                    for k, j in enumerate(all_best):
                        index_map[k] = j
                        pred = predictions[j]
                        fant_ensemble_prediction[:] = weighted_ensemble_prediction + (1.0 / float(s + 1)) * pred
                        # Calculates the regret using the second metric for each prediction in all_best
                        scores_tiebreak[k] = self._calculate_regret(y_true=labels, y_pred_proba=fant_ensemble_prediction, metric=secondary_metric)
                    # Takes the lowest scorers and return their indices in a 1D array
                    all_best_tiebreak = np.argwhere(scores_tiebreak == np.nanmin(scores_tiebreak)).flatten()
                    # Returns the model predictions to all_best
                    all_best = [index_map[index] for index in all_best_tiebreak]

        # Randomly chooses a model from the models with the lowest score after the tie breaker
        best = self.random_state.choice(all_best)

        # The metrics below correspond to the following problem_types
        # 'roc_auc', 'problem_type': 'binary' 
        # 'rmse', 'problem_type': 'regression'
        # 'log_loss', 'problem_type': 'multiclass'

        # The intial weights are set to a sparce vector of zeros where a value of 1 is assigned to the model 
        # with the best prediction after one iteration of GES
        initial_weights = np.zeros(predictions.shape[0]) 
        initial_weights[best] = 1

        sigma0 = 0.2 # The standard deviation is set to 0.2 as recommended in Lennart's paper
        es = self._setup_cma(initial_weights, sigma0)
        val_loss_over_iterations = []

        # Iterations
        for _itr in range(1, ensemble_size + 1):
            # Ask/tell
            solutions = es.ask()
            # Evaluates the candidate solutions 
            es.tell(solutions, self._evaluate_batch_of_solutions(solutions, predictions, labels))

            # Iteration finalization
            val_loss_over_iterations.append(es.result.fbest)

            # This is the logic to enforce the time limit on the method
            if time_limit is not None:
                time_elapsed = time.time() - time_start
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.warning(
                        "Warning: CMAES ran out of time, early stopping at iteration %s. This may mean that the time_limit specified is very small for this problem."
                        % (_itr + 1)
                    )
                    break

        # The best weights vector is assigned to self.weights_
        self.weights_ = es.result.xbest
    
    
    def _setup_cma(self, _start_weight_vector, sigma0=0.5) -> cma.CMAEvolutionStrategy:
        """
        This method initializes the CMA-ES algorithm with the given start weight vector and 
        standard deviation. It returns an instance of cma.CMAEvolutionStrategy configured with these settings.

        Parameters
        ----------
        _start_weight_vector: np.ndarray
            Initial weight vector solution x_0 passed to the CMA-ES algorithm
        sigma0: float, default = 0.5
            Initial standard deviation dictating how far new candidate solutions deviate from the starting weight vector

        Returns
        -------
        self: CMAEvolutionStrategy object
        """
        opts = cma.CMAOptions()
        opts.set("seed", self.random_state.randint(0, 1000000))
        # opts.set("popsize", self.batch_size) # No functionality for that yet
        # Maximum number of iterations, so it matches the number of iterations that would've been used in ensemble selection

        return cma.CMAEvolutionStrategy(_start_weight_vector, sigma0, inopts=opts)
    

    def _evaluate_batch_of_solutions(
        self,
        solutions: np.ndarray,
        predictions,
        labels,
    ) -> np.ndarray:
        """
        This method calculates the loss for a batch of candidate solutions by applying the evaluate_single_solution 
        method to each solution. It returns the calculates losses. 

        Parameters
        ----------
        solutions: np.ndarray
            New candidate solutions proposed by the CMA-ES algorithm
         predictions: np.ndarray
            Prediction probabilities from multiple models
        labels: np.ndarray
            Ground truth labels 

        Returns
        -------
        losses: np.ndarray
        """
        func_args = [predictions, labels, self.metric, True]
        internal_solutions = solutions

        # -- Normalize if enabled
        #if True:
            #internal_solutions = np.array([self._normalize_weights(s) for s in internal_solutions])

        if len(internal_solutions) > 1:
            losses = np.apply_along_axis(
                partial(self.evaluate_single_solution, *func_args),
                axis=1,
                arr=internal_solutions,
            )
        else:
            losses = np.array([self.evaluate_single_solution(*func_args, internal_solutions[0])])

        return losses
    
    def evaluate_single_solution(
        self,
        predictions,
        labels,
        score_metric,
        normalize_predict_proba,
        weight_vector,
    ):
        """
        This method calculates the loss for a batch of candidate solutions by applying the evaluate_single_solution 
        method to each solution. It returns the calculates losses. 

        Parameters
        ----------
        predictions: np.ndarray
            Prediction probabilities from multiple models
        labels: np.ndarray
            Ground truth labels 
        score_metric: 
            Evaluation metric to evaluate the predicted labels against the ground truth labels
        normalize_predict_proba:
            Option not implemented
        weight_vector: np.ndarray
            Weighted ensemble vector

        Returns
        -------
        Ensemble performance evaluated using passed in score metric
        """
        y_pred_ensemble = AbstractWeightedEnsemble.weight_pred_probas(
            predictions,
            weight_vector,
        )
        return score_metric(labels, y_pred_ensemble)

    def _calculate_regret(self, y_true: np.ndarray, y_pred_proba: np.ndarray, metric, sample_weight=None):
        if metric.needs_pred or metric.needs_quantile:
            preds = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        else:
            preds = y_pred_proba
        score = compute_weighted_metric(y_true, preds, metric, sample_weight, quantile_levels=self.quantile_levels)
        return metric._optimum - score

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights


class SimpleWeightedEnsemble(AbstractWeightedEnsemble):
    """Predefined user-weights ensemble"""

    def __init__(self, weights, problem_type, **kwargs):
        self.weights_ = weights
        self.problem_type = problem_type

    @property
    def ensemble_size(self):
        return len(self.weights_)
