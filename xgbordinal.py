"""
===================================================================
Title:          xgbordinal.py
Description:    Ordinal XGBoost classifier
Authors:        Fabian Kahl
===================================================================
"""

import numpy as np
from typing import Any, Dict
from numpy.typing import ArrayLike
from sklearn.base import clone
import copy
from xgboost import XGBClassifier
try:
    import utils
except ImportError:
    from . import utils

class XGBOrdinal(XGBClassifier):
    def __init__(self, aggregation='weighted', norm=True, **extra_params):
        # Initialize the ordinal classifier
        self.clf = XGBClassifier
        self.aggregation = aggregation
        self.norm = norm
        self.model_weights = None  # Weighting for model
        self.kwargs = extra_params  # Store extra parameters
        super().__init__(**extra_params)

    def set_model_weights(self, y):
        # Calculate model weights based on aggregation type
        if self.aggregation == 'weighted':
            class_dist = np.bincount(y)
            absolute_model_weights = class_dist[:-1] + class_dist[1:]
            self.model_weights = (absolute_model_weights) / np.sum(absolute_model_weights)
        elif self.aggregation == 'equal':
            self.model_weights = np.full((self.number_clfs,), 1.0 / self.number_clfs)
        else:
            raise NotImplementedError(f"{self.aggregation} is not implemented yet!")

    def fit(self, X: ArrayLike, y: ArrayLike, **fit_params: Any) -> "XGBOrdinal":
        # Identify the number of unique ordinal classes
        self.unique_classes = np.unique(y)
        self.number_classes = len(self.unique_classes)
        self.number_clfs = self.number_classes - 1
        self.clfs = {}  # Dictionary to hold the binary classifiers

        # Extract eval_set from fit_params if provided
        self.eval_set = fit_params.pop('eval_set', None)

        # Set model weights
        self.set_model_weights(y)

        # Create and fit k-1 classifiers
        for i in range(self.number_clfs):
            # Generate binary labels for "class > C_i"
            binary_y = (y > self.unique_classes[i]).astype(int)
            
            # Create a binary classifier for each threshold
            self.clfs[i] = clone(self.clf(**self.kwargs))

            # Adjust eval_set for the current binary classifier if provided
            if self.eval_set is not None:
                binary_eval_set = [(X_val, (y_val > self.unique_classes[i]).astype(int)) for X_val, y_val in self.eval_set]
                self.clfs[i].fit(X, binary_y, eval_set=binary_eval_set, **fit_params)
            else:
                self.clfs[i].fit(X, binary_y, **fit_params)

        # Aggregate errors across classifiers if eval_set is provided
        if self.eval_set is not None:
            self.evals_result_ = utils.copy_structure(self.clfs[0].evals_result_)
            for j in self.clfs[0].evals_result_:
                for k in self.clfs[0].evals_result_[j]:
                    max_length = max(len(self.clfs[m].evals_result_[j][k]) for m in self.clfs)
                    errors_mean = np.zeros(max_length)
                    for n in self.clfs:
                        errors = np.array(self.clfs[n].evals_result_[j][k])
                        errors_padded = np.pad(errors, (0, max(0, max_length - len(errors))), 'edge')
                        errors_mean += errors_padded * self.model_weights[n]
                    self.evals_result_[j][k] = errors_mean
        return self
    
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        # Predict cumulative probabilities for each binary classifier
        cumulative_probas = np.array([self.clfs[i].predict_proba(X)[:, 1] for i in range(self.number_clfs)]).T

        # Calculate probabilities for ordinal classification
        probas = np.zeros((X.shape[0], self.number_classes))

        probas[:, 0] = 1 - cumulative_probas[:, 0]
        probas[:, 1:-1] = cumulative_probas[:, :-1] - cumulative_probas[:, 1:]
        probas[:, -1] = cumulative_probas[:, -1]

        if self.norm:
            # Replace all negative values with zero
            probas = np.maximum(0, probas)
            # Normalize the results so that the probabilities sum to 1 for each sample
            probas = probas / probas.sum(axis=1, keepdims=True)

        return probas

    def predict(self, X: ArrayLike) -> ArrayLike:
        # Predict the class with the highest probability
        probas = self.predict_proba(X)
        return self.unique_classes[np.argmax(probas, axis=1)]

    def set_params(self, **params: Any) -> "XGBOrdinal":
        # Store parameters in kwargs to apply them later in fit
        for key, value in params.items():
            self.kwargs[key] = value
    
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # Get parameters of the estimator
        params = super().get_params(deep)
        # Include additional parameters stored in kwargs
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        return params

    def feature_importance(self, importance_type: str = 'gain') -> dict:
        # Calculate mean feature importance across all classifiers
        mean_feature_importance = {}
        for p in range(self.number_clfs):
            feature_importance = self.clfs[p].get_booster().get_score(importance_type=importance_type)
            mean_feature_importance = {key: mean_feature_importance.get(key, 0) + feature_importance.get(key, 0)*self.model_weights[p] for key in set(mean_feature_importance) | set(feature_importance)}
        mean_feature_importance = dict(sorted(mean_feature_importance.items(), key=lambda item: item[1], reverse=True))
        return mean_feature_importance