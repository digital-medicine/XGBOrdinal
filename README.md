# XGBOrdinal

This GitHub repository contains the code used in the paper

`XGBOrdinal: An XGBoost Extension for Ordinal Data`

by Fabian Kahl, Iris Kahl, Stephan M. Jonas, the paper is currently submitted.

## Requirements
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Demos
- `./demo.ipynb` to run XGBOrdinal with and without GridSearchCV in Jupyter Notebook.
- `./demo.py` to run XGBOrdinal with and without GridSearchCV in Python.

## Parameters
**`XGBOrdinal(aggregation='weighted', norm=True, **extra_params)`**
- `aggregation: str`
    - **Description**: Defines the method for aggregating model across the classifiers.
    - **Purpose**: Controls how to combine the classifiers. Supported values:
        - `'weighted'`: Uses class distribution-based weights.
        - `'equal'`: Uses equal weights for all classifiers.
    - **Default**: `'weighted'`.
- `norm: bool`
    - **Description**: Whether to replace all negative outcomes with zero and normalize them so they sum to 1.
    - **Purpose**: Ensures the outputs are probabilities for each sample.
    - **Default**: `True`.
- `**extra_params`:
    - **Description**: Additional parameters passed to the underlying `XGBClassifier`s.
    - **Purpose**: Customize the underlying classifiers in terms of hyperparameters.
    - **Example**: `'learning_rate'=0.1`, `'max_depth'=3`.

## Methods
- `fit(X, y, **fit_params)`
    - **Description**: Trains multiple binary classifiers based on the ordinal thresholds derived from `unique_classes`.
    - **Parameters**:
        - `X`: The feature matrix for training.
        - `y`: The target vector for training.
        - `**fit_params`: Additional parameters passed to the underlying `XGBClassifier`s (e.g., `eval_set`).
- `predict(X)`
    - **Description**: Predicts the class label for each sample based on the highest predicted probability.
    - **Parameters**:
        - `X`: The feature matrix for prediction.
    - **Returns**: The predicted class labels.
- `predict_proba(X)`
    - **Description**: Predicts the probabilities for each ordinal class.
    - **Parameters**:
        - `X`: The feature matrix for prediction.
    - **Returns**: A 2D array where each row contains the predicted probabilities for each class.
    - **Note**: If `norm=True`, the probabilities will sum to 1 for each sample.
- `feature_importance(importance_type='gain')`
    - **Description**: Computes feature importance across all classifiers, aggregated using the specified `aggregation` strategy.
    - **Parameters**:
        - `importance_type`: Type of `XGBoost` importance to compute (e.g., `'gain'`, `'weight'`, `'cover'`).
    - **Returns**: A dictionary of feature importance scores.

## Folder Structure
- `./experiments` contains the experiments of the paper.
