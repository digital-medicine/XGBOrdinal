"""
===================================================================
Title:          demo.py
Description:    Sample code for using XGBOrdinal with and without GridSearchCV
Authors:        Fabian Kahl
===================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import xgbordinal as xgbo
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes

# True: run with GridSearchCV, False: run without GidSearchCV
optimize_params = False

y_col = 'target_quantiles'
number_quantiles = 10

# Load and split the dataset
data = load_diabetes()
X_df = pd.DataFrame(data["data"], columns = data["feature_names"])
y = data["target"]

# Assign quantile categories
y = pd.qcut(y, q=number_quantiles, labels=False)

# Split the data
X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42,
                    stratify=y)

# Hyperparameter optimization or direct training
if optimize_params:
    param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.5],
        'subsample': [0.8, 1.0],
    }
    optimal_params = GridSearchCV(estimator=xgbo.XGBOrdinal(unique_classes=np.unique(y_train)),
                                            param_grid=param_grid, cv=5)
    optimal_params.fit(X_train_df, y_train)
    model = optimal_params.best_estimator_
else:
    model = xgbo.XGBOrdinal(unique_classes=np.unique(y_train))
    model.fit(X_train_df, y_train)

# Make predictions and calculate metrics
y_test_pred = model.predict(X_test_df)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

print(f'Mean squared error (ordinal): {mse:.3f}')
print(f'Mean absolute error (ordinal): {mae:.3f}')

