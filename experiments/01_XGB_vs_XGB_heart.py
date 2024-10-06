"""
===================================================================
Title:          01_XGB_vs_XGB_heart.py
Description:    Compute results for Table 1 second row
Authors:        Fabian Kahl
===================================================================
"""

import sys
import os
sys.path.append(os.path.abspath('../'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgbordinal as xgbo
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Source: https://archive.ics.uci.edu/dataset/45/heart+disease
data_path = '../data/heart+disease/processed.cleveland.data'
y_col = 'num'      # Target column (labels to predict)
epochs = 100       # Number of trained models
decimals = 3       # Decimal places for rounding the final output metrics
experiment = '01'  # Experiment identifier (used in output file naming)
dataset = 'heart'  # Dataset name (used in output file naming)

# Load, one-hot encode, and split the dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]
df = pd.read_csv(data_path, header=None, names=column_names)
df = df.replace('?', np.nan)
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)
X_df = df.drop(y_col, axis=1).copy()
y = np.array(df[y_col])

# Lists to store results
mse_ord_list = []
mse_clf_list = []
mse_reg_list = []
mae_ord_list = []
mae_clf_list = []
mae_reg_list = []

for i in range(epochs):
    print('Run: '+str(i))
    # Split the dataset into training and testing sets (80/20 split)
    # Stratify ensures the training set has a balanced distribution of the target variable
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=i,
                        stratify=y)
    
    # XGBOrdinal
    ordinal_xgb=xgbo.XGBOrdinal()
    ordinal_xgb.fit(X_train_df, y_train)
    
    # XGBClassifier
    classifier_xgb = xgb.XGBClassifier()
    classifier_xgb.fit(X_train_df, y_train)
    
    # XGBRegressor
    regressor_xgb = xgb.XGBRegressor()
    regressor_xgb.fit(X_train_df, y_train)
    
    # Make predictions
    y_ordinal_pred = ordinal_xgb.predict(X_test_df)
    y_classifier_pred = classifier_xgb.predict(X_test_df)
    y_regressor_pred = np.clip(np.round(regressor_xgb.predict(X_test_df)).astype(int), y_train.min(), y_train.max())
            
    # Mean Squared Errors
    mse_ord = mean_squared_error(y_test, y_ordinal_pred)
    mse_clf = mean_squared_error(y_test, y_classifier_pred)
    mse_reg = mean_squared_error(y_test, y_regressor_pred)
    mse_ord_list.append(mse_ord)
    mse_clf_list.append(mse_clf)
    mse_reg_list.append(mse_reg)

    # Mean Absolute Errors
    mae_ord = mean_absolute_error(y_test, y_ordinal_pred)
    mae_clf = mean_absolute_error(y_test, y_classifier_pred)
    mae_reg = mean_absolute_error(y_test, y_regressor_pred)
    mae_ord_list.append(mae_ord)
    mae_clf_list.append(mae_clf)
    mae_reg_list.append(mae_reg)

# Format the final output showing mean and standard deviation for both MSE and MAE across the epochs
output = (
    f'Epochs: {epochs}\n'
    f'\n'
    f'Mean squared error (ordinal): {np.round(np.mean(mse_ord_list), decimals)} ± {np.round(np.std(mse_ord_list), decimals)}\n'
    f'Mean squared error (classifier): {np.round(np.mean(mse_clf_list), decimals)} ± {np.round(np.std(mse_clf_list), decimals)}\n'
    f'Mean squared error (regressor): {np.round(np.mean(mse_reg_list), decimals)} ± {np.round(np.std(mse_reg_list), decimals)}\n'
    f'\n'
    f'Mean absolute error (ordinal): {np.round(np.mean(mae_ord_list), decimals)} ± {np.round(np.std(mae_ord_list), decimals)}\n'
    f'Mean absolute error (classifier): {np.round(np.mean(mae_clf_list), decimals)} ± {np.round(np.std(mae_clf_list), decimals)}\n'
    f'Mean absolute error (regressor): {np.round(np.mean(mae_reg_list), decimals)} ± {np.round(np.std(mae_reg_list), decimals)}\n'
)
with open(f'./out/output_{experiment}_{dataset}.txt', 'w') as f:
    f.write(output)

