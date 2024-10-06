"""
===================================================================
Title:          01_XGB_vs_XGB_cars.py
Description:    Compute results for Table 1 first row
Authors:        Fabian Kahl
===================================================================
"""

import sys
import os
sys.path.append(os.path.abspath('../'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgbordinal as xgbo
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Source: https://archive.ics.uci.edu/dataset/19/car+evaluation
data_path = '../data/car+evaluation/car.data'
y_col = 'class'    # Target column (labels to predict)
y_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
epochs = 100       # Number of trained models
decimals = 3       # Decimal places for rounding the final output metrics
experiment = '01'  # Experiment identifier (used in output file naming)
dataset = 'cars'   # Dataset name (used in output file naming)

# Initialize a LabelEncoder to encode target labels into numerical form
label_encoder = LabelEncoder()

# Load, one-hot encode, and split the dataset
df = pd.read_csv(data_path, header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df_encoded = pd.get_dummies(df.drop(y_col, axis=1))
df_encoded[y_col] = df[y_col].map(y_mapping)
X_df = df_encoded.drop(y_col, axis=1).copy()
y = np.array(df_encoded[y_col])
y_encoded = label_encoder.fit_transform(y)

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
    X_train_df, X_test_df, y_train_encoded, y_test_encoded = train_test_split(X_df, y_encoded, test_size=0.2, random_state=i,
                        stratify=y_encoded)
    
    # XGBOrdinal
    ordinal_xgb=xgbo.XGBOrdinal()
    ordinal_xgb.fit(X_train_df, y_train_encoded)
    
    # XGBClassifier
    classifier_xgb = xgb.XGBClassifier()
    classifier_xgb.fit(X_train_df, y_train_encoded)
    
    # XGBRegressor
    regressor_xgb = xgb.XGBRegressor()
    regressor_xgb.fit(X_train_df, y_train_encoded)
    
    # Make predictions
    y_ordinal_pred = ordinal_xgb.predict(X_test_df)
    y_classifier_pred = classifier_xgb.predict(X_test_df)
    y_regressor_pred = np.clip(np.round(regressor_xgb.predict(X_test_df)).astype(int), y_train_encoded.min(), y_train_encoded.max())
            
    # Mean Squared Errors
    mse_ord = mean_squared_error(y_test_encoded, y_ordinal_pred)
    mse_clf = mean_squared_error(y_test_encoded, y_classifier_pred)
    mse_reg = mean_squared_error(y_test_encoded, y_regressor_pred)
    mse_ord_list.append(mse_ord)
    mse_clf_list.append(mse_clf)
    mse_reg_list.append(mse_reg)

    # Mean Absolute Errors
    mae_ord = mean_absolute_error(y_test_encoded, y_ordinal_pred)
    mae_clf = mean_absolute_error(y_test_encoded, y_classifier_pred)
    mae_reg = mean_absolute_error(y_test_encoded, y_regressor_pred)
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

