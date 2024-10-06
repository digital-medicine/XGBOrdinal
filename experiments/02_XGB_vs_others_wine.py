"""
===================================================================
Title:          02_XGB_vs_others_wine.py
Description:    Compute results for Table 2 third row
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
from statsmodels.miscmodels.ordinal_model import OrderedModel
from ordinalgbt.lgb import LGBMOrdinal

# https://github.com/tomealbuquerque/ordinal-datasets?tab=readme-ov-file
data_path = '../data/winequality/winequality-white.csv'
y_col = 'quality'
epochs = 100       # Number of trained models
decimals = 3       # Decimal places for rounding the final output metrics
experiment = '02'  # Experiment identifier (used in output file naming)
dataset = 'wine'   # Dataset name (used in output file naming)

# Initialize a LabelEncoder to encode target labels into numerical form
label_encoder = LabelEncoder()

# Load and split the dataset
df = pd.read_csv(data_path, sep=';')
X_df = df.drop(y_col, axis=1).copy()
y = np.array(df[y_col])
y_encoded = label_encoder.fit_transform(y)

# Lists to store results
mse_xgb_list = []
mse_statsmodels_list = []
mse_lgbm_list = []
mae_xgb_list = []
mae_statsmodels_list = []
mae_lgbm_list = []

for i in range(epochs):
    print('Run: '+str(i))
    # Split the dataset into training and testing sets (80/20 split)
    # Stratify ensures the training set has a balanced distribution of the target variable
    X_train_df, X_test_df, y_train_encoded, y_test_encoded = train_test_split(X_df, y_encoded, test_size=0.2, random_state=i,
                        stratify=y_encoded)
    
    # XGBOrdinal
    ordinal_xgb = xgbo.XGBOrdinal()
    ordinal_xgb.fit(X_train_df, y_train_encoded)

    # Ordinal Regression
    mod_prob = OrderedModel(y_train_encoded, X_train_df, hasconst=False)
    ordinal_reg = mod_prob.fit(method='bfgs')

    # LGBMOrdinal
    ordinal_lgbm = LGBMOrdinal()
    ordinal_lgbm.fit(X_train_df, y_train_encoded)
    
    # Make predictions
    y_xgb_pred = ordinal_xgb.predict(X_test_df)
    y_statsmodels_pred = ordinal_reg.predict(X_test_df).idxmax(axis=1)
    y_lgbm_pred = ordinal_lgbm.predict(X_test_df)

    # Mean Squared Errors
    mse_xgb = mean_squared_error(y_test_encoded, y_xgb_pred)
    mse_statsmodels = mean_squared_error(y_test_encoded, y_statsmodels_pred)
    mse_lgbm = mean_squared_error(y_test_encoded, y_lgbm_pred)
    mse_xgb_list.append(mse_xgb)
    mse_statsmodels_list.append(mse_statsmodels)
    mse_lgbm_list.append(mse_lgbm)

    # Mean Absolute Errors
    mae_xgb = mean_absolute_error(y_test_encoded, y_xgb_pred)
    mae_statsmodels = mean_absolute_error(y_test_encoded, y_statsmodels_pred)
    mae_lgbm = mean_absolute_error(y_test_encoded, y_lgbm_pred)
    mae_xgb_list.append(mae_xgb)
    mae_statsmodels_list.append(mae_statsmodels)
    mae_lgbm_list.append(mae_lgbm)
    
# Format the final output showing mean and standard deviation for both MSE and MAE across the epochs
output = (
    f'Epochs: {epochs}\n'
    f'\n'
    f'Mean squared error (xgb): {np.round(np.mean(mse_xgb_list), decimals)} ± {np.round(np.std(mse_xgb_list), decimals)}\n'
    f'Mean squared error (statsmodels): {np.round(np.mean(mse_statsmodels_list), decimals)} ± {np.round(np.std(mse_statsmodels_list), decimals)}\n'
    f'Mean squared error (lgbm): {np.round(np.mean(mse_lgbm_list), decimals)} ± {np.round(np.std(mse_lgbm_list), decimals)}\n'
    f'\n'
    f'Mean absolute error (xgb): {np.round(np.mean(mae_xgb_list), decimals)} ± {np.round(np.std(mae_xgb_list), decimals)}\n'
    f'Mean absolute error (statsmodels): {np.round(np.mean(mae_statsmodels_list), decimals)} ± {np.round(np.std(mae_statsmodels_list), decimals)}\n'
    f'Mean absolute error (lgbm): {np.round(np.mean(mae_lgbm_list), decimals)} ± {np.round(np.std(mae_lgbm_list), decimals)}\n'
)
with open(f'./out/output_{experiment}_{dataset}.txt', 'w') as f:
    f.write(output)

