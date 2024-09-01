import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import shap
import os
import matplotlib.pyplot as plt
import mlflow

# Parameters for GridSearchCV
param_grid = {
    'model__n_estimators': [10,20,100],
    'model__max_depth': [None, 20],
    'model__learning_rate': [0.1, 0.2]
}

model = XGBRegressor()

# Function to get feature names after transformation
def get_feature_names(trf, original_feature_names):
    ohe_feature_names = trf.named_transformers_['team_ohe'].get_feature_names_out(original_feature_names[:2])
    remainder_feature_names = original_feature_names[2:]
    return np.concatenate([ohe_feature_names, remainder_feature_names])

# Preprocessing and model training with GridSearchCV
def preprocessing(X_train, y_train, X_test, y_test):
    # Define the ColumnTransformer correctly
    trf1 = ColumnTransformer([
        ('team_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), [0, 1])
    ], remainder='passthrough')
    
    pipe = Pipeline([
        ('trf1', trf1),
        ('model', model)
    ])

    # GridSearch definition
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
    
    mlflow.set_experiment('ipl_xgboost_exp')

    with mlflow.start_run():    
        # Fit the GridSearchCV
        grid_search.fit(X_train, y_train)

        # Best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Log each parameter set in a nested run
        for i in range(len(grid_search.cv_results_['params'])):
            with mlflow.start_run(nested=True) as child:
                # Log the parameters for this specific run
                mlflow.log_params(grid_search.cv_results_['params'][i])

                # Log the corresponding mean test score
                mlflow.log_metric('mean_test_score', grid_search.cv_results_['mean_test_score'][i])
                
                # Optionally log additional metrics from cv_results_ (like std_dev, rank, etc.)

        # Log the best parameters found
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('mean_test_score', grid_search.best_score_)

        # Log the script itself as an artifact
        mlflow.log_artifact(__file__)

        # Log the best model found
        mlflow.sklearn.log_model(best_model, "xgboost_best_model")

        mlflow.set_tag('author', 'nigil')
        mlflow.set_tag('modelipl', 'xgboost')
    
    return best_model

# Main execution
df_final = pd.read_csv("./data/processed/df_final.csv")
X = df_final.drop(['total_score'], axis=1)
y = df_final['total_score']
original_feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

best_model = preprocessing(X_train, y_train, X_test, y_test)
