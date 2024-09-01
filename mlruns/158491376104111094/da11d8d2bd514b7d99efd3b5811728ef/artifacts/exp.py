import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import shap
import os
import matplotlib.pyplot as plt
import yaml
import mlflow
import dagshub

#parameters for Grid search CV

pram_grid = {
    'model__n_estimators': [10, 20, 30],
    'model__max_depth': [None, 10, 20],
    'model__learning_rate': [0.1, 0.2]
}

model=XGBRegressor()

#function to get feature names after transformation

def get_feature_names(trf, original_feature_names):
    ohe_feature_names = trf.named_transformers_['team_ohe'].get_feature_names_out(original_feature_names[:2])
    remainder_feature_names = original_feature_names[2:]
    return np.concatenate([ohe_feature_names, remainder_feature_names])

#Preprocessing and model training with Grid search CV

def preprocessing(X_train, y_train, X_test, y_test):
    # Define the ColumnTransformer correctly
    trf1 = ColumnTransformer([
        ('team_ohe', OneHotEncoder(handle_unknown='ignore', drop='first'), [0, 1])
    ], remainder='passthrough')
    
    model=XGBRegressor()
    pipe = Pipeline([
            ('trf1', trf1),
            ('model', model)
        ])

    #Grid search definition
    grid_search = GridSearchCV(estimator=pipe,param_grid=pram_grid,cv=5,n_jobs=1,verbose=2)
    
    mlflow.set_experiment('ipl_xgboost_exp')

    with mlflow.start_run():    

         # Fit the GridSearchCV
        grid_search.fit(X_train, y_train)

        # Best model from GridSearchCV
        best_model = grid_search.best_estimator_ 

    
        
        #log mlflow experiments
        mlflow.log_param('best_params', grid_search.best_params_)
  
        mlflow.log_artifact(__file__)

        mlflow.sklearn.log_model(best_model, "xgboost_best_model")

        mlflow.set_tag('author','nigil')
        mlflow.set_tag('modelipl','xgboost')
        
        
        return pipe

df_final = pd.read_csv("./data/processed/df_final.csv")
X = df_final.drop(['total_score'], axis=1)
y = df_final['total_score']
original_feature_names = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=22)

pipe= preprocessing(X_train, y_train, X_test, y_test)



