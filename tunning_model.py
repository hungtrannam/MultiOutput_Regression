from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna

from sklearn.linear_model import ElasticNet, BayesianRidge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR



import numpy as np
from joblib import Parallel, delayed
from data import generate_mc_data
import pandas as pd


class ModelOptimizer:
    def __init__(self, X, y, use_scaling=True, num_sim=100, num_iterations = 100, noise_level_X=0.1, noise_level_y=0.1):
        self.use_scaling = use_scaling
        self.num_sim = num_sim
        self.num_iterations = num_iterations
        self.noise_level_X = noise_level_X
        self.noise_level_y = noise_level_y
        

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

    def objective(self, trial, model):
        if model == ElasticNet:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 1e3, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 1e-5, 1),
                'max_iter': trial.suggest_int('max_iter', 1e4, 1e7)
            }
        elif model == BayesianRidge:
            params = {
                'alpha_1': trial.suggest_float('alpha_1', 1e-5, 10, log=True),
                'alpha_2': trial.suggest_float('alpha_2', 1e-5, 10, log=True),
                'lambda_1': trial.suggest_float('lambda_1', 1e-5, 10, log=True),
                'lambda_2': trial.suggest_float('lambda_2', 1e-5, 10, log=True),
            }
        elif model == GradientBoostingRegressor:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 20),
            }
        elif model == XGBRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),  
                'max_depth': trial.suggest_int('max_depth', 3, 20),  
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),  
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),  
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),  
                'objective': "reg:squarederror",  
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),  
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 
                'gamma': trial.suggest_float('gamma', 0, 5), 
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  
                'tree_method': "hist", 
                'multi_strategy': "multi_output_tree"
            }

        elif model == RandomForestRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                # 'max_features': trial.suggest_categorical('max_features',['sqrt', 'log2'])
            }
        elif model == MLPRegressor:
            params = {
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 1e3, 1e6)
            }
        elif model == SVR:
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),  
                'epsilon': trial.suggest_float('epsilon', 1e-5, 1), 
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3, 
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']), 
            }
        else:
            raise ValueError(f"Model {model} not supported")

        return self.fit_flow_mccv(model, params)

    def fit_flow_mccv(self, model_class, params, val_size=0.1):
        def train_and_evaluate():
            # Split the data randomly into train and validation sets
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                self.X_train, self.y_train, test_size=val_size, random_state=np.random.randint(1e6)
            )

            # Generate Monte Carlo augmented data
            X_train_mc, y_train_mc = generate_mc_data(
                X_train_fold, y_train_fold, self.num_sim, self.noise_level_X, self.noise_level_y
            )

            if self.use_scaling:
                scaler = MinMaxScaler().fit(X_train_mc)
                X_train_mc = scaler.transform(X_train_mc)
                X_val_fold = scaler.transform(X_val_fold)

                # Convert back to DataFrame to retain feature names
                X_train_mc = pd.DataFrame(X_train_mc, columns=self.X_train.columns)
                X_val_fold = pd.DataFrame(X_val_fold, columns=self.X_train.columns)
                
                X_train_mc['Sintering_temperature_&_Sintering_time'] = X_train_mc['Sintering_temperature'] * X_train_mc['Sintering_time']
                X_val_fold['Sintering_temperature_&_Sintering_time'] = X_val_fold['Sintering_temperature'] * X_val_fold['Sintering_time']


            # Build the model pipeline
            model = MultiOutputRegressor(model_class(**params))


            # Train the model and calculate MAE on the validation set
            model.fit(X_train_mc, y_train_mc)
            y_pred = model.predict(X_val_fold)
            
            # Compute MAPE
            mape = np.mean(np.abs((y_val_fold - y_pred) / (y_val_fold + 1e-6))) * 100
            
            return mape

        # Use Parallel to evaluate multiple MCCV iterations in parallel
        scores = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)() for _ in range(self.num_iterations)
        )

        return np.mean(scores)

    def optimize_model(self, model, n_trials=10):
        # Create an Optuna study
        study = optuna.create_study(direction='minimize')
        # study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, model), n_trials=n_trials)

        # Save the study as an attribute for later use
        self.study = study

        print(f"Best parameters for {model.__name__}: {study.best_params}")
        print(f"Best MAPE for {model.__name__}: {study.best_value}")

        # Get the best hyperparameters
        best_params = study.best_params
        best_model = MultiOutputRegressor(model(**best_params))


        if self.num_sim > 0:
            X_train_mc, y_train_mc = generate_mc_data(
                self.X_train, self.y_train, self.num_sim, self.noise_level_X, self.noise_level_y
            )
        else:
            X_train_mc, y_train_mc = self.X_train, self.y_train

        if self.use_scaling:
            scaler = MinMaxScaler()
            X_train_mc = scaler.fit_transform(X_train_mc)
            self.X_test = scaler.transform(self.X_test)

            X_train_mc = pd.DataFrame(X_train_mc, columns=self.X_train.columns)
            self.X_test = pd.DataFrame(self.X_test, columns=self.X_train.columns)
            
            X_train_mc['Sintering_temperature_&_Sintering_time'] = X_train_mc['Sintering_temperature'] * X_train_mc['Sintering_time']

            self.X_test['Sintering_temperature_&_Sintering_time'] = self.X_test['Sintering_temperature'] * self.X_test['Sintering_time']
                
            
        best_model.fit(X_train_mc, y_train_mc)

        return best_model
