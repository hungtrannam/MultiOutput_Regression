from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error, r2_score
import optuna

from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from data import generate_mc_data

class ModelOptimizer:
    """
    A class for managing data and optimizing regression models using hyperparameter tuning.

    Attributes:
        X_original (pd.DataFrame): Original input data (features).
        y_original (pd.DataFrame): Original output data (targets).
        use_scaling (bool): Whether to scale the data before training.
        use_chain (bool): Whether to use RegressorChain or MultiOutputRegressor.
        num_samples (int): Number of Monte Carlo samples to generate.
        noise_level_X (float): Noise level for input features.
        noise_level_y (float): Noise level for output targets.
        X (pd.DataFrame): Monte Carlo-generated input data.
        y (pd.DataFrame): Monte Carlo-generated output data.
    """
    def __init__(self, X, y, use_scaling=True, use_chain=False, num_samples=100, noise_level_X=0.1, noise_level_y=0.1):
        """
        Initialize the ModelOptimizer with Monte Carlo data generation.

        Parameters:
            X (pd.DataFrame): Input data (features).
            y (pd.DataFrame): Output data (targets).
            use_scaling (bool): Whether to scale the data before training.
            use_chain (bool): Whether to use RegressorChain or MultiOutputRegressor.
            num_samples (int): Number of Monte Carlo samples to generate.
            noise_level_X (float): Noise level for input features.
            noise_level_y (float): Noise level for output targets.
        """
        self.X_original = X
        self.y_original = y
        self.use_scaling = use_scaling
        self.use_chain = use_chain
        self.num_samples = num_samples
        self.noise_level_X = noise_level_X
        self.noise_level_y = noise_level_y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_original, self.y_original, test_size=0.01
        )

        self.X_mc, self.y_mc = generate_mc_data(
            self.X_train, self.y_train, self.num_samples, self.noise_level_X, self.noise_level_y
        )


        
    def fit_flow_loocv(self, model_class, params):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) to optimize model hyperparameters with parallelization.

        Parameters:
            model_class (class): The regression model class to optimize (e.g., ElasticNet, SVR).
            params (dict): Hyperparameters of the model.

        Returns:
            float: The average Mean Squared Error (MSE) across all folds.
        """
        # Create a pipeline for model training
        steps = []
        if self.use_scaling:
            steps.append(('scaler', StandardScaler()))  # Scale the data if required
        if self.use_chain:
            steps.append(('model', RegressorChain(model_class(**params))))  # Use RegressorChain
        else:
            steps.append(('model', MultiOutputRegressor(model_class(**params))))  # Use MultiOutputRegressor

        model = Pipeline(steps)
        # loo = LeaveOneOut()  # Leave-One-Out Cross-Validation

        kf = KFold(n_splits=30, shuffle=True)  # K-Fold Cross-Validation

        def train_and_evaluate(train_idx, val_idx):
            """Train and evaluate the model on a single K-Fold split."""
            X_train, X_val = self.X_mc.iloc[train_idx], self.X_mc.iloc[val_idx]
            y_train, y_val = self.y_mc.iloc[train_idx], self.y_mc.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return r2_score(y_val, y_pred)

        # Use joblib for parallel processing
        scores = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)(train_idx, val_idx) for train_idx, val_idx in kf.split(self.X_mc)
        )

        return -np.mean(scores)
    

    def objective(self, trial, model):
        """
        Define the hyperparameter search space and objective function for Optuna.

        Parameters:
            trial (optuna.trial.Trial): The current Optuna trial.
            model (class): The regression model to optimize.

        Returns:
            float: The objective value (MSE) for the trial.
        """
        # Define hyperparameters based on the model type
        if model == DecisionTreeRegressor:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 50),
                'max_features': trial.suggest_int('max_features', 1, self.X_mc.shape[1]),
            }
        elif model == SVR:
            params = {
                'C': trial.suggest_float('C', 1e-5, 10),
                'epsilon': trial.suggest_float('epsilon', 1e-4, 1e-1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            }
        elif model == ElasticNet:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
                'max_iter': trial.suggest_int('max_iter', 1e2, 1e7, step=100),
            }
        elif model == BayesianRidge:
            params = {
                'alpha_1': trial.suggest_float('alpha_1', 1e-6, 1e-1, log=True),
                'alpha_2': trial.suggest_float('alpha_2', 1e-6, 1e-1, log=True),
                'lambda_1': trial.suggest_float('lambda_1', 1e-6, 1e-1, log=True),
                'lambda_2': trial.suggest_float('lambda_2', 1e-6, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 1e2, 1e7, step=100),
            }
        elif model == LinearRegression:
            params = {}
        elif model == GradientBoostingRegressor:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 50),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
                'max_features': trial.suggest_int('max_features', 1, self.X_mc.shape[1]),
            }
        else:
            raise ValueError(f"Model {model} not supported")

        return self.fit_flow_loocv(model, params)

    def optimize_model(self, model, n_trials=30):
        """
        Optimize the model by finding the best hyperparameters using Optuna.

        Parameters:
            model (class): The regression model class to optimize.
            n_trials (int): The number of optimization trials.

        Returns:
            model: The optimized regression model.
        """
        study = optuna.create_study(direction='minimize', storage="sqlite:///optuna_study.db", load_if_exists=True)
        study.optimize(lambda trial: self.objective(trial, model), n_trials=n_trials)

        self.study = study

        print(f"Best parameters for {model.__name__}: {study.best_params}")
        print(f"Best MSE for {model.__name__}: {study.best_value}")

        # Build and train the best model with the best hyperparameters using K-Fold CV to prevent data leakage
        best_params = study.best_params
        
        # Train the model on the full dataset with best parameters
        if self.use_chain:
            best_model = RegressorChain(model(**best_params))
        else:
            best_model = MultiOutputRegressor(model(**best_params))

        best_model.fit(self.X_train, self.y_train)
        return best_model

