from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.metrics import mean_squared_error
import optuna

from sklearn.linear_model import ElasticNet, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


class ModelOptimizer:
    def __init__(self, X, y, use_scaling=True, use_chain=False):
        """
        Class to manage the data and optimize the model.

        Parameters:
            X (pd.DataFrame): Input data (features).
            y (pd.DataFrame): Output data (targets).
            use_scaling (bool): Whether to scale the data or not.
            use_chain (bool): Whether to use output chains (RegressorChain) or multi-output (MultiOutputRegressor).
        """
        self.X = X
        self.y = y
        self.use_scaling = use_scaling
        self.use_chain = use_chain

    def fit_flow(self, model_class, params):
        """
        Objective function for Optuna to optimize hyperparameters of the model.

        Parameters:
            model_class (class): The model class to be optimized (e.g., ElasticNet, RandomForestRegressor).
            params (dict): Dictionary containing the hyperparameters of the model.

        Returns:
            float: The average Mean Squared Error (MSE) for Leave-One-Out Validation.
        """
        # Create a pipeline for model training
        steps = []
        if self.use_scaling:
            steps.append(('scaler', StandardScaler()))  # Standardize features if required
        if self.use_chain:
            steps.append(('model', RegressorChain(model_class(**params))))  # Use RegressorChain
        else:
            steps.append(('model', MultiOutputRegressor(model_class(**params))))  # Use MultiOutputRegressor

        model = Pipeline(steps)

        # Perform Leave-One-Out Validation
        loo = LeaveOneOut()
        errors = []
        for train_idx, test_idx in loo.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            model.fit(X_train, y_train)     # Train model on training data
            y_pred = model.predict(X_test)  # Predict on test data
            errors.append(mean_squared_error(y_test, y_pred))  # Compute the MSE for the prediction

        return sum(errors) / len(errors)

    def objective(self, trial, model):
        """
        Define the parameters to optimize for the model.
        The parameters vary depending on the model.

        Parameters:
            trial (optuna.trial.Trial): The current Optuna trial.
            model (class): The model to optimize (e.g., DecisionTreeRegressor, SVR, etc.).

        Returns:
            float: The objective value (MSE) for the trial with the optimized parameters.
        """
        if model == DecisionTreeRegressor:
            # Parameters for DecisionTreeRegressor
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 50),
                # 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'max_features': trial.suggest_int('max_features', 1, self.X.shape[1]),
            }
        elif model == SVR:
            # Parameters for Support Vector Regressor (SVR)
            params = {
                'C': trial.suggest_float('C', 1e-5, 10),
                'epsilon': trial.suggest_float('epsilon', 1e-4, 1e-1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            }
        elif model == ElasticNet:
            # Parameters for ElasticNet
            params = {
                'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
                'max_iter': trial.suggest_int('max_iter', 1e2, 1e7, step=100),
            }
        elif model == LinearRegression:
            # Linear Regression doesn't require hyperparameter tuning
            params = {}
        elif model == BayesianRidge:
            # Parameters for BayesianRidge
            params = {
                'alpha_1': trial.suggest_float('alpha_1', 1e-6, 1e-1, log=True),
                'alpha_2': trial.suggest_float('alpha_2', 1e-6, 1e-1, log=True),
                'lambda_1': trial.suggest_float('lambda_1', 1e-6, 1e-1, log=True),
                'lambda_2': trial.suggest_float('lambda_2', 1e-6, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 1e2, 1e6, step=100),
            }
        elif model == GradientBoostingRegressor:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
                # 'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'max_features': trial.suggest_int('max_features', 1, self.X.shape[1]),
            }
        else:
            raise ValueError(f"Model {model} not supported")  # Raise an error if an unsupported model is provided

        # Return the Mean Squared Error
        return self.fit_flow(model, params)

    def optimize_model(self, model, n_trials=50):
        """
        Optimize the model by finding the best hyperparameters using Optuna.

        Parameters:
            model (class): The model class to optimize.
            n_trials (int): The number of optimization trials.

        Returns:
            model: The optimized model.
        """
        study = optuna.create_study(direction='minimize')  # Minimize MSE
        study.optimize(lambda trial: self.objective(trial, model), n_trials=n_trials)

        print(f"Best parameters for {model.__name__}: {study.best_params}")  # Log the best hyperparameters
        print(f"Best MSE for {model.__name__}: {study.best_value}")  # Log the best MSE

        # Build the best model
        if self.use_chain:
            best_model = RegressorChain(model(**study.best_params))  # Use RegressorChain
        else:
            best_model = MultiOutputRegressor(model(**study.best_params))  # Use MultiOutputRegressor

        # Fit the model 
        best_model.fit(self.X, self.y)

        return best_model 
