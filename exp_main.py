import argparse
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import ElasticNet, BayesianRidge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR



from data import process_data
from tunning_model import ModelOptimizer
from shap_explainer import shap_bar_plot, plot_predictions, shap_summary_plot, plot_permutation_importance
from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
            )

import numpy as np

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Optimization with Monte Carlo Simulation")
    parser.add_argument(
        '--file-path', type=str, required=True, help="Path to the input data file (e.g., 'ex.txt')"
    )
    parser.add_argument(
    '--models', type=str, nargs='+', 
    choices=['ElasticNet', 'BayesianRidge', 'GradientBoostingRegressor', 
             'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor', 
             'MLPRegressor', 'SVR'],
    help="List of models to optimize."
    )

    parser.add_argument(
        '--num-sim', type=int, default=100, help="Number of Monte Carlo samples (default: 100)"
    )
    parser.add_argument(
        '--noise-level-X', type=float, default=0.01, help="Noise level for input features in Monte Carlo simulations (default: 0.01)"
    )
    parser.add_argument(
        '--noise-level-y', type=float, default=0.01, help="Noise level for output targets in Monte Carlo simulations (default: 0.01)"
    )
    parser.add_argument(
        '--num-trials', type=int, default=10, help="Number of trials when using Optuna (default: 10)"
    )
    parser.add_argument(
        '--num-iter', type=int, default=100, help="Number of trials when using Optuna (default: 10)"
    )
    return parser.parse_args()

def main():
    """
    Main function to optimize regression models using Monte Carlo simulations.
    """
    args = parse_args()
    print(f"Arguments: {args}")

    print("Loading data...")
    X, y = process_data(args.file_path)

    model_map = {
        'ElasticNet': ElasticNet,
        'BayesianRidge': BayesianRidge,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'XGBRegressor': XGBRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'MLPRegressor':MLPRegressor,
        'SVR': SVR,

    }

    results = []
    os.makedirs('./Output', exist_ok=True)
    os.makedirs('./Figs', exist_ok=True)

    if args.models:
        print(f"Selected models: {args.models}")
        for model_name in args.models:
            if model_name in model_map:
                model_class = model_map[model_name]
                use_scaling = model_name not in ['RandomForestRegressor']

                try:
                    tuner = ModelOptimizer(
                        X, y,
                        use_scaling=use_scaling,
                        num_sim=args.num_sim,
                        num_iterations=args.num_iter,
                        noise_level_X=args.noise_level_X,
                        noise_level_y=args.noise_level_y
                    )
                    print(f"Initialized tuner for {model_class.__name__}. Running optimization...")
                    optimized_model = tuner.optimize_model(model_class, n_trials=args.num_trials)
                except Exception as e:
                    print(f"Error during optimization for {model_class.__name__}: {e}")
                    continue

                y_pred = optimized_model.predict(tuner.X_test)
                r2 = r2_score(tuner.y_test, y_pred)
                mse = mean_squared_error(tuner.y_test, y_pred)
                mae = mean_absolute_error(tuner.y_test, y_pred)
                mape = np.mean(np.abs((tuner.y_test - y_pred) / tuner.y_test)) * 100

                results.append({
                    'Model': model_class.__name__,
                    'R2 Score': r2,
                    'MSE': mse,
                    'MAE': mae,
                    'MAPE':mape

                })
                print(f"Results for {model_class.__name__}: R2 = {r2}, MSE = {mse}, MAE = {mae}, MAPE = {mape}%")

                tuner.X_test = pd.DataFrame(tuner.X_test, columns=X.columns)
                
                try:
                    # Visualize
                    shap_bar_plot(
                        optimized_model,
                        tuner.X_test,
                        feature_names=tuner.X_test.columns.tolist(),
                        save_path=f"Figs/SHAP_{model_class.__name__}.png"
                    )
                    shap_summary_plot(
                        optimized_model,
                        tuner.X_test,
                        feature_names=tuner.X_test.columns.tolist(),
                        save_path=f"Figs/SHAPsum_{model_class.__name__}.png"
                    )
                    plot_predictions(
                        optimized_model,
                        tuner.X_test, tuner.y_test,
                        num_outputs=tuner.y_test.shape[1],
                        save_path=f"Figs/Prediction_{model_class.__name__}.png"
                    )
                    plot_permutation_importance(
                        optimized_model,
                        tuner.X_test, tuner.y_test,
                        save_path=f"Figs/PI_{model_class.__name__}.png"
                    )
                    fig_opt_history = plot_optimization_history(tuner.study)
                    fig_opt_history.write_image(f"Figs/OptHistory_{model_class.__name__}.png")
                    fig_param_importance = plot_param_importances(tuner.study)
                    fig_param_importance.write_image(f"Figs/ParamImportance_{model_class.__name__}.png")
                except Exception as e:
                    print(f"Error during plotting for {model_class.__name__}: {e}")
                    continue

    if results:
        results_df = pd.DataFrame(results)
        print("\nFinal Results:")
        print(results_df)
        results_df.to_csv('./Output/results_comparison.csv', index=False)
        print("Results saved.")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()  
