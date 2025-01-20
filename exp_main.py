import argparse
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from data import process_data
from tunning_model import ModelOptimizer
from shap_explainer import shap_bar_plot, plot_predictions, shap_summary_plot
from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )

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
        choices=['ElasticNet', 'SVR', 
                 'DecisionTreeRegressor', 'LinearRegression', 
                 'BayesianRidge', 'GradientBoostingRegressor',
                 'XGBRegressor', 'RandomForestRegressor',
                 'TabNetRegressor', 'LGBMRegressor'],
        help="List of models to optimize."
    )
    parser.add_argument(
        '--num-samples', type=int, default=100, help="Number of Monte Carlo samples (default: 100)"
    )
    parser.add_argument(
        '--noise-level-X', type=float, default=0.05, help="Noise level for input features in Monte Carlo simulations (default: 0.05)"
    )
    parser.add_argument(
        '--noise-level-y', type=float, default=0.05, help="Noise level for output targets in Monte Carlo simulations (default: 0.05)"
    )
    return parser.parse_args()

def main():
    """
    Main function to optimize regression models using Monte Carlo simulations.

    Steps:
        1. Load and preprocess input data.
        2. Initialize model configurations.
        3. Optimize each model using Optuna.
        4. Evaluate and save results.
        5. Generate SHAP explanations and prediction plots.
    """
    args = parse_args()

    # Step 1: Load and process data
    X, y = process_data(args.file_path)

    # Step 2: Define the mapping of model names to their corresponding classes
    model_map = {
        'ElasticNet': ElasticNet,
        'SVR': SVR,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'LinearRegression': LinearRegression,
        'BayesianRidge': BayesianRidge,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'XGBRegressor': XGBRegressor,
        'RandomForestRegressor':RandomForestRegressor,
        'TabNetRegressor':TabNetRegressor,
        'LGBMRegressor':LGBMRegressor,
    }

    models = []  # List to hold model configurations
    if args.models:
        for model_name in args.models:
            if model_name in model_map:
                # Determine if scaling is needed for the model
                models.append({'model': model_map[model_name], 'use_scaling': model_name != 'DecisionTreeRegressor'})

    results = []  # To store results

    # Step 3: Loop through each model and configuration
    for model_info in models:
        model_class = model_info['model']
        use_scaling = model_info['use_scaling']

        for use_chain in [True, False]:  # Test both RegressorChain and MultiOutputRegressor
            print(f"\nOptimizing model: {model_class.__name__} | Use Chain: {use_chain}")

            # Initialize ModelOptimizer with Monte Carlo settings
            tuner = ModelOptimizer(
                X, y, 
                use_scaling=use_scaling, 
                use_chain=use_chain,
                num_samples=args.num_samples, 
                noise_level_X=args.noise_level_X, 
                noise_level_y=args.noise_level_y
            )
            
            # Step 4: Optimize the model
            optimized_model = tuner.optimize_model(model_class)

            # Evaluate metrics
            y_pred = optimized_model.predict(tuner.X_test)
            r2 = r2_score(tuner.y_test, y_pred)
            mse = mean_squared_error(tuner.y_test, y_pred)

            # Append results
            results.append({
                'Model': model_class.__name__,
                'Use Chain': 'Yes' if use_chain else 'No',
                'R2 Score': r2,
                'MSE': mse
            })

            # Step 5.1: Generate SHAP and prediction plots
            shap_bar_plot(
                optimized_model, 
                tuner.X_test, 
                feature_names=tuner.X_test.columns.tolist(), 
                save_path=f"Figs/SHAP_{model_class.__name__}_Chain_{use_chain}.png"
            )

            shap_summary_plot(
                optimized_model,
                tuner.X_test,
                feature_names = tuner.X_test.columns.tolist(),
                save_path=f"Figs/SHAP_summary_{model_class.__name__}_Chain_{use_chain}.png"
            )
            plot_predictions(
                optimized_model, 
                tuner.X_test, tuner.y_test, 
                num_outputs=4, 
                save_path=f"Figs/Prediction_{model_class.__name__}_Chain_{use_chain}.png"
            )

            # Step 5.2 Generate Optuna visualization plots
            fig_opt_history = plot_optimization_history(tuner.study)
            fig_opt_history.write_image(f"Figs/OptHistory_{model_class.__name__}_Chain_{use_chain}.png")

            # Plot parameter importances
            fig_param_importance = plot_param_importances(tuner.study)
            fig_param_importance.write_image(f"Figs/ParamImportance_{model_class.__name__}_Chain_{use_chain}.png")

            # # Plot parallel coordinate plot
            # fig_parallel_coordinate = plot_parallel_coordinate(tuner.study)
            # fig_parallel_coordinate.write_image(f"Figs/ParallelCoordinate_{model_class.__name__}_Chain_{use_chain}.png")

    # Step 6: Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("./Output", exist_ok=True)
    results_df.to_csv('./Output/results_comparison.csv', index=False)
    tuner.X_mc.to_csv('./Output/MC_data.csv', index=False)

    # Print final results
    print("\nFinal Results:")
    print(results_df)

if __name__ == "__main__":
    main()
