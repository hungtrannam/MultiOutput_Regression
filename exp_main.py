import argparse
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from data import process_data
from tunning_model import ModelOptimizer
from shap_explainer import shap_bar_plot, plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Model Optimization Comparison")
    parser.add_argument(
        '--file-path', type=str, required=True, help="Path to the input data file (e.g., 'ex.txt')"
    )
    parser.add_argument(
        '--models', type=str, nargs='+', 
        choices=['ElasticNet', 'SVR', 'DecisionTreeRegressor', 'LinearRegression', 'BayesianRidge', 'GradientBoostingRegressor'],
        help="List of models to optimize."
    )
    parser.add_argument(
        '--noise-level', type=float, default=0.05, help="Noise level for Monte Carlo simulations (default: 0.05)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    X, y = process_data(args.file_path, args.noise_level)

    # Model map
    model_map = {
        'ElasticNet': ElasticNet,
        'SVR': SVR,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'LinearRegression': LinearRegression,
        'BayesianRidge': BayesianRidge,
        'GradientBoostingRegressor': GradientBoostingRegressor
    }

    models = []
    if args.models:
        for model_name in args.models:
            if model_name in model_map:
                models.append({'model': model_map[model_name], 'use_scaling': model_name != 'DecisionTreeRegressor'})

    results = []  # To store results

    # Loop through each model and each configuration
    for model_info in models:
        model_class = model_info['model']
        use_scaling = model_info['use_scaling']

        for use_chain in [True, False]:
            for use_mc in [True, False]:
                print(f"\nOptimizing model: {model_class.__name__} | Use Chain: {use_chain} | Use MC: {use_mc}")

                tuner = ModelOptimizer(X, y, 
                                       use_scaling=use_scaling, 
                                       use_chain=use_chain, 
                                       use_mc=use_mc)
                optimized_model = tuner.optimize_model(model_class)

                # Evaluate metrics
                y_pred = optimized_model.predict(X)
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)

                # Append results
                results.append({
                    'Model': model_class.__name__,
                    'Use Chain': 'Yes' if use_chain else 'No',
                    'Use MC': 'Yes' if use_mc else 'No',
                    'R2 Score': r2,
                    'MSE': mse
                })

                # Generate SHAP and prediction plots
                shap_bar_plot(optimized_model, 
                              X, 
                              feature_names=X.columns.tolist(), 
                              save_path=f"Figs/SHAP_{model_class.__name__}_Chain_{use_chain}_MC_{use_mc}.png")
                plot_predictions(optimized_model, 
                                 X, y, 
                                 num_outputs=4, 
                                 save_path=f"Figs/Prediction_{model_class.__name__}_Chain_{use_chain}_MC_{use_mc}.png")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("./Output", exist_ok=True)
    results_df.to_csv('./Output/results_comparison.csv', index=False)
    X.to_csv('./Output/noisy_data.csv', index=False)

    # Print results table
    print("\nFinal Results:")
    print(results_df)

if __name__ == "__main__":
    main()
