import argparse
import os

from data import process_data
from tunning_model import ModelOptimizer
from shap_explainer import shap_bar_plot, plot_predictions

from sklearn.metrics import r2_score

from sklearn.linear_model import ElasticNet, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd


def parse_args():
    """
    Function to parse command-line arguments.

    Returns:
        args: Parsed arguments containing the file path and selected models.
    """
    parser = argparse.ArgumentParser(description="Model Optimization with and without chaining")
    parser.add_argument(
        '--file-path', type=str, required=True, help="Path to the input data file (e.g., 'ex.txt')"
    )
    parser.add_argument(
        '--models', type=str, nargs='+', 
        choices=['ElasticNet', 
                 'SVR', 
                 'DecisionTreeRegressor', 
                 'LinearRegression', 
                 'BayesianRidge',
                 'GradientBoostingRegressor'],
        help="List of models to optimize. Options: 'ElasticNet', 'SVR', 'DecisionTreeRegressor', 'LinearRegression', 'BayesianRidge', 'GradientBoostingRegressor'"
    )
    return parser.parse_args()

def main():
    """
    Main function to load data, optimize models, and compute R2 scores.
    """
    # Parse command-line arguments
    args = parse_args()

    # Load data using the process_data function
    X, y = process_data(args.file_path)

    # Model classes
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

    results = []  

    # Loop through each model
    for model_info in models:
        model_name  = model_info['model']
        use_scaling = model_info['use_scaling']

        print(f"\nOptimizing model: {model_name.__name__}")

        # Optimize model with RegressorChain
        tuner_RegressorChain       = ModelOptimizer(X, y, use_scaling=use_scaling, use_chain=True)
        model_RegressorChain       = tuner_RegressorChain.optimize_model(model_name)

        # Optimize model with MultiOutputRegressor
        tuner_MultiOutputRegressor = ModelOptimizer(X, y, use_scaling=use_scaling, use_chain=False)
        model_MultiOutputRegressor = tuner_MultiOutputRegressor.optimize_model(model_name)

        # Append results to the results list
        results.append({
            'Model': model_name.__name__,
            'Use Chain': 'x',
            'R2 Score': r2_score(y, model_RegressorChain.predict(X))
        })
        results.append({
            'Model': model_name.__name__,
            'Use Chain': 'o',
            'R2 Score': r2_score(y, model_MultiOutputRegressor.predict(X))
        })

        shap_bar_plot(model_RegressorChain, 
                      X, 
                      feature_names=X.columns.tolist(), 
                      save_path=f"Figs/shap_RegressorChain_{model_name.__name__}.png")
        plot_predictions(model_RegressorChain, 
                         X, y, 
                         num_outputs=4, 
                         save_path=f"Figs/Prediction_RegressorChain_{model_name.__name__}.png")
        shap_bar_plot(model_MultiOutputRegressor, 
                      X, 
                      feature_names=X.columns.tolist(), 
                      save_path=f"Figs/shap_MultiOutputRegressor_{model_name.__name__}.png")
        plot_predictions(model_MultiOutputRegressor, 
                         X, y, 
                         num_outputs=4, 
                         save_path=f"Figs/Prediction_MultiOutputRegressor_{model_name.__name__}.png")
    #############################

    # Results and print
    results_df = pd.DataFrame(results)
    results_df.to_csv('./Output/res_R2.csv')
    print("\nFinal Results:")
    print(results_df)


# Main function
if __name__ == "__main__":
    main()
