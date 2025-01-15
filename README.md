# Model Optimization with and without Chaining

This repository provides a Python-based implementation for optimizing machine learning models using different strategies, including Regressor Chains and MultiOutputRegressors. The implementation supports various regression models and includes functionalities for SHAP analysis and prediction visualization.

## Features
- **Supported Models**:
  - ElasticNet
  - SVR (Support Vector Regressor)
  - DecisionTreeRegressor
  - LinearRegression
  - BayesianRidge
  - GradientBoostingRegressor
- **Optimization Strategies**:
  - Regressor Chain
  - MultiOutputRegressor
- **SHAP Analysis**:
  - Generate SHAP bar plots to explain model predictions.
- **Prediction Visualization**:
  - Visualize predictions for a subset of outputs.
- **Metrics**:
  - Evaluate models using R2 scores.

## Requirements

### Python Packages
Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `shap`

### Directory Structure
Create the following directories in the project root to store results:
- `Figs/` for saving SHAP plots and prediction visualizations.
- `Output/` for saving evaluation results (e.g., CSV files).

## Usage

### 1. Prepare Data
The input data should be in a format supported by the `process_data` function. Ensure that your data file includes features and target variables appropriately structured.

### 2. Command-Line Arguments
Run the script using the following arguments:
- `--file-path` (required): Path to the input data file (e.g., `data/ex.txt`).
- `--models` (optional): List of models to optimize. Choose from the following:
  - `ElasticNet`
  - `SVR`
  - `DecisionTreeRegressor`
  - `LinearRegression`
  - `BayesianRidge`
  - `GradientBoostingRegressor`

#### Example
```bash
python exp_main.py \
--file-path ex.txt \
--models ElasticNet LinearRegression BayesianRidge DecisionTreeRegressor GradientBoostingRegressor
```

### 3. Output
- **Evaluation Results**:
  - Saved as `Output/res_R2.csv`.
  - Includes R2 scores for each model and optimization strategy.
- **Visualizations**:
  - SHAP plots: Saved in `Figs/` directory with filenames like `shap_RegressorChain_ModelName.png`.
  - Prediction plots: Saved in `Figs/` directory with filenames like `Prediction_RegressorChain_ModelName.png`.

## Implementation Details

### `process_data(file_path)`
This function preprocesses the input data file and splits it into features (`X`) and target variables (`y`).

### `ModelOptimizer`
A custom class used for tuning machine learning models. Supports both Regressor Chains and MultiOutputRegressors for optimization.

### `shap_bar_plot`
Generates SHAP bar plots to provide feature importance analysis.

### `plot_predictions`
Visualizes model predictions against the true target values for up to 4 outputs.

## Results
The final evaluation results, including R2 scores for each model and optimization strategy, will be saved in a CSV file. Example output format:

| Model                   | Use Chain | R2 Score |
|-------------------------|-----------|----------|
| ElasticNet              | x         | 0.9678   |
| ElasticNet              | o         | 0.9687   |
| LinearRegression        | x         | 0.9700   |
| LinearRegression        | o         | 0.9700   |
| BayesianRidge           | x         | 0.9637   |
| BayesianRidge           | o         | 0.9618   |
| DecisionTreeRegressor   | x         | 0.6691   |
| DecisionTreeRegressor   | o         | 0.7553   |
| GradientBoostingRegressor | x         | 0.6318   |
| GradientBoostingRegressor | o         | 0.8361   |

This table highlights the performance of various models with and without using chaining techniques.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the code or documentation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

