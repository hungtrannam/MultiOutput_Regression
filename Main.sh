# "ElasticNet" "DecisionTreeRegressor" "LinearRegression" "BayesianRidge" "GradientBoostingRegressor"

#!/bin/bash

# Define variables
FILE_PATH="ex.txt"  # Replace with your actual data file path
MODELS=("ElasticNet" "LinearRegression" "BayesianRidge")
NOISE_LEVEL=2

# Run Python script for all models
echo "Running model comparison..."
python exp_main.py \
  --file-path "$FILE_PATH" \
  --models "${MODELS[@]}" \
  --noise-level $NOISE_LEVEL

# Check results
echo "Model comparison completed. Results saved in ./Output/results_comparison.csv."
