#!/bin/bash

# Define variables
#
# "ElasticNet" "LinearRegression" "BayesianRidge" "DecisionTreeRegressor" "GradientBoostingRegressor" RandomForestRegressor
# XGBRegressor 'LGBMRegressor' 
FILE_PATH="ex.txt" 
MODELS=("ElasticNet" "BayesianRidge" "RandomForestRegressor" "LinearRegression" "GradientBoostingRegressor" "DecisionTreeRegressor")
NOISE_LEVEL=0.01
NUM_SAMPLE=500

# Run Python script for all models
echo "Running model comparison..."
python exp_main.py \
  --file-path "$FILE_PATH" \
  --models "${MODELS[@]}" \
  --num-samples "$NUM_SAMPLE" \
  --noise-level-X "$NOISE_LEVEL" \
  --noise-level-y "$NOISE_LEVEL"

# Check results
echo "Model comparison completed. Results saved in ./Output/results_comparison.csv."
