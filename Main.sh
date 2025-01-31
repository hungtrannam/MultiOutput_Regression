#!/bin/bash
set -e  # Exit on error
set -x  # Print commands before running
# Define variables
#
# "ElasticNet" "BayesianRidge" "GradientBoostingRegressor" RandomForestRegressor
# XGBRegressor 'LGBMRegressor'



# ElasticNet BayesianRidge  RandomForestRegressor  XGBRegressor  LGBMRegressor CatBoostRegressor
FILE_PATH="ex.txt" 
MODELS=("SVR" "MLPRegressor" "BayesianRidge" "ElasticNet" "RandomForestRegressor" "GradientBoostingRegressor" "XGBRegressor")
NOISE_LEVEL=0.01
NUM_SAMPLE=20
num_iterations=100
NUM_TRIALS=50

# Run Python script for all models
echo "Running model comparison..."
python exp_main.py \
  --file-path "$FILE_PATH" \
  --models "${MODELS[@]}" \
  --num-samples "$NUM_SAMPLE" \
  --num_iterations "$num_iterations"\
  --num_trials "$NUM_TRIALS"\
  --noise-level-X "$NOISE_LEVEL" \
  --noise-level-y "$NOISE_LEVEL" > output.log 2>&1
echo "Done. Check output.log for details."

cat output.log

# Check results
# echo "Model comparison completed. Results saved in ./Output/results_comparison.csv."
