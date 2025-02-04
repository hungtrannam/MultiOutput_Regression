#!/bin/bash
set -e  # Exit on error
set -x  # Print commands before running

# "SVR" "MLPRegressor" "BayesianRidge" "ElasticNet" "RandomForestRegressor" "GradientBoostingRegressor" "XGBRegressor"

FILE_PATH="ex.txt" 
MODELS=("XGBRegressor")
NOISE_LEVEL=0.01
NUM_SIM=20
NUM_MCCV=100
NUM_TRIALS=50

# Run Python script for all models
echo "Running model comparison..."
python exp_main.py \
  --file-path "$FILE_PATH" \
  --models "${MODELS[@]}" \
  --num-sim "$NUM_SIM" \
  --num-iter "$NUM_MCCV"\
  --num-trials "$NUM_TRIALS"\
  --noise-level-X "$NOISE_LEVEL" \
  --noise-level-y "$NOISE_LEVEL" > output.log 2>&1
echo "Done. Check output.log for details."

cat output.log

# Check results
# echo "Model comparison completed. Results saved in ./Output/results_comparison.csv."
