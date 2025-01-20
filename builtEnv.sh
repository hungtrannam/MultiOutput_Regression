#!/bin/bash

set -e  # Stop the script if any command fails
# LOGFILE="SetUp_venv.log"
# echo "Setup started at $(date)" > "$LOGFILE"

echo "Updating the system..."
sudo apt update -y && sudo apt upgrade -y

############## Create a virtual environment ##############
echo "Creating a virtual environment..."
python3 -m venv .venv

############## Activate the virtual environment ##############
echo "Activating the virtual environment..."
source .venv/bin/activate

############## Upgrade pip ##############
echo "Upgrading pip..."
pip install --upgrade pip

############## Install required packages ##############
echo "Installing required packages..."
pip install numpy matplotlib scipy pandas psutil scikit-learn\
 ipykernel tqdm seaborn shap xgboost optuna plotly kaleido pytorch-tabnet lightgbm

############## Check Python and pip versions ##############
# echo "Checking installation..."
# python --version
# pip --version

############## Export the list of installed packages to requirements.txt ##############
echo "Exporting installed packages to requirements.txt..."
pip freeze > requirements.txt

############## Jupyter kernel configuration ##############
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name "Python (venv)"

############## Additional Utilities ##############
# echo "Adding useful utilities..."
# echo "alias activate_env='source $(pwd)/.venv/bin/activate'" >> ~/.bashrc
source ~/.bashrc
source .venv/bin/activate

############## Final Message ##############
echo "---> The environment has been successfully set up!"
echo "---> To activate the environment, use the command: source .venv/bin/activate"
echo "---> Setup completed at $(date)"
