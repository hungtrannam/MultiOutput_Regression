import pandas as pd
import os
import numpy as np

def process_data(file_path):
    """
    Read the file, process data, create interaction terms, and split into X and y.

    Parameters:
        file_path (str): Path to the input data file (e.g., .txt format).

    Returns:
        X (pd.DataFrame): Input features, including interaction terms.
        y (pd.DataFrame): Target variables.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep=r'\s+')

    # Check if the required columns are present
    features = ['Synthesis_temperature', 'Sintering_temperature', 'Sintering_time', 'Heating_rate']
    targets = ['Particle_size', 'Standard_deviation', 'Surface_area', 'Tab_density']

    # Extract features
    X = df[features].copy()

    # Create interaction terms
    X['Sintering_temperature_&_Sintering_time'] = X['Sintering_temperature'] * X['Sintering_time']



    # Extract target variables
    y = df[targets]

    return X, y

def generate_mc_data(X, y, num_samples, noise_level_X, noise_level_y, random_state=42):
    """
    Generate Monte Carlo (MC) data with controlled noise and include the original data.

    Parameters:
        X (pd.DataFrame): Original input data (features).
        y (pd.DataFrame): Original output data (targets).
        num_samples (int): Number of Monte Carlo samples to generate.
        noise_level_X (float): Noise level to add to specific input features.
        noise_level_y (float): Noise level to add to output targets.
        random_state (int): Seed for reproducibility.

    Returns:
        X_mc (pd.DataFrame): Augmented input data, including original data.
        y_mc (pd.DataFrame): Augmented output data, including original data.
    """
    rng = np.random.default_rng(random_state)
    X_mc, y_mc = [], []

    for _ in range(num_samples):
        # Add noise to input features
        X_noisy = X.copy()
        for col in ['Synthesis_temperature', 'Sintering_temperature', 'Sintering_time', 'Heating_rate']:
            X_noisy[col] = np.clip(
                X[col] + rng.normal(0, noise_level_X * X[col], size=len(X)),
                a_min=0,  # Ensure non-negative values
                a_max=None
            )

        X_noisy['Sintering_temperature_&_Sintering_time'] = X_noisy['Sintering_temperature'] * X_noisy['Sintering_time']

        # Add noise to output targets
        y_noisy = y.copy()
        for col in y.columns:
            y_noisy[col] = np.clip(
                y[col] + rng.normal(0, noise_level_y * y[col], size=len(X)),
                a_min=0,  # Ensure non-negative values for outputs
                a_max=None
            )

        # Append to lists
        X_mc.append(X_noisy)
        y_mc.append(y_noisy)

    # Concatenate all generated samples and include the original data
    X_mc = pd.concat(X_mc + [X], ignore_index=True)
    y_mc = pd.concat(y_mc + [y], ignore_index=True)

    return X_mc, y_mc
