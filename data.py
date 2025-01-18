import pandas as pd

def process_data(file_path):
    """
    Read the file, process data, create interaction terms, and split into X and y.
    
    Parameters:
        file_path (str): Path to the input data file (e.g., .txt format).
        noise_level (float): Level of noise to add for Monte Carlo simulations.
    
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

    # Create interaction terms between features
    X['Sintering_temperature_&_Heating_rate'] = X['Sintering_temperature'] * X['Heating_rate']
    X['Sintering_temperature_&_Sintering_time'] = X['Sintering_temperature'] * X['Sintering_time']
    X['Heating_rate_&_Sintering_time'] = X['Heating_rate'] * X['Sintering_time']

    # Extract target variables
    y = df[targets]

    return X, y


def generate_mc_data(X, y, num_samples, noise_level_X, noise_level_y, random_state=42):
    """
    Generate Monte Carlo (MC) data with noise added to the original dataset, preserving column names.

    Parameters:
        X (pd.DataFrame): Original input data (features).
        y (pd.DataFrame): Original output data (targets).
        num_samples (int): Number of Monte Carlo samples to generate.
        noise_level_X (float): Noise level to add to specific input features.
        noise_level_y (float): Noise level to add to output targets.
        random_state (int): Seed for random number generator to ensure reproducibility.

    Returns:
        X_mc (pd.DataFrame): Monte Carlo-generated input data with column names.
        y_mc (pd.DataFrame): Monte Carlo-generated output data with column names.
    """
    import numpy as np
    rng = np.random.default_rng(random_state)
    X_mc = []
    y_mc = []

    for _ in range(num_samples):
        # Add noise only to specific features (not interaction terms)
        X_noisy = X.copy()
        for col in ['Synthesis_temperature', 'Sintering_temperature', 'Sintering_time', 'Heating_rate']:
            X_noisy[col] += rng.normal(0, noise_level_X, X_noisy[col].shape)

        # Keep interaction terms intact
        X_noisy['Sintering_temperature_&_Heating_rate'] = (
            X_noisy['Sintering_temperature'] * X_noisy['Heating_rate']
        )
        X_noisy['Sintering_temperature_&_Sintering_time'] = (
            X_noisy['Sintering_temperature'] * X_noisy['Sintering_time']
        )
        X_noisy['Heating_rate_&_Sintering_time'] = (
            X_noisy['Heating_rate'] * X_noisy['Sintering_time']
        )

        # Add noise to output targets
        y_noisy = y + rng.normal(0, noise_level_y, y.shape)

        # Convert to DataFrame to maintain column names
        X_noisy = pd.DataFrame(X_noisy, columns=X.columns)
        y_noisy = pd.DataFrame(y_noisy, columns=y.columns)

        X_mc.append(X_noisy)
        y_mc.append(y_noisy)

    # Concatenate all generated samples
    X_mc = pd.concat(X_mc, ignore_index=True)
    y_mc = pd.concat(y_mc, ignore_index=True)

    return X_mc, y_mc