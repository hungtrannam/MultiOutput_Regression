import pandas as pd
import numpy as np

def add_uncertainty(X, noise_level=0.05):
    """
    Add uncertainty to input features by introducing noise.
    
    Args:
        X: Input data (numpy array or dataframe).
        noise_level: Standard deviation of the noise.

    Returns:
        X_noisy: Noisy version of the input data.
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def process_data(file_path, noise_level=0.05):
    """
    Read the file, process data, create interaction terms, and split into X and y.
    
    Parameters:
        file_path (str): Path to the input data file (e.g., .txt format).
    
    Returns:
        X (DataFrame): Input features, including interaction terms.
        y (DataFrame): Target variables.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep=r'\s+')

    X = df[['Synthesis_temperature', 'Sintering_temperature', 'Sintering_time', 'Heating_rate']]
    Xnoisy = add_uncertainty(X, noise_level)



    # Feature engineering: Create interaction terms between features
    Xnoisy['Sintering_temperature_&_Heating_rate'] = df['Sintering_temperature'] * df['Heating_rate']
    Xnoisy['Sintering_temperature_&_Sintering_time'] = df['Sintering_temperature'] * df['Sintering_time']
    Xnoisy['Heating_rate_&_Sintering_time'] = df['Heating_rate'] * df['Sintering_time']


    y = df[['Particle_size', 'Standard_deviation', 'Surface_area', 'Tab_density']]
    
    return Xnoisy, y


