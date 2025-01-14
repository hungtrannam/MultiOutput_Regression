import pandas as pd

def process_data(file_path):
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

    # Feature engineering: Create interaction terms between features
    df['Sintering_temperature_&_Heating_rate'] = df['Sintering_temperature'] * df['Heating_rate']
    df['Sintering_temperature_&_Sintering_time'] = df['Sintering_temperature'] * df['Sintering_time']
    df['Heating_rate_&_Sintering_time'] = df['Heating_rate'] * df['Sintering_time']
    
    # Select input features (X) and target variables (y)
    X = df[['Synthesis_temperature', 'Sintering_temperature', 'Sintering_time', 'Heating_rate', 
            'Sintering_temperature_&_Heating_rate', 
            'Sintering_temperature_&_Sintering_time', 
            'Heating_rate_&_Sintering_time']]
    
    y = df[['Particle_size', 'Standard_deviation', 'Surface_area', 'Tab_density']]
    
    return X, y
