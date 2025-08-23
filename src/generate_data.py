import pandas as pd
import numpy as np
import time

# --- Configuration ---
NUM_SAMPLES = 10000
OUTPUT_FILE = 'healthy_motor_data.csv'

def generate_healthy_data():
    """
    Generates a dataset of sensor readings from a "healthy" motor.
    In this simulation, vibration remains stable and does not increase over time.
    """
    print(f"Generating {NUM_SAMPLES} samples of healthy motor data...")
    
    # Create a range of timestamps
    timestamps = np.arange(NUM_SAMPLES)
    
    # Simulate stable, healthy sensor readings
    rpm = 700 + np.random.normal(0, 15, NUM_SAMPLES)
    temperature = 85.5 + np.random.normal(0, 1.0, NUM_SAMPLES)
    vibration = 3.5 + np.random.normal(0, 0.5, NUM_SAMPLES)
    
    # Create a pandas DataFrame
    data = {
        'timestamp': timestamps,
        'rpm': np.round(rpm, 2),
        'temperature': np.round(temperature, 2),
        'vibration': np.round(vibration, 2)
    }
    df = pd.DataFrame(data)
    
    return df

# --- Main Script ---
if __name__ == "__main__":
    start_time = time.time()
    
    dataset = generate_healthy_data()
    
    # Save the dataset to a CSV file
    dataset.to_csv(OUTPUT_FILE, index=False)
    
    end_time = time.time()
    
    print(f"Successfully created '{OUTPUT_FILE}' with {len(dataset)} samples.")
    print(f"Data generation took {end_time - start_time:.2f} seconds.")
