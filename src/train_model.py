import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib # To save the scaler
import time

# --- Configuration ---
DATA_FILE = 'healthy_motor_data.csv'
MODEL_FILE = 'model.pth'
SCALER_FILE = 'scaler.gz'
INPUT_FEATURES = ['rpm', 'temperature', 'vibration']
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# --- Model Definition ---
class Autoencoder(nn.Module):
    """
    A simple Autoencoder model to learn the normal operating patterns of the motor.
    It compresses the input data to a bottleneck layer and then reconstructs it.
    """
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder: Compresses the input data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1) # Bottleneck layer with 1 neuron
        )
        # Decoder: Reconstructs the data from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Main Training Script ---
if __name__ == "__main__":
    print("--- Starting Model Training ---")
    start_time = time.time()

    # 1. Load and Preprocess Data
    print(f"Loading data from '{DATA_FILE}'...")
    df = pd.read_csv(DATA_FILE)
    data = df[INPUT_FEATURES].values.astype('float32')

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    print("Data scaled successfully.")

    # Save the scaler for later use in the live publisher
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to '{SCALER_FILE}'.")

    # 2. Prepare Data for PyTorch
    dataset = TensorDataset(torch.from_numpy(data_scaled))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = Autoencoder(input_dim=len(INPUT_FEATURES))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nModel Architecture:")
    print(model)

    # 4. Training Loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for data_batch in dataloader:
            inputs = data_batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')

    print("Training complete.")

    # 5. Save the Trained Model
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Trained model saved to '{MODEL_FILE}'.")

    # --- Step 5: Determine Anomaly Threshold ---
    print("\n--- Determining Anomaly Threshold ---")
    model.eval() # Set model to evaluation mode
    reconstruction_errors = []
    
    # We don't need gradients for this part
    with torch.no_grad():
        for data_batch in dataloader:
            inputs = data_batch[0]
            reconstructions = model(inputs)
            # Calculate loss for each item in the batch
            batch_loss = torch.mean((inputs - reconstructions) ** 2, dim=1)
            reconstruction_errors.extend(batch_loss.numpy())

    max_error = np.max(reconstruction_errors)
    # Set the threshold slightly higher than the max error found in healthy data
    threshold = max_error * 1.1 

    print(f"Maximum reconstruction error on healthy data: {max_error:.6f}")
    print(f"Anomaly detection threshold set to: {threshold:.6f}")
    print("You will use this threshold value in the publisher.py script.")
    
    end_time = time.time()
    print(f"\nTotal script time: {end_time - start_time:.2f} seconds.")
    print("--- Script Finished ---")
