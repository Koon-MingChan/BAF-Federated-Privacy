import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# 1. This function splits the big CSV into "Bank" CSVs
def split_data_into_banks(source_path='data/Base.csv'):
    print("Partitioning data into simulated banks...")
    df = pd.read_csv(source_path)
    
    # Split by the 'month' column (0 to 7)
    bank_a = df[df['month'].isin([0, 1, 2])]
    bank_b = df[df['month'].isin([3, 4, 5])]
    bank_c = df[df['month'].isin([6, 7])]
    
    bank_a.to_csv('data/bank_a.csv', index=False)
    bank_b.to_csv('data/bank_b.csv', index=False)
    bank_c.to_csv('data/bank_c.csv', index=False)
    print(f"Created: Bank A ({len(bank_a)} rows), Bank B ({len(bank_b)}), Bank C ({len(bank_c)})")

# 2. This class converts a CSV into a format PyTorch understands
class BAFDataset(Dataset):

    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        
        # Separate labels and features
        self.y = torch.tensor(data['fraud_bool'].values, dtype=torch.float32).unsqueeze(1)
        X_raw = data.drop(columns=['fraud_bool'])
        
        # Basic Pre-processing: Convert categories to codes and scale numbers
        # This is critical for MLP models to converge
        X_numeric = X_raw.select_dtypes(include=['number']).fillna(0)
        self.feature_names = list(X_numeric.columns)
        
        # Scaling is mandatory for Differential Privacy to work properly later
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Helper to get the PyTorch DataLoader
def get_dataloader(bank_name, batch_size=64):
    path = f'data/bank_{bank_name}.csv'
    if not os.path.exists(path):
        split_data_into_banks()
    
    dataset = BAFDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class TieredPrivacyDataset(Dataset):
    def __init__(self, csv_path, sensitivity_report_path, privacy_level=1.0):
        # 1. Load data as usual
        data = pd.read_csv(csv_path)
        self.y = torch.tensor(data['fraud_bool'].values, dtype=torch.float32).unsqueeze(1)
        X_raw = data.drop(columns=['fraud_bool'])
        X_numeric = X_raw.select_dtypes(include=['number']).fillna(0)
        
        # 2. Load your FLSS Report
        report = pd.read_csv(sensitivity_report_path)
        # Create a mapping of Feature -> Entropy
        entropy_map = dict(zip(report['Feature'], report['Entropy_Risk']))
        
        # 3. Apply Tiered Noise
        X_noised = X_numeric.copy()
        for col in X_noised.columns:
            if col in entropy_map:
                # Calculate noise scale based on Entropy
                # Higher Entropy = Higher Noise
                # we use (entropy / max_entropy) as a scaling factor
                col_entropy = entropy_map[col]
                max_entropy = report['Entropy_Risk'].max()
                
                # This is your Tiered formula
                noise_scale = (col_entropy / max_entropy) * privacy_level
                
                noise = np.random.normal(0, noise_scale, size=X_noised[col].shape)
                X_noised[col] = X_noised[col] + noise
        
        # 4. Scale the noised data
        scaler = StandardScaler()
        self.X = torch.tensor(scaler.fit_transform(X_noised), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_noisedataloader(bank_name, batch_size=64, use_privacy=True):
    path = f'data/bank_{bank_name}.csv'
    report_path = 'feature_sensitivity_report.csv'
    
    if use_privacy:
        dataset = TieredPrivacyDataset(path, report_path, privacy_level=0.1) # Start small
    else:
        dataset = BAFDataset(path)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Test the split
    split_data_into_banks()