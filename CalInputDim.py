import pandas as pd
from data_loader import BAFDataset

# Load one of the bank files you just created
dataset = BAFDataset('data/bank_a.csv')

# Print the shape of the features (X)
# Shape will be [Number of Rows, Number of Columns]
print(f"Your input_dim should be: {dataset.X.shape[1]}")