import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")
    df = pd.read_csv(path)
    return df
