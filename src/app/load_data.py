import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SENSOR_DATA_DIR = ROOT / "sensor_output"

def load_data(data_path):
  try:
    data_df = pd.read_csv(data_path)
    data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], unit='s')
    # data_df.drop(columns=['timestamp'], inplace=True)
    print(f"[OK] Data loaded successfully from {data_path}. Shape: {data_df.shape}")
    return data_df
  except Exception as e:
    print(f"[ERROR] Failed to load data from {data_path}: {e}")
    return None