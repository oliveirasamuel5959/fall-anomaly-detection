import pandas as pd
import numpy as np
from pathlib import Path
from src.training.modules.FeatureEng import rotation_matrix
from src.training.modules.Model import load_model
from src.training.modules.TimeSeriesData import create_data_repetition
from src.training.modules.Processing import DataPreprocessor
from src.core.logger import get_logger

from src.app.load_data import load_data
from src.app.plot import plot_inference

ROOT = Path(__file__).parent.parent.parent

OUT_DIR = ROOT / "results-inference"
SENSOR_DATA_DIR = ROOT / "sensor_output"
MODEL_PATH = ROOT / "models" / "base-w-600" / "lstm-epoch_19_valloss_0.2680_valacc_0.8901.h5"

logger = get_logger(__name__)
preprocessor = DataPreprocessor(scaler_type='standard')

fall_windows = []  # Example fall windows in seconds

FEATURE_COLUMNS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

def inference():
  """
  Perform inference using the provided model and data.
  
  Args:
    model: The trained model to use for inference.
    data: The input data for which to perform inference.
  
  Returns:
    The model's predictions for the input data.
  """
  # Load the trained model
  model = load_model(MODEL_PATH)
  
  if model is None:
    return None
  
  # Load sensor data for inference
  sensor_data = load_data(f"{SENSOR_DATA_DIR}/sensor_data_fall_6.csv")
  logger.info(f"Loaded sensor data with shape:\n {sensor_data.shape}")
  logger.info(f"Sensor data timestamp range:\n {sensor_data.timestamp.min()} - {sensor_data.timestamp.max()}")
  logger.info(f"Sample of loaded sensor data:\n{sensor_data.head()}")
  
  # Apply rotation to align sensor data to a common reference frame
  df_rot = rotation_matrix(df=sensor_data, theta=-np.pi, axis='z')
  logger.info(f"Applied rotation to sensor data to align with reference frame:\n{df_rot.head()}")
  
  # Select the relevant columns for inference (e.g., accelerometer and gyroscope data)
  x_sample = df_rot.copy()
  logger.info(f"Selected columns for inference:\n{FEATURE_COLUMNS}")
  logger.info(f"Sample of selected data:\n{x_sample.head()}")
  
  # Create data sequences for timeseries LSTM model
  X, W = create_data_repetition(feature_columns=FEATURE_COLUMNS, data_type='inference', df=x_sample, w=600, s=100)

  W = [tuple(window) for window in W]
  
  logger.info(f"Created {len(X)} data windows for inference.\n")
  logger.info(f"Windows:\n{W}")
  
  # Normalize the data using the same scaler as during training
  X_scaled = preprocessor.scaler_fit_transform(X)
  
  # Perform inference
  y_probs = model.predict(X_scaled, verbose=0)
  y_pred_cls = np.argmax(y_probs, axis=1)
  
  # Map predicted class indices to category names
  categories = ['Fall', 'Normal']
  
  # Log predictions and identify fall windows
  for i, pred_idx in enumerate(y_pred_cls):
    
    # Log the predicted class for the current window
    if categories[pred_idx] == 'Fall':  # Fall detected
      logger.info("Fall detected in the current window.\n")
      
      fall_windows.append(W[i])  # Store the time window of the detected fall
      
      logger.info(f"Fall window (start, end): {W[i]}")
      
      # Log the data in the fall window for debugging/analysis
      X_df = pd.DataFrame(X_scaled[i], columns=FEATURE_COLUMNS)
      logger.info(f"Sample of data in the fall window:\n{X_df.head()}")
      
    logger.info(f"Predicted class: {categories[pred_idx]}\n")
    # fall_windows.append(X_df.index)
    
  plot_inference(sensor_data, fall_windows=fall_windows, title="Inference Data Visualization", out_dir=OUT_DIR)
  
if __name__ == "__main__":
  inference()