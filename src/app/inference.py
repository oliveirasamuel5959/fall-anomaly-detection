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
SENSOR_DATA_DIR = ROOT / "sensor_output"

MODEL_PATH = ROOT / "models" / "base-w-600" / "lstm-epoch_19_valloss_0.2680_valacc_0.8901.h5"

logger = get_logger(__name__)
preprocessor = DataPreprocessor(scaler_type='standard')

fall_windows = [(12.3, 13.1), (27.0, 27.9)]  # Example fall windows in seconds

def inference():
  """
  Perform inference using the provided model and data.
  
  Args:
    model: The trained model to use for inference.
    data: The input data for which to perform inference.
  
  Returns:
    The model's predictions for the input data.
  """
  model = load_model(MODEL_PATH)
  
  if model is None:
    return None
  
  sensor_data = load_data(f"{SENSOR_DATA_DIR}/sensor_data_fall.csv")
  print(sensor_data.head())
  
  # sensor_data = sensor_data.drop(columns=['timestamp'])
  logger.info(f"Sensor data shape: {sensor_data.shape}")
  
  df_rot = rotation_matrix(df=sensor_data, theta=np.pi, axis='z')
  
  x_sample = df_rot.iloc[:,1:7]
  print(x_sample.head())
  
  # Create data sequences for timeseries LSTM model
  X = create_data_repetition(feature_columns=x_sample.columns.tolist(), data_type='inference', df=x_sample, w=600, s=200)
  X_scaled = preprocessor.scaler_fit_transform(X)
  X_df = pd.DataFrame(X_scaled[0], columns=x_sample.columns.tolist())

  logger.info(f"X window sequences shape: {X.shape}")
  print(X_df.head())
  
  y_probs = model.predict(X_scaled, verbose=0)
  y_pred_cls = np.argmax(y_probs, axis=1)
  
  categories = ['Fall', 'Normal']
  for pred_idx in y_pred_cls:
    logger.info(f"Predicted class: {categories[pred_idx]}")
    # fall_windows.append(X_df.index)
    
  plot_inference(sensor_data, fall_windows=[], title="Inference Data Visualization")
  
if __name__ == "__main__":
  inference()