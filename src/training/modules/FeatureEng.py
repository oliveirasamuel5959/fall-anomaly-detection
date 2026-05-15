import numpy as np
import pandas as pd

ADC_13_BITS = 8192.0
ADC_16_BITS = 65536.0
RANGE_ACC = 16
RANGE_GYRO = 2000

def convert_sisfall_units(df):
    df_copy = df.copy()
    acc_cols = ["AccX", "AccY", "AccZ"]
    gyro_cols = ["GyroX", "GyroY", "GyroZ"]

    # Convert accelerometer to g
    df_copy[acc_cols] = ((2 * RANGE_ACC) / (ADC_13_BITS)) * df_copy[acc_cols]
    df_copy[gyro_cols] = ((2 * RANGE_GYRO) / (ADC_16_BITS)) * df_copy[gyro_cols]

    # Convert gyroscope to rad/s
    df_copy[gyro_cols] = df_copy[gyro_cols] * (np.pi / 180.0)
    return df_copy
  
def feature_c1c2(df):
  df_temp = df.copy()
  c1 = np.sqrt(df_temp["AccX"]**2 + df_temp["AccY"]**2 + df_temp["AccZ"]**2)
  c2 = np.sqrt(df_temp["AccX"]**2 + df_temp["AccZ"]**2)

  df_temp.insert(6, "C1", c1, True)
  df_temp.insert(7, "C2", c2, True)
  return df_temp
  
def sampling_rate(df):
  """
  Data acquisition sampling rate place at 200 Hz.

  A time window of 3s provides 600 data points.

  data_points = 3s / sampling_rate(s)

  """
  df_copy = df.copy()

  dt = 1 / 200

  df_copy["timestamp"] = np.arange(len(df_copy)) * dt

  return df_copy


def prepare_timestamp(data_df):
    df = data_df.copy()

    # Convert ms → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Ensure sorted (critical for plotting + resampling)
    df = df.sort_values("timestamp")

    return df
  
def feature_engineering(train_df, val_df, test_df, method="base"):
  # Featuring Engineering raw data to real physical values
  train_df = convert_sisfall_units(train_df)
  test_df = convert_sisfall_units(test_df)
  val_df = convert_sisfall_units(val_df)

  if method == "base":
    # time series samplig rate
    train_df = sampling_rate(train_df)
    val_df = sampling_rate(val_df)
    test_df = sampling_rate(test_df)
    # SA15_df = sampling_rate(SA15_df)

    # Define global feature columns
    FEATURE_COLUMNS = train_df.columns[:6].to_list()

  elif method == "c1_c2":
    # Calculate magnitudes of accel and rotations
    train_df = feature_c1c2(train_df)
    val_df = feature_c1c2(val_df)
    test_df = feature_c1c2(test_df)
    SA15_df = feature_c1c2(SA15_df)

    # time series samplig rate
    train_df = sampling_rate(train_df)
    val_df = sampling_rate(val_df)
    test_df = sampling_rate(test_df)
    # SA15_df = sampling_rate(SA15_df)

    # Define global feature columns
    FEATURE_COLUMNS = train_df.columns[:8].to_list()

  else:
    raise ValueError(f"Invalid feature engineering method: {method}")
  
  return FEATURE_COLUMNS, train_df, val_df, test_df

def rotation_matrix(df, theta, axis='z'):
  df_rot = df.copy()

  if axis == 'z':
    R = np.array([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta),  np.cos(theta), 0],
      [0,              0,             1]
    ])
  else:
    raise ValueError("Only 'z' implemented")

  # Apply separately to accel and gyro
  acc = df_rot[df.columns[1:4]].values
  gyro = df_rot[df.columns[4:]].values

  df_rot[df.columns[1:4]] = acc @ R.T
  df_rot[df.columns[4:]] = gyro @ R.T

  return df_rot