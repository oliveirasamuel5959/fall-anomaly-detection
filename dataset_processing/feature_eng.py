import numpy as np
import pandas as pd

def convert_units(df):
  
    ADC_13_BITS = 8192.0
    ADC_14_BITS = 16384.0
    ADC_16_BITS = 65536.0
    RANGE_ACC = 16
    RANGE_GYRO = 2000
    
    df_copy = df.copy()

    acc_cols = ["AccX", "AccY", "AccZ"]
    gyro_cols = ["GyroX", "GyroY", "GyroZ"]

    # Convert accelerometer to g
    df_copy[acc_cols] = ((2 * RANGE_ACC) / (ADC_13_BITS)) * df_copy[acc_cols]

    # Convert gyroscope to rad/s
    df_copy[gyro_cols] = ((2 * RANGE_GYRO) / (ADC_16_BITS)) * df_copy[gyro_cols]
    # df[gyro_cols] = df[gyro_cols] * (180.0 / np.pi)

    return df_copy
  
  
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