import os
import json
import pandas as pd

from tqdm import tqdm

COLUMNS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "AccX2", "AccY2", "AccZ2"]

def list_subject_data(subject, DATA_DIR):
  data_dir = os.path.join(DATA_DIR, subject)
  files = os.listdir(data_dir)
  return files

def load_fall_detection_data(subject=None, DATA_DIR=None):
  data_df_list = []
  subject_data_list = list_subject_data(subject=subject, DATA_DIR=DATA_DIR)

  for idx in tqdm(range(len(subject_data_list)), desc=f"Loading data for subject {subject}"):
    raw_data = subject_data_list[idx]

    activity_code = raw_data.split("_")[0]
    repetition = raw_data.split("_")[2].split(".")[0]

    raw_data_path = os.path.join(DATA_DIR, f"{subject}/{raw_data}")

    df = pd.read_csv(raw_data_path, header=None, names=COLUMNS)
    df.drop(columns=df.columns[-3:], inplace=True)

    df["activity_code"] = activity_code
    df["repetition"] = repetition
  
    data_df_list.append(df)

  ds_df = pd.concat(data_df_list, ignore_index=True)

  return ds_df