import os
import pandas as pd

from tqdm import tqdm

def list_subject_data(subject, DATA_DIR):
  data_dir = os.path.join(DATA_DIR, subject)
  files = os.listdir(data_dir)
  return files

def load2_fall_detection_data(subject=None, DATA_DIR=None):
  cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "AccX2", "AccY2", "AccZ2"]
  data_df_list = []
  subject_data_list = list_subject_data(subject=subject, DATA_DIR=DATA_DIR)

  for idx in tqdm(range(len(subject_data_list)), desc=f"Loading data for subject {subject}"):
    raw_data = subject_data_list[idx]

    activity_code = raw_data.split("_")[0]
    repetition = raw_data.split("_")[2].split(".")[0]

    raw_data_path = os.path.join(DATA_DIR, f"{subject}/{raw_data}")

    df = pd.read_csv(raw_data_path, header=None, names=cols)
    df.drop(columns=df.columns[-3:], inplace=True)

    df["activity_code"] = activity_code
    df["repetition"] = repetition

    data_df_list.append(df)

  full_df = pd.concat(data_df_list, ignore_index=True)

  return full_df