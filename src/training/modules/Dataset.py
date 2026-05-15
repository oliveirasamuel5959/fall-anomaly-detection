import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from tqdm import tqdm

def list_subject_data(subject, DATA_DIR):
  data_dir = os.path.join(DATA_DIR, subject)
  files = os.listdir(data_dir)

  # filter for files with .ini extension, which are invalid activity codes
  for f in files:
    if f.split(".")[1] == 'ini':
      # print(f"File {f} has an invalid activity code and will be skipped.")
      files.remove(f)
  return files

def load_fall_detection_data(subject=None, my_seed=42, DATA_DIR=None, ACTIVITY_CODES=None):
  # Define columns names
  cols = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "AccX2", "AccY2", "AccZ2"]
  data_df_list = []

  subject_data_list = list_subject_data(subject=subject, DATA_DIR=DATA_DIR)

  data_shuffled = shuffle(subject_data_list, random_state=my_seed)

  pbar = tqdm(range(len(data_shuffled)), desc='STARTING')

  for idx in pbar:
    # Get raw data from subject
    raw_data = data_shuffled[idx]

    # Get activity / repetition and split caracters
    activity_code = raw_data.split("_")[0]
    repetition = raw_data.split("_")[2].split(".")[0]

    # filter classes
    if activity_code not in ACTIVITY_CODES.keys():
      continue

    # Assign variables to progress bar
    pbar.set_description(f"LOADING {subject} DATA: CODE {activity_code} - REPETITION {repetition}")

    # Define waw data absolute path for pd reading csv
    raw_data_path = os.path.join(DATA_DIR, f"{subject}/{raw_data}")

    # print(f"Reading dataset {raw_data.split(".txt")[0]}")
    # Read dataset to pandas Dataframe
    df = pd.read_csv(raw_data_path, header=None, names=cols)
    df.drop(columns=df.columns[-3:], inplace=True)

    # Create two more columns for activity code and repetition
    df["activity_code"] = activity_code
    df["repetition"] = repetition

    # Append Dataframe to the list of Dataframe
    data_df_list.append(df)

  # Concatenate Dataframes to a unique full Dataframe
  full_df = pd.concat(data_df_list, ignore_index=True)

  return full_df

def load_train_data(my_seed=42, DATA_DIR=None, ACTIVITY_CODES=None):
  # Load train data
  data_01 = load_fall_detection_data('SA01', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  data_02 = load_fall_detection_data('SA02', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  data_03 = load_fall_detection_data('SA03', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  # data_04 = load_fall_detection_data('SA04', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  # data_05 = load_fall_detection_data('SA05', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  
  train_df = pd.concat([data_01, data_02, data_03], ignore_index=True)
  train_df['label'] = np.where(train_df['activity_code'].str.startswith('D'),'Normal','Fall')
  
  # Convert object data to string
  train_df['activity_code'] = train_df['activity_code'].astype('string')
  train_df['repetition'] = train_df['repetition'].astype('string')
  train_df['label'] = train_df['label'].astype('string')
  
  return train_df
  
def load_val_data(my_seed=42, DATA_DIR=None, ACTIVITY_CODES=None):
  # Load validation data
  data_09 = load_fall_detection_data('SA09', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  data_10 = load_fall_detection_data('SA10', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  
  val_df = pd.concat([data_09, data_10], ignore_index=True)
  val_df['label'] = np.where(val_df['activity_code'].str.startswith('D'),'Normal','Fall')
  
  val_df['activity_code'] = val_df['activity_code'].astype('string')
  val_df['repetition'] = val_df['repetition'].astype('string')
  val_df['label'] = val_df['label'].astype('string')
  
  return val_df
  
def load_test_data(my_seed=42, DATA_DIR=None, ACTIVITY_CODES=None):
  # Load test data
  data_06 = load_fall_detection_data('SA06', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  data_07 = load_fall_detection_data('SA07', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  # data_08 = load_fall_detection_data('SA08', my_seed=my_seed, DATA_DIR=DATA_DIR, ACTIVITY_CODES=ACTIVITY_CODES)
  
  test_df = pd.concat([data_06, data_07], ignore_index=True)
  test_df['label'] = np.where(test_df['activity_code'].str.startswith('D'),'Normal','Fall')
  
  test_df['activity_code'] = test_df['activity_code'].astype('string')
  test_df['repetition'] = test_df['repetition'].astype('string')
  test_df['label'] = test_df['label'].astype('string')
  
  return test_df