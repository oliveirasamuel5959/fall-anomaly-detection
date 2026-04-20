import numpy as np
from tqdm import tqdm

def sliding_window(df, w, s):
  X = []
  Y = []

  # Iterate through the rows of the Dataframe in steps of size s
  for i in tqdm(range(0, len(df) - w, s), desc='Creating sequences'):
    # Extract a window of width w from the Dataframe, starting at row i
    x = np.array(df.iloc[i:i+w,:6])
    # Extract the target value (activity_code) for the last row of the window
    y = np.array(df.iloc[i+w-1,6])
    # Append the window and target value to the X and Y lists
    X.append(x)
    Y.append(y)

  return np.array(X), np.array(Y)


def create_adl_fault_to_repetition(df, w, s):
  all_codes = df.activity_code.unique()
  all_repetitions = df.repetition.unique()

  X_all = []
  Y_all = []

  for code in all_codes:
    for repetition in all_repetitions:
      df_k = df[
          (df['activity_code'] == code) &
          (df['repetition'] == repetition)
      ]
      if len(df_k) < w:
          continue  # skip too short sequences

      x_temp, y_temp = sliding_window(df_k, w=w, s=s)

      if len(x_temp) == 0:
          continue

      X_all.append(x_temp)
      Y_all.append(y_temp)

  X = np.concatenate(X_all, axis=0)
  Y = np.concatenate(Y_all, axis=0)

  return X, Y