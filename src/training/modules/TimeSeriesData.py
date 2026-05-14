import numpy as np
from tqdm import tqdm

def sliding_window(feature_columns, data_type, df, w, s):
  X = []
  Y = []

  # Iterate through the rows of the Dataframe in steps of size s
  for i in range(0, len(df) - w, s):
    # for training, create windows for features and labels
    if data_type == 'train':
      # Extract a window of width w from the Dataframe, starting at row i
      x = np.array(df.iloc[i:i+w,:len(feature_columns)])

      # for multiclass classification
      y = df.iloc[i:i+w,len(feature_columns)+2].mode()[0]
      # y = df.iloc[i+w-1, len(feature_columns) + 2]

      # Append the window and target value to the X and Y lists
      X.append(x)
      Y.append(y)

    else:
      # for inference, create windows for features only
      x = np.array(df.iloc[i:i+w,:len(feature_columns)])
      X.append(x)

  return np.array(X), np.array(Y)


def create_data_repetition(feature_columns, data_type, df, w, s):

  X_all = []
  Y_all = []

  if data_type == 'train':
    # get codes and repetitions list from df
    all_codes = df.activity_code.unique()
    all_repetitions = df.repetition.unique()

    # loop over codes and repetitions
    for code in all_codes:
      for repetition in all_repetitions:
        df_k = df[
          (df['activity_code'] == code) &
          (df['repetition'] == repetition)
        ]

        # if there is no data in filter, continue
        if len(df_k) < w:
          continue  # skip too short sequences

        # get windows and labels for each window
        x_temp, y_temp = sliding_window(feature_columns=feature_columns, data_type=data_type, df=df_k, w=w, s=s)

        # if no window, continue
        if len(x_temp) == 0:
          continue

        # append windows and labels to create train data
        X_all.append(x_temp)
        Y_all.append(y_temp)

    # concatenate all x_windows dataframes to a unique sequence dataframe
    X = np.concatenate(X_all, axis=0)

    # if training process, concatenate labels
    Y = np.concatenate(Y_all, axis=0)

    # return windows and labels
    return X, Y

  else:
    # check if there is enough data to create sequences for the given window_size
    if len(df) < w:
      raise ValueError(f"Dataframe with size: ({len(df)}) is too short to create windows with window_size: ({w}).")

    # get windows from dataframe
    x_temp, _ = sliding_window(feature_columns=feature_columns, data_type=data_type, df=df, w=w, s=s)

    # append windows and labels to create train data
    X_all.append(x_temp)

    # concatenate all x_windows dataframes to a unique sequence dataframe
    X = np.concatenate(X_all, axis=0)

    # return only features windows if inference process
    return X