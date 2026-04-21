import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

from .utils.labels import ACTIVITY_CODES

def plot_data_distribution(data_df, title="Data Distribution"):
    plt.figure(figsize=(10, 6))
    sns.countplot(
      data=data_df,
      x="activity_code",
      order=data_df["activity_code"].value_counts().index,
      palette="Set2"
    )
    plt.title(title)
    plt.xlabel("Activity Code")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    

def plot_data_repetition_distribution(data_df, title="Data Repetition Distribution"):
    plt.figure(figsize=(10, 6))
    sns.countplot(
      data=data_df,
      x="repetition",
      order=data_df["repetition"].value_counts().index,
      palette="Set2"
    )
    plt.title(title)
    plt.xlabel("Repetition")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()    

def plot_activity(df, activity, from_df='features', num_samples=600, repetition=None):

    if from_df == 'features':
      if repetition:
        df_k = df[(df['activity_code'] == activity) & (df['repetition'] == repetition)][:num_samples]
        x_axis = df_k.timestamp
      else:
        df_k = df[df['activity_code'] == activity][:num_samples]
        x_axis = df_k.timestamp
    else:
      df_k = df.copy()
      x_axis = df_k.samples

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    # ---- LEFT: ACCEL ----
    for col in ["AccX", "AccY", "AccZ"]:
        axes[0].plot(x_axis, df_k[col], label=col)

    axes[0].set_xlim(xmin=x_axis.min(), xmax=x_axis.max())
    axes[0].set_title("Acceleration")
    axes[0].set_xlabel("timestamp (s)")
    axes[0].set_ylabel("g")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # ---- RIGHT: GYRO ----
    for col in ["GyroX", "GyroY", "GyroZ"]:
        axes[1].plot(x_axis, df_k[col], label=col)

    axes[1].set_xlim(xmin=x_axis.min(), xmax=x_axis.max())
    axes[1].set_title("Gyroscope")
    axes[1].set_xlabel("timestamp (s)")
    axes[1].set_ylabel("°/s")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # ---- MAIN TITLE ----
    plt.suptitle(f"{activity} - {ACTIVITY_CODES.get(activity)}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()