import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

ACTIVITY_CODES = {
    # Daily Activities (D)
    "D01": "Walking slowly",
    "D02": "Walking quickly",
    "D03": "Jogging slowly",
    "D04": "Jogging quickly",
    "D05": "Walking upstairs and downstairs slowly",
    "D06": "Walking upstairs and downstairs quickly",
    "D07": "Sit (half chair) slowly → wait → stand slowly",
    "D08": "Sit (half chair) quickly → wait → stand quickly",
    "D09": "Sit (low chair) slowly → wait → stand slowly",
    "D10": "Sit (low chair) quickly → wait → stand quickly",
    "D11": "Sit → attempt to stand → collapse into chair",
    "D12": "Sit → lie slowly → wait → sit again",
    "D13": "Sit → lie quickly → wait → sit again",
    "D14": "Supine → lateral → wait → supine",
    "D15": "Standing → bend knees slowly → stand up",
    "D16": "Standing → bend (no knees) → stand up",
    "D17": "Enter car → sit → exit car",
    "D18": "Stumble while walking",
    "D19": "Gentle jump (no fall)",

    # Falls (F)
    "F01": "Forward fall (slip while walking)",
    "F02": "Backward fall (slip while walking)",
    "F03": "Lateral fall (slip while walking)",
    "F04": "Forward fall (trip while walking)",
    "F05": "Forward fall (trip while jogging)",
    "F06": "Vertical fall (fainting while walking)",
    "F07": "Fall with hand support on table (fainting)",
    "F08": "Forward fall while getting up",
    "F09": "Lateral fall while getting up",
    "F10": "Forward fall while sitting down",
    "F11": "Backward fall while sitting down",
    "F12": "Lateral fall while sitting down",
    "F13": "Forward fall while sitting (faint/sleep)",
    "F14": "Backward fall while sitting (faint/sleep)",
    "F15": "Lateral fall while sitting (faint/sleep)"
}

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
    
    
import matplotlib.pyplot as plt

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