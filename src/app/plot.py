import csv
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
  "axes.facecolor":   "#f5f5f5",
  "figure.facecolor": "white",
  "axes.edgecolor":   "#cccccc",
  "grid.color":       "#e2e2e2",
  "grid.linestyle":   "--",
  "grid.linewidth":   0.6,
  "axes.grid":        True,
})

def plot_inference(data_df, fall_windows=None, title="Inference Data Visualization"):
  # Relative time axis (seconds from first sample)
  time = data_df["timestamp"] - data_df["timestamp"].iloc[0]
  
  fig = plt.figure(figsize=(14, 7), facecolor="white")
  fig.suptitle(title, fontsize=14, fontweight="bold", color="#222222")
  
  gs = gridspec.GridSpec(2, 1, hspace=0.4, left=0.07, right=0.97, top=0.92, bottom=0.08)
  
  ax_acc  = fig.add_subplot(gs[0])
  ax_gyro = fig.add_subplot(gs[1])
  
  # Accelerometer
  ax_acc.plot(data_df.timestamp, data_df["AccX"], color="#e63946", lw=1.2, label="AccX")
  ax_acc.plot(data_df.timestamp, data_df["AccY"], color="#2a9d8f", lw=1.2, label="AccY")
  ax_acc.plot(data_df.timestamp, data_df["AccZ"], color="#457b9d", lw=1.2, label="AccZ")
  ax_acc.set_title("Accelerometer (m/s²)", color="#444444", fontsize=10)
  ax_acc.set_ylabel("m/s²")
  ax_acc.legend(loc="upper right", fontsize=8, framealpha=0.7)
  
  # Gyroscope
  ax_gyro.plot(data_df.timestamp, data_df["GyroX"], color="#e9822c", lw=1.2, label="GyroX")
  ax_gyro.plot(data_df.timestamp, data_df["GyroY"], color="#7b2d8b", lw=1.2, label="GyroY")
  ax_gyro.plot(data_df.timestamp, data_df["GyroZ"], color="#f4a261", lw=1.2, label="GyroZ")
  ax_gyro.set_title("Gyroscope (rad/s)", color="#444444", fontsize=10)
  ax_gyro.set_ylabel("rad/s")
  ax_gyro.set_xlabel("Time (s)")
  
  
  # Paint fall windows
  if fall_windows:
    for i, (t_start, t_end) in enumerate(fall_windows):
      ax_acc.axvspan(t_start, t_end, color="red", alpha=0.15, label="Fall detected" if i == 0 else None)
      ax_gyro.axvspan(t_start, t_end, color="red", alpha=0.15)

  ax_acc.legend(loc="upper right", fontsize=8, framealpha=0.7)
  ax_gyro.legend(loc="upper right", fontsize=8, framealpha=0.7)
  
  n = len(data_df)
  duration = time.iloc[-1]
  fig.text(
    0.5, 
    0.01, 
    f"{n:,} samples  |  duration: {duration:.2f}s  |  sampling rate: {n/duration:.2f} Hz",
    ha="center",
    fontsize=8, 
    color="#888888"
  )
 
  plt.show()