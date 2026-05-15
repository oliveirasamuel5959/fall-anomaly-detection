# --------------------
# Imports
# --------------------
import time
import csv
import os
from pathlib import Path
from collections import deque

# --------------------
# Local imports
# --------------------
from collector import fetch_latest
from preprocess import preprocess

FIELDS = ["timestamp", "AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
WINDOW_SIZE = 100

ROOT = Path.cwd()
CSV_FILE = ROOT / "sensor_output" / "sensor_data.csv"

# Create directory if it doesn't exist
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

# Create file with header if it doesn't exist
if not os.path.exists(CSV_FILE):
  with open(CSV_FILE, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()

def append_to_csv(row):
  with open(CSV_FILE, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writerow(row)
    
# buffer = deque(maxlen=WINDOW_SIZE)

while True:
  data = fetch_latest()
  append_to_csv(data)
  print(data)
  
  # buffer.append(sample)

  # if len(buffer) == WINDOW_SIZE:
  #   X = preprocess(buffer)

  # 200Hz sample rate
  time.sleep(0.005)