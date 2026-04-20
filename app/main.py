import time
import csv
import os
from collections import deque

from collector import fetch_latest
from preprocess import preprocess

FIELDS = ["timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
CSV_FILE = 'sensor-output/' + "sensor_data.csv"
WINDOW_SIZE = 100

# Create file with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

def append_to_csv(row):
  with open(CSV_FILE, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writerow(row)
    
buffer = deque(maxlen=WINDOW_SIZE)

while True:
  sample = fetch_latest()
  
  append_to_csv(sample)
  
  print(sample)
  
  # buffer.append(sample)

  # if len(buffer) == WINDOW_SIZE:
  #   X = preprocess(buffer)

  time.sleep(0.05)