import requests
import time
import numpy as np

URL = "http://192.168.0.11/get?accX&accY&accZ&gyroX&gyroY&gyroZ"

def fetch_latest():
  r = requests.get(URL).json()["buffer"]
  
  def safe_get(key):
    if key in r and len(r[key]["buffer"]) > 0:
      return r[key]["buffer"][-1]
    return 0.0  # fallback (keeps schema consistent)

  return {
    "timestamp": time.time(),
    "AccX": safe_get("accX"),
    "AccY": safe_get("accY"),
    "AccZ": safe_get("accZ"),
    "GyroX": safe_get("gyroX"),
    "GyroY": safe_get("gyroY"),
    "GyroZ": safe_get("gyroZ"),
  }
  