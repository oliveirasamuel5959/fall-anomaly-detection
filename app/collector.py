import requests
import time
import numpy as np


URL = "http://192.168.0.15/get?accX&accY&accZ&gyroX&gyroY&gyroZ"

def fetch_latest():
  r = requests.get(URL).json()["buffer"]
  
  def safe_get(key):
    if key in r and len(r[key]["buffer"]) > 0:
      return r[key]["buffer"][-1]
    return 0.0  # fallback (keeps schema consistent)

  return {
    "timestamp": time.time(),
    "accX": safe_get("accX"),
    "accY": safe_get("accY"),
    "accZ": safe_get("accZ"),
    "gyroX": safe_get("gyroX"),
    "gyroY": safe_get("gyroY"),
    "gyroZ": safe_get("gyroZ"),
  }
  