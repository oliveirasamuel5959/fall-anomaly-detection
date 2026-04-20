# Fall Detection using LSTM Neural Networks

## Project Overview

This project implements a **real-time fall detection system** using Long Short-Term Memory (LSTM) neural networks. The system analyzes accelerometer and gyroscope data from smartphone sensors to detect falls in real-time using the phyphox application for sensor data collection.

### Main Goal
To develop a machine learning model that accurately classifies human activities and detects falls using time-series sensor data from mobile devices, enabling real-time inference for fall detection applications.

---

## Project Structure

### 1. **Dataset Processing** (`dataset_processing/`)
Handles data preparation and feature engineering from raw sensor files.

- **`data_read.py`** - Reads raw sensor data from text files (format: `D##_SA##_R##.txt`)
- **`feature_eng.py`** - Extracts and engineers features from sensor signals (magnitude, statistical features, etc.)
- **`dp_utils/dataset-report.py`** - Generates dataset analysis and statistics
- **`data/`** - Contains raw sensor data organized by subject (SA01, SA02, etc.)
  - Each file contains accelerometer (X, Y, Z) and gyroscope readings
  - Multiple repetitions (R##) per device activity

### 2. **Model Training** (`training/`)
Implements the LSTM model training pipeline.

- **`train.py`** - Main training script that orchestrates the model training process
- **`modules/`** - Core training components:
  - **`Model.py`** - LSTM architecture definition
  - **`Dataset.py`** - Dataset loading and preprocessing for time-series data
  - **`FeatureEng.py`** - Feature normalization and scaling
  - **`TimeSeriesData.py`** - Time-series data windowing and sequencing
  - **`Looping.py`** - Training loop implementation
  - **`Metrics.py`** - Evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
  - **`Plot.py`** - Visualization utilities for training history and results

### 3. **Results** (`results-fall-detection/`)
Stores training outputs and metrics.

- **`LSTM_history.csv`** - Training history logs (loss, accuracy per epoch)

### 4. **Real-Time Inference App** (`app/`)
Mobile-friendly application for real-time fall detection using phyphox sensor data.

- **`main.py`** - Entry point for the inference application
- **`collector.py`** - Collects sensor data from phyphox
- **`preprocess.py`** - Preprocesses incoming sensor data to match training format
- **`sensor-output/`** - Stores collected sensor readings (CSV format)
  - Example files: `sensor_data_F1.csv`, `sensor_data_R2.csv`, `sensor_data_R3.csv`
- **`model/`** - Stores the trained LSTM model for inference

---

## Feature Engineering

The feature engineering pipeline extracts meaningful representations from raw 6-axis sensor data (3-axis accelerometer + 3-axis gyroscope):

### Key Features Extracted:
- **Statistical Features**: Mean, standard deviation, min, max of each axis
- **Signal Magnitude**: Combined magnitude from 3-axis accelerometer and gyroscope
- **Time-Domain Features**: Root mean square, variance, energy
- **Derived Features**: Gravity-corrected acceleration, rotation rates

### Processing Steps:
1. Read raw sensor signals from data files
2. Apply signal preprocessing (optional filtering)
3. Extract time-series windows (sliding windows)
4. Normalize features to zero-mean, unit-variance
5. Create labeled sequences for supervised learning

---

## Model: LSTM Neural Network

### Architecture:
- **Input**: Time-series windows of sensor data
- **Hidden Layers**: Stacked LSTM cells capturing temporal dependencies
- **Output**: Activity classification (Normal activity vs. Fall)

### Why LSTM?
LSTMs are ideal for this task because:
- They capture long-term temporal dependencies in sensor data
- They learn patterns of sequential accelerometer/gyroscope changes
- They handle variable-length sequences effectively

### Training Process:
1. Split dataset into training/validation/test sets
2. Normalize input features using training set statistics
3. Train LSTM model using categorical cross-entropy loss
4. Monitor validation metrics and apply early stopping
5. Evaluate on test set using standard ML metrics

---

## Metrics Analyzed

The model evaluation includes comprehensive metrics:

- **Accuracy**: Overall correct classification rate
- **Precision**: True positive rate among predicted positives (important to avoid false alarms)
- **Recall**: True positive rate among actual positives (important to catch actual falls)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of true/false positives and negatives
- **Training History**: Loss and accuracy curves across epochs

These metrics help assess:
- Model reliability for real-world deployment
- Trade-offs between detecting falls and avoiding false alarms
- Overall generalization performance

---

## Real-Time Inference Pipeline

### Workflow:
1. **Sensor Collection (Phyphox)**
   - User runs phyphox app on smartphone
   - App streams accelerometer and gyroscope data
   
2. **Data Collection (`app/collector.py`)**
   - Captures live sensor streams
   - Buffers data for processing
   - Saves to CSV files in `sensor-output/`

3. **Preprocessing (`app/preprocess.py`)**
   - Normalizes incoming data to match training format
   - Creates time-series windows
   - Applies same feature transformations as training

4. **Inference (`app/main.py`)**
   - Loads trained LSTM model
   - Runs preprocessing
   - Generates real-time predictions
   - Triggers fall alerts when detected

---

## Installation & Usage

### Requirements
- Python 3.8+
- TensorFlow/Keras for model training
- NumPy, Pandas for data processing
- Scikit-learn for metrics and preprocessing
- Phyphox app on smartphone for sensor data

### Dataset Processing
```python
python dataset_processing/data_read.py
python dataset_processing/feature_eng.py
```

### Model Training
```python
python training/train.py
```

### Real-Time Inference
```python
python app/main.py
```

---

## Data Format

### Raw Sensor Files
Located in `dataset_processing/data/SA##/`:
- Format: `D##_SA##_R##.txt`
- Content: Time-series accelerometer (X, Y, Z) and gyroscope readings

### Phyphox Sensor Output
Located in `app/sensor-output/`:
- Format: CSV files with columns for timestamp, acceleration, rotation
- Example: `sensor_data_F1.csv` (F1 = Fall 1 classification)

---

## Project Workflow

```
Raw Sensor Data
    ↓
Dataset Processing (Feature Engineering)
    ↓
Labeled Time-Series Data
    ↓
LSTM Model Training
    ↓
Trained Model + Metrics
    ↓
Real-Time Inference App (Phyphox Integration)
    ↓
Fall Detection & Alerts
```

---

## References

- **Phyphox**: Simple phone-based physics experiments - https://phyphox.org/
- **LSTM Networks**: Hochreiter & Schmidhuber (1997)
- **Time-Series Classification**: Goodfellow et al., Deep Learning textbook

---

## Author
Fall Detection Project - Computer Science

## License
See LICENSE file for details.
