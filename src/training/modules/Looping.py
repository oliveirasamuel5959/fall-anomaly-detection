
import os
import time
from xml.parsers.expat import model

import tensorflow as tf
import numpy as np
import random
import json
import math
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


from src.training.modules.Dataset import load_fall_detection_data
from src.training.modules.Dataset import load_train_data
from src.training.modules.Dataset import load_val_data
from src.training.modules.Dataset import load_test_data
from src.training.modules.Model import build_model

from src.training.modules.Plot import plot_activity
from src.training.modules.Plot import plot_data_distribution

# from src.training.modules.FeatureEng import sampling_rate
# from src.training.modules.FeatureEng import convert_units
from src.training.modules.FeatureEng import feature_engineering
from src.training.modules.TimeSeriesData import create_data_repetition
from src.training.modules.Metrics import save_history_and_plots
from src.training.modules.Metrics import plot_confusion_matrix
from src.training.modules.Metrics import compute_metrics_and_confmat
from src.training.modules.Metrics import save_metrics
from src.training.modules.Metrics import save_model_architecture

from src.training.modules.utils.labels import ACTIVITY_CODES
from src.training.modules.utils.utils import create_output_dir

from src.training.modules.Processing import DataPreprocessor

# ---------------------
# Training loop
# ---------------------
def process_train(
    dataset_root, 
    output_root, 
    model_name="LSTM",
    window_size=100,
    stride=20, 
    learning_rate=0.001, 
    epochs=100, 
    batch_size=64, 
    early_stop_patience=10, 
    my_seed=42):
    
    tf.keras.utils.set_random_seed(my_seed)
    tf.random.set_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # ---------------------
    # Load dataset
    # ---------------------
    train_df = load_train_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    val_df = load_val_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    test_df = load_test_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    
    # ---------------------
    # Feature engineering 
    # and data preparation
    # ---------------------
    FEATURE_COLUMNS, train_df, val_df, test_df = feature_engineering(train_df, val_df, test_df, method="base")

    # ----------------------
    # Create window sequences for 
    # train, val and test data
    # ----------------------
    X_train_seq, y_train_seq = create_data_repetition(data_type="train", df=train_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)
    X_val_seq, y_val_seq = create_data_repetition(data_type="val", df=val_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)
    X_test_seq, y_test_seq = create_data_repetition(data_type="test", df=test_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)
    
    # -----------------------
    # Print shapes and class 
    # distribution of window sequences
    # -----------------------
    print("X_train window sequences shape: ", X_train_seq.shape)
    print("y_train window sequences shape: ", y_train_seq.shape)
    print("X_val window sequences shape: ", X_val_seq.shape)
    print("y_val window sequences shape: ", y_val_seq.shape)
    print("X_test window sequences shape: ", X_test_seq.shape)
    print("y_test window sequences shape: ", y_test_seq.shape)
    
    print(np.unique(y_train_seq, return_counts=True))
    print(np.unique(y_val_seq, return_counts=True))
    print(np.unique(y_test_seq, return_counts=True))
    
    # -----------------------
    # Shuffle training data
    # -----------------------
    idx = np.random.permutation(len(X_train_seq))

    X_train_seq = X_train_seq[idx]
    y_train_seq = y_train_seq[idx]
    
    # -----------------------
    # Normalize Data
    # -----------------------
    
    # Initialize preprocessor and fit on training data
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train = preprocessor.fit_transform_scaler(X_train_seq)
    
    # Transform validation and test data using the same scaler
    X_test = preprocessor.transform_scaler(X_test_seq)
    X_val = preprocessor.transform_scaler(X_val_seq)
    
    # ------------------------
    # One-hot encode labels
    # ------------------------
    y_train = preprocessor.fit_transform_encoder(y_train_seq)
    y_val = preprocessor.transform_encoder(y_val_seq)
    y_test = preprocessor.transform_encoder(y_test_seq)
    
    # ------------------------
    # MODEL BUILDING
    # ------------------------
    model = build_model(model_name=model_name, learning_rate=learning_rate, X_train=X_train, y_train=y_train)
    print(model.summary())
    
    # ---------------------------------------------
    # Define output directory for this training run
    # ---------------------------------------------
    output_dir = create_output_dir(output_root)
    model_save_path = os.path.join(output_dir, f"saved_models")
    os.makedirs(model_save_path, exist_ok=True)
    
    # ---------------------------------------------
    # Model training with checkpoint and early stopping
    # ---------------------------------------------
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = output_dir / model_save_path / (
        f"lstm-epoch_{{epoch:02d}}_"
        f"valloss_{{val_loss:.4f}}_valacc_{{val_accuracy:.4f}}.h5"
    ),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=15,
        min_lr=1e-5
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    time_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler, checkpoint_callback]
    )

    time_end = time.time()
    training_time = time_end - time_start
    print(f"Training time: {training_time:.2f} seconds")
    
    save_history_and_plots(history, output_root, prefix=model.name)
    save_model_architecture(model, output_root)
    
    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------
    y_probs = model.predict(X_test, verbose=0)
    y_true_cls = np.argmax(y_test, axis=1)
    y_pred_cls = np.argmax(y_probs, axis=1)
    
    acc, prec, rec, f1, tnr, cm = compute_metrics_and_confmat(y_true=y_true_cls, y_pred=y_pred_cls)
    
    metrics_report = {
        "accuracy": acc,
        "precision": prec,
        "true-negative-rate": tnr,
        "recall": rec,
        "f1-score": f1
    }
    
    save_metrics(metrics_report, output_dir=output_root)
    plot_confusion_matrix(cm, labels=ACTIVITY_CODES.keys(), output_dir=output_root)