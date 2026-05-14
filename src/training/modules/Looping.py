
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

from src.core.logger import get_logger, log_task

logger = get_logger(__name__)

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
    
    # -------------------------------------
    # Set random seeds for reproducibility
    # -------------------------------------
    tf.keras.utils.set_random_seed(my_seed)
    tf.random.set_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    
    # -------------------------------
    # Define dataset and output paths
    # -------------------------------
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # ---------------------
    # Load dataset
    # ---------------------
    logger.info("Loading dataset...")
    
    train_df = load_train_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    logger.info("Train data loaded successfully")
    
    val_df = load_val_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    logger.info("Validation data loaded successfully")
    
    test_df = load_test_data(my_seed=my_seed, DATA_DIR=dataset_root, ACTIVITY_CODES=ACTIVITY_CODES)
    logger.info("Test data loaded successfully")
    
    # ---------------------
    # Feature engineering 
    # and data preparation
    # ---------------------
    logger.info("Starting feature engineering and data preparation...")
    FEATURE_COLUMNS, train_df, val_df, test_df = feature_engineering(train_df, val_df, test_df, method="base")
    logger.info("Feature engineering completed successfully")

    # ----------------------
    # Create window sequences for 
    # train, val and test data
    # ----------------------
    with log_task("Creating window sequences for train", "Train window sequences created successfully"):
        X_train_seq, y_train_seq = create_data_repetition(data_type="train", df=train_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)
    
    with log_task("Creating window sequences for validation", "Validation window sequences created successfully"):
        X_val_seq, y_val_seq = create_data_repetition(data_type="train", df=val_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)
    
    with log_task("Creating window sequences for test", "Test window sequences created successfully"):
        X_test_seq, y_test_seq = create_data_repetition(data_type="train", df=test_df, w=window_size, s=stride, feature_columns=FEATURE_COLUMNS)

    logger.info("Window sequences created successfully")
    # -----------------------
    # Print shapes and class 
    # distribution of window sequences
    # -----------------------
    logger.info("X_train window sequences shape: %s", X_train_seq.shape)
    logger.info("y_train window sequences shape: %s", y_train_seq.shape)
    logger.info("X_val window sequences shape: %s", X_val_seq.shape)
    logger.info("y_val window sequences shape: %s", y_val_seq.shape)
    logger.info("X_test window sequences shape: %s", X_test_seq.shape)
    logger.info("y_test window sequences shape: %s", y_test_seq.shape)
    
    logger.info("y_train class distribution: %s", np.unique(y_train_seq, return_counts=True))
    logger.info("y_val class distribution: %s", np.unique(y_val_seq, return_counts=True))
    logger.info("y_test class distribution: %s", np.unique(y_test_seq, return_counts=True))
    
    # -----------------------
    # Shuffle training data
    # -----------------------
    idx = np.random.permutation(len(X_train_seq))

    X_train_seq = X_train_seq[idx]
    y_train_seq = y_train_seq[idx]
    logger.info("Training data shuffled successfully")
    # -----------------------
    # Normalize Data
    # -----------------------
    logger.info("Starting data normalization...")
    # Initialize preprocessor and fit on training data
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train = preprocessor.fit_transform_scaler(X_train_seq)
    
    # Transform validation and test data using the same scaler
    X_val = preprocessor.transform_scaler(X_val_seq)
    X_test = preprocessor.transform_scaler(X_test_seq)
    
    logger.info("Data normalization completed successfully")
    
    # ------------------------
    # One-hot encode labels
    # ------------------------
    y_train = preprocessor.fit_transform_encoder(y_train_seq)
    y_val = preprocessor.transform_encoder(y_val_seq)
    y_test = preprocessor.transform_encoder(y_test_seq)
    logger.info("One-hot encoding labels successfully")
    
    logger.info("Final shapes after preprocessing:")
    logger.info("X_train shape: %s", X_train.shape)
    logger.info("y_train shape: %s", y_train.shape)
    
    # ------------------------
    # MODEL BUILDING
    # ------------------------
    model = build_model(model_name=model_name, learning_rate=learning_rate, X_train=X_train, y_train=y_train)
    logger.info("Model built successfully")
    logger.info("Model summary:\n%s", model.summary())
    
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

    logger.info("Model training Start")
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
    logger.info("Training time: %.2f seconds", training_time)
    
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
    logger.info("Metrics saved successfully")
    plot_confusion_matrix(cm, labels=ACTIVITY_CODES.keys(), output_dir=output_root)