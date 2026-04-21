
import os
from xml.parsers.expat import model

import tensorflow as tf
import numpy as np
import random
import json
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from pathlib import Path

from modules.Dataset import load2_fall_detection_data
from modules.Model import build_model
from modules.Plot import plot_activity, plot_data_distribution
from modules.FeatureEng import sampling_rate, convert_units
from modules.TimeSeriesData import create_adl_fault_to_repetition
from modules.Metrics import save_history_and_plots
from modules.Metrics import plot_confusion_matrix
from modules.Metrics import compute_metrics_and_confmat
from modules.Metrics import save_metrics
from modules.Metrics import save_model_architecture

from .utils.utils import create_output_dir
from .utils.labels import ACTIVITY_CODES

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
    
    # Load dataset
    df_data = load2_fall_detection_data('SA01', dataset_root)
    
    df_data = sampling_rate(df_data)
    df_data = convert_units(df_data)
    
    # Build model
    print(df_data.head())
    
    # plot_data_distribution(df_data, title="Data Distribution for SA01")
    
    # num_samples = len(df_data)
    # plot_activity(df=df_data, activity='D05', from_df='features', num_samples=num_samples, repetition='R05')
    
    X, Y = create_adl_fault_to_repetition(df_data, w=window_size, s=stride)
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape}")
    
    # Normalize Data
    sc = StandardScaler()
    sc.fit(df_data.iloc[:,:6])
    
    X_sc = sc.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # One-hot encode labels
    enc = OneHotEncoder(sparse_output=False)
    enc.fit(Y.reshape(-1, 1))
    Y_enc = enc.transform(Y.reshape(-1, 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_sc, Y_enc, test_size=0.2, random_state=42)
    
    
    print("X_train model input shape: ", X_train.shape)
    print("y_train model input shape: ", y_train.shape)

    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)
    
    # Model training
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
        filepath = os.path.join(model_save_path, (
            f"lstm-epoch_{{epoch:02d}}_"
            f"valloss_{{val_loss:.4f}}_valacc_{{val_accuracy:.4f}}.h5"
        )),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    save_history_and_plots(history, output_root, prefix=model.name)
    save_model_architecture(model, output_root)
    
    # --------------------------------------------------------
    # Evaluate model
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