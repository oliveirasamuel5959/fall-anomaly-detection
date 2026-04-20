
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

# Looping de treino

def process_train(
    dataset_root, 
    output_root, 
    model_name="LSTM", 
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
    
    num_samples = len(df_data)
    # plot_activity(df=df_data, activity='D05', from_df='features', num_samples=num_samples, repetition='R05')
    
    w = 100
    s = 20
    X, Y = create_adl_fault_to_repetition(df_data, w, s)
    
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
        callbacks=[early_stopping]
    )
    
    save_history_and_plots(history, output_root, prefix=model_name)
    
    