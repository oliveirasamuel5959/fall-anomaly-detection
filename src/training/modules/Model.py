import h5py
import json
import tensorflow as tf

print(tf.__version__)

def build_model(model_name="LSTM", learning_rate=0.001, X_train=None, y_train=None):
  # Define the input layer
  input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

  # Define encoder layers
  # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True))(input_layer)
  x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(input_layer)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(0.2)(x)

  # Define decode layers
  # x = tf.keras.layers.Dense(16, activation='relu')(x)
  # x = tf.keras.layers.BatchNormalization()(x)
  # x = tf.keras.layers.Dropout(0.2)(x)

  output = tf.keras.layers.Dense(y_train.shape[1], activation='softmax')(x)

  # Define LSTM model
  lstm_model = tf.keras.models.Model(input_layer, output, name=f"{model_name}_Fall_Detection")
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
  lstm_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy']
  )

  return lstm_model

def load_model(model_path):
  with h5py.File(model_path, 'r+') as f:
    model_config = json.loads(f.attrs['model_config'])
    
    def remove_key(obj, key):
      if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
          remove_key(v, key)
      elif isinstance(obj, list):
        for item in obj:
          remove_key(item, key)
    
    remove_key(model_config, 'quantization_config')
    f.attrs['model_config'] = json.dumps(model_config)
  
  return tf.keras.models.load_model(model_path, compile=False)