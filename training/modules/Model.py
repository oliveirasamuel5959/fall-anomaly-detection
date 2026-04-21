import tensorflow as tf

def build_model(model_name="LSTM", learning_rate=0.001, X_train=None, y_train=None):
  # Define the input layer
  input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

  # Define encoder layers
  encoded = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True))(input_layer)
  encoded = tf.keras.layers.LSTM(128, activation='tanh')(encoded)

  # Define decoders layers
  decoded = tf.keras.layers.Dense(300, activation='relu')(encoded)
  decoded = tf.keras.layers.Dropout(0.5)(decoded)
  decoded = tf.keras.layers.Dense(y_train.shape[1], activation='softmax')(decoded)

  # Define LSTM model
  lstm_model = tf.keras.models.Model(input_layer, decoded, name=f"{model_name}_Fall_Detection")
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
  lstm_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=['accuracy']
  )

  return lstm_model
  
print("Model module loaded successfully.")