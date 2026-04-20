import tensorflow as tf

from pathlib import Path

from modules.Looping import process_train

print('TensorFlow version:', tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print(f"GPUs available: {len(gpus)}")
else:
  print("No GPUs available.")
  

# Configurations
MODEL_NAME = 'LSTM'
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 10
SEED = 42

ROOT = Path.cwd().parent

OUTPUT_ROOT = ROOT / "results-fall-detection"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_ROOT = ROOT / "dataset_processing/data"

def main():
  dataset_root = Path(DATASET_ROOT)

  if not dataset_root.exists():
    print(f"Dataset root directory '{dataset_root}' does not exist.")
    return
  
  process_train(
    dataset_root=dataset_root, 
    output_root=OUTPUT_ROOT, 
    model_name=MODEL_NAME, 
    learning_rate=LEARNING_RATE, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    early_stop_patience=EARLY_STOP_PATIENCE, 
    my_seed=SEED
  )

if __name__ == "__main__":
  main()
