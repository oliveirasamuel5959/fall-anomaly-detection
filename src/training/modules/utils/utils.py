import os

# ------------------------------
# Define train output directory
# ------------------------------
def create_output_dir(root_dir):
  # Get all matching files
  existing_files = list(root_dir.glob("train*"))

  # Extract existing numbers
  train_nums = []

  for f in existing_files:
    name = f.stem  # e.g., "train01_variables"
    try:
      num = int(name.split("0")[1].replace("train", ""))
      print(num)
      train_nums.append(num)
    except ValueError:
      continue

  # Determine next number
  next_num = max(train_nums, default=0) + 1

  # Create new folder for train
  OUTPUT_TRAIN_PATH = root_dir / f"train{next_num:02d}"
  os.makedirs(OUTPUT_TRAIN_PATH, exist_ok=True)
  
  return OUTPUT_TRAIN_PATH