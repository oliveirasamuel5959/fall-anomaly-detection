import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .utils.labels import ACTIVITY_CODES

# ------------------------------------
# Save model architecture to text file
# ------------------------------------
def save_model_architecture(model, output_dir):
  summary_path = output_dir / f"model-architecture.txt"

  with open(summary_path, "w") as f:
      model.summary(print_fn=lambda x: f.write(x + "\n"))

  print(f"[OK] Model architecture saved to: {summary_path}")

# ------------------------------------
# Compute metrics and confusion matrix
# ------------------------------------
def compute_metrics_and_confmat(y_true, y_pred):
  # y_pred = (y_probs >= 0.5).astype(int)
  acc = float(accuracy_score(y_true, y_pred))
  prec = float(precision_score(y_true, y_pred, average='macro'))
  rec = float(recall_score(y_true, y_pred, average='macro'))
  f1 = float(f1_score(y_true, y_pred, average='macro'))
  cm = confusion_matrix(y_true, y_pred)
  tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
  tnr = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
  sensitivity = float(tp / (tp + fn))
  specificity = float(tn / (tn + fp))

  return acc, prec, rec, f1, tnr, cm
  
  
# ------------------------------------
# save_metrics to json
# ------------------------------------
def save_metrics(metrics_report, output_dir):
  """
  metrics_dict: dict com accuracy, precision, etc (valores escalares)
  output_dir : pasta do fold
  stage_name : "stage1" ou "stage2"
  """
  path = os.path.join(output_dir, f"metrics.json")
  with open(path, "w") as f:
      json.dump(metrics_report, f, indent=4)

  print(f"[OK] Métricas salvas em {path}")
  

# ------------------------------------
# Plot and save confusion matrix
# ------------------------------------
def plot_confusion_matrix(cm, labels=ACTIVITY_CODES.keys(), output_dir=None):
  fig, ax = plt.subplots(figsize=(15, 15))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

  ax.set_title("Confusion Matrix")
  ax.set_xlabel("Predicted labels")
  ax.set_ylabel("True labels")

  ax.set_xticks(np.arange(len(labels)) + 0.5)
  ax.set_yticks(np.arange(len(labels)) + 0.5)

  ax.set_xticklabels(labels, rotation=45, ha='right')
  ax.set_yticklabels(labels)

  ax.set_ylim(len(labels), 0)

  plt.tight_layout()

  img_path = os.path.join(
    output_dir,
    f"confusion_matrix.png"
  )

  plt.savefig(img_path, dpi=300, bbox_inches="tight")
  plt.show()
  plt.close()
  
  print(f"[OK] Confusion matrix saved to {img_path}")

# ----------------------------------
# Plot and save history and plots
# ----------------------------------
def save_history_and_plots(history, output_dir, prefix):
  """

  Args:
      history (_type_): objeto retornado pelo model.fit
      output_dir (_type_): save directory
      prefix (_type_): ex.: "train" ou "val"
  """

  # os.makedirs(output_dir, exist_ok=True)

  # ---------------------
  # History to Dataframe
  # ---------------------
  hist_df = pd.DataFrame(history.history)
  hist_df["epoch"] = hist_df.index + 1

  csv_path = os.path.join(output_dir, f"{prefix}_history.csv")
  hist_df.to_csv(csv_path, index=False)

  # ---------------------
  # Accuracy plot
  # ---------------------
  fig_acc, ax_acc = plt.subplots(figsize=(7, 5))

  ax_acc.plot(hist_df["epoch"], hist_df["accuracy"], label="Train Accuracy")

  if "val_accuracy" in hist_df:
    ax_acc.plot(hist_df["epoch"], hist_df["val_accuracy"], label="Validation Accuracy")

  ax_acc.set_xlabel("Epoch")
  ax_acc.set_ylabel("Accuracy")
  ax_acc.set_title("Training and Validation Accuracy")
  ax_acc.legend()
  ax_acc.grid(True)

  acc_plot_path = os.path.join(output_dir, f"acc_plot.png")
  fig_acc.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_acc)

  # ---------------------
  # Loss plot
  # ---------------------
  fig_loss, ax_loss = plt.subplots(figsize=(7, 5))

  ax_loss.plot(hist_df["epoch"], hist_df["loss"], label="Train Loss")

  if "val_loss" in hist_df:
    ax_loss.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss")

  ax_loss.set_xlabel("Epoch")
  ax_loss.set_ylabel("Loss")
  ax_loss.set_title("Training and Validation Loss")
  ax_loss.legend()
  ax_loss.grid(True)

  loss_plot_path = os.path.join(output_dir, f"loss_plot.png")
  fig_loss.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close(fig_loss)

  print()
  print(f"[OK] History saved to {csv_path}")
  print(f"[OK] Accuracy plot saved to {acc_plot_path}")
  print(f"[OK] Loss plot saved to {loss_plot_path}")
  