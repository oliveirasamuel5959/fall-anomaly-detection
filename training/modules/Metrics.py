import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics_and_confmat(y_true, y_pred, labels):
  labels = list(labels)

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

  # fig, ax = plt.subplots(figsize=(15, 15))
  # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

  # ax.set_title("Confusion Matrix")
  # ax.set_xlabel("Predicted labels")
  # ax.set_ylabel("True labels")

  # ax.set_xticks(np.arange(len(labels)) + 0.5)
  # ax.set_yticks(np.arange(len(labels)) + 0.5)

  # ax.set_xticklabels(labels, rotation=45, ha='right')
  # ax.set_yticklabels(labels)

  # ax.set_ylim(len(labels), 0)

  # plt.tight_layout()
  # plt.show()

  return {
      'accuracy': acc,
      'precision': prec,
      'recall': rec,
      'f1_score': f1,
      'sensitivity': sensitivity,
      'specificity': specificity,
      'true_negative_rate': tnr
  }, {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
  
  


def save_history_and_plots(history, output_dir, prefix):
  """

  Args:
      history (_type_): objeto retornado pelo model.fi
      output_dir (_type_): save directory
      prefix (_type_): ex.: "train" ou "val"
  """
  
  os.makedirs(output_dir, exist_ok=True)
  
  # ---------------------
  # History to Dataframe
  # ---------------------
  hist_df = pd.DataFrame(history.history)
  hist_df["epoch"] = hist_df.index + 1
  
  csv_path = os.path.join(output_dir, f"{prefix}_history.csv")
  hist_df.to_csv(csv_path, index=False)
  
  # ---------------------
  # Plot Accuracy
  # ---------------------
  plt.figure()
  plt.plot(hist_df["epoch"], hist_df["accuracy"], label="Train Accuracy")
  plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label="Validation Accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.grid(True)

  acc_plot_path = os.path.join(output_dir, f"{prefix}_accuracy_plot.png")
  plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()
  
  # ---------------------
  # Plot Loss
  # ---------------------
  plt.figure()
  plt.plot(hist_df["epoch"], hist_df["loss"], label="Train Loss")
  plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training and Validation Loss")
  plt.legend()
  plt.grid(True)

  loss_plot_path = os.path.join(output_dir, f"{prefix}_loss_plot.png")
  plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close() 
  
  print(f"History saved to {csv_path}")
  print(f"Accuracy plot saved to {acc_plot_path}")
  print(f"Loss plot saved to {loss_plot_path}")
  