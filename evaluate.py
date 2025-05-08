import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from model import load_trained_model
from data_loader import load_dataset

def evaluate_model(model_path='models/cnn_model.keras', batch_size=32, target_size=(224, 224)):
    output_dir = 'outputs/result'
    os.makedirs(output_dir, exist_ok=True)

    model = load_trained_model(model_path)
    _, _, test_ds, class_names = load_dataset(batch_size, target_size)

    metrics = model.evaluate(test_ds, return_dict=True)
    print("Evaluation metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    y_true = []
    y_scores = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_scores.extend(predictions.flatten())

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Use a custom threshold to prioritize recall
    custom_threshold = 0.3  # Giảm ngưỡng từ 0.5 xuống 0.3 để tăng recall
    print(f"Custom threshold for evaluation: {custom_threshold:.4f}")

    with open(os.path.join(output_dir, 'optimal_threshold.txt'), 'w') as f:
        f.write(str(custom_threshold))

    y_pred = (np.array(y_scores) > custom_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    print(f"Evaluation results saved to '{output_dir}'")
    return cm, report, roc_auc

if __name__ == "__main__":
    evaluate_model()