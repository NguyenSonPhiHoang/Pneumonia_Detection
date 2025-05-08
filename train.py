import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
from data_loader import load_dataset
from sklearn.utils.class_weight import compute_class_weight

tf.get_logger().setLevel('ERROR')

def train_model(epochs=30, batch_size=32, target_size=(224, 224)):
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    # Load dataset (only test_ds is needed from load_dataset)
    _, _, test_ds, class_names = load_dataset(batch_size, target_size)
    print("Class names:", class_names)

    # Use ImageDataGenerator for training and validation with enhanced augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'Data/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=123
    )

    val_generator = train_datagen.flow_from_directory(
        'Data/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=123
    )

    # Compute class weights
    labels = train_generator.labels
    normal_count = np.sum(labels == 0)
    pneumonia_count = np.sum(labels == 1)
    print(f"Normal samples: {normal_count}, Pneumonia samples: {pneumonia_count}")

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weight_dict = dict(enumerate(class_weights))
    class_weight_dict[0] = class_weight_dict[0] * 2.0
    print("Adjusted class weights:", class_weight_dict)

    # Build model
    model = build_model(input_shape=(*target_size, 3))

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_recall',
        patience=5,
        mode='max',
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/cnn_model.keras',
        monitor='val_recall',
        mode='max',
        save_best_only=True
    )

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        class_weight=class_weight_dict
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/plots/training_history.png')
    plt.close()

    # Evaluate on test set
    test_metrics = model.evaluate(test_ds, return_dict=True)
    test_loss = test_metrics['loss']
    test_acc = test_metrics['accuracy']
    test_precision = test_metrics['precision']
    test_recall = test_metrics['recall']
    test_auc = test_metrics['auc']
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}, Test precision: {test_precision:.4f}, Test recall: {test_recall:.4f}, Test AUC: {test_auc:.4f}")

    # Save model
    model.save('models/cnn_model.keras')

    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Model training completed and saved to models/cnn_model.keras")