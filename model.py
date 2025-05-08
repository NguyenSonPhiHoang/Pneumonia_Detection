import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def build_model(input_shape=(224, 224, 3), num_classes=1):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the convolutional base to prevent weights from being updated
    base_model.trainable = False

    # Build the model on top of VGG16
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile model with additional metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def load_trained_model(model_path='models/cnn_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        print(f"Could not load model from {model_path}. Building new model.")
        return build_model()