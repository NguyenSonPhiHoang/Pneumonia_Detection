import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array

def load_dataset(batch_size=32, target_size=(224, 224)):
    train_dir = "Data/train"
    test_dir = "Data/test"

    # Load train dataset and get class_names before transformations
    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary'
    )
    class_names = train_ds_raw.class_names  # Lưu class_names trước khi biến đổi

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='binary'
    )

    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names  # Trả về class_names trực tiếp