import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values to the range [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

def apply_data_augmentation(train_images):
    # Create an instance of the ImageDataGenerator with augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=20,      # Rotate images by up to 20 degrees
        width_shift_range=0.1,  # Shift images horizontally by up to 10%
        height_shift_range=0.1, # Shift images vertically by up to 10%
        horizontal_flip=True    # Flip images horizontally
    )
    datagen.fit(train_images)  # Fit the generator on the training data
    
    return datagen
