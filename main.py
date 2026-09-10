from data_preprocessing import load_and_preprocess_data, apply_data_augmentation
from model_creation import create_cnn_model
from training import train_model
from evaluation import evaluate_model


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    
    model = create_cnn_model(input_shape=train_images.shape[1:], num_classes=10)
    
    epochs = 20
    
    datagen = apply_data_augmentation(train_images)
    train_generator = datagen.flow(train_images, train_labels, batch_size=64)  # Adjust batch size as needed
    
    
    history = train_model(model, train_generator, test_images, test_labels, epochs)

    evaluate_model(model, test_images, test_labels, class_names=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

    model.save('image_classification_model.h5')