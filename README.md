# Image Classification using Convolutional Neural Networks (CNN)

This project demonstrates image classification using Convolutional Neural Networks (CNNs) in TensorFlow. The goal is to classify images from the CIFAR-10 dataset into predefined categories.

## Features

- Data preprocessing: Load and preprocess the CIFAR-10 dataset.
- Model creation: Create a CNN model for image classification.
- Data augmentation: Apply data augmentation techniques to enhance training.
- Model training: Train the model using batch training and early stopping.
- Model evaluation: Evaluate the trained model's performance using various metrics and visualizations.

## Getting Started

1. Install the required packages by running:

   ```bash
   pip install -r requirements.txt
Run the main.py script to train and evaluate the image classification model:

bash
Copy code
python main.py
Project Structure
data_preprocessing.py: Load and preprocess the CIFAR-10 dataset, and apply data augmentation.
model_creation.py: Define the CNN model architecture.
training.py: Train the model using batch training and early stopping.
evaluation.py: Evaluate the trained model's performance using metrics and visualizations.
main.py: Main script to execute the complete image classification process.
requirements.txt: List of required packages for this project.
Results
After running the main.py script, the trained model will be saved as image_classification_model.h5. The script will also display the test accuracy, a classification report with precision, recall, and F1-score for each class, and a confusion matrix visualization.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The CIFAR-10 dataset is used for educational purposes.
