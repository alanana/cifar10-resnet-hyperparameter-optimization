# CIFAR-10 Classification with ResNet and Hyperparameter Optimization

This project implements a ResNet-style architecture for classifying images from the CIFAR-10 dataset. It includes random search hyperparameter optimization to find the optimal model configuration.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different classes. This project aims to achieve high classification accuracy using:

- ResNet-style architecture with skip connections
- Random search for hyperparameter optimization
- Data augmentation techniques
- Learning rate scheduling

## Key Features

- **CNN Architecture**: Custom implementation of ResNet-style architecture with residual connections
- **Hyperparameter Optimization**: Random search across multiple parameters including:
  - Learning rates (log-uniform distribution between 0.0001 and 0.0005)
  - Batch sizes (128, 256)
  - Network depth (3-5 layers)
  - Units per layer (512, 768, 1024)
  - Optimizers (Adam, AdamW)
  - Regularization techniques (L2, dropout)
  - Learning rate schedules (cosine decay, exponential decay)
- **Data Augmentation**: Applied rotation, shifting, flipping, and zoom to prevent overfitting
- **Comprehensive Analysis**: Visualization of hyperparameter impact on model performance

## Results

The best model configuration achieved significant accuracy on the CIFAR-10 test set with the following parameters:

========== BEST MODEL ==========
Test Accuracy: 0.9191

Best Hyperparameters:
- batch_size: 128
- dropout_rate: 0.3
- epochs: 40
- l2_reg: 0.0005
- learning_rate: 0.0002920517093676461
- learning_rate_decay: 0.0
- learning_rate_schedule: exponential_decay
- momentum: 0.9
- num_layers: 4
- optimizer: adam
- units_per_layer: 1024

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.95      0.93      1000
           1       0.91      0.98      0.95      1000
           2       0.93      0.87      0.90      1000
           3       0.81      0.87      0.84      1000
           4       0.94      0.92      0.93      1000
           5       0.94      0.81      0.87      1000
           6       0.91      0.97      0.94      1000
           7       0.93      0.95      0.94      1000
           8       0.97      0.93      0.95      1000
           9       0.94      0.94      0.94      1000

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000


## Repository Structure

- `CNN_RestNet_CIFAR10_Implementation.ipynb`: Main Jupyter notebook containing the entire workflow
- `README.md`: Project documentation

## Requirements

- TensorFlow 2.12.0
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow Addons

## How to Run

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install tensorflow==2.12.0 tensorflow-addons matplotlib scikit-learn scipy
   ```
3. Open and run the Jupyter notebook:
   ```
   jupyter notebook 03_CNN_Assessment_after_RestNet_RandomSearch.ipynb
   ```

## Future Improvements

Potential areas for enhancement:
1. Further optimization with focused search around best parameters
2. More aggressive data augmentation techniques
3. Experimenting with deeper ResNet architectures
4. Transfer learning from pre-trained models
5. Ensemble methods combining top-performing models

## License

This project is licensed under the MIT License.

## Author

Allan Lukyamuzi
