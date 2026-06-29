# Face Mask Detection using EfficientNetB3

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-lightgrey.svg)

## Project Overview
This project implements a robust deep learning model to detect whether a person is wearing a face mask or not. Utilizing **Transfer Learning** with the **EfficientNetB3** architecture, the model efficiently extracts high-level features to classify images accurately. This project demonstrates end-to-end machine learning pipeline skills, from data preprocessing and augmentation to model training, evaluation, and visualization.

## Key Features
* **Transfer Learning**: Leveraged a pre-trained **EfficientNetB3** model (ImageNet weights) to achieve high accuracy and reduce training time.
* **Data Augmentation**: Implemented extensive image augmentation (rotation, zooming, shifting, brightness adjustment, and horizontal flipping) using `ImageDataGenerator` to prevent overfitting and improve model generalization.
* **Stratified Data Splitting**: Ensured balanced class distribution across the training (70%), validation (15%), and testing (15%) sets using `scikit-learn`.
* **Early Stopping**: Integrated callbacks to halt training when validation accuracy stops improving, ensuring optimal model weights are restored and avoiding overfitting.
* **Comprehensive Evaluation**: Evaluated model performance using standard metrics including Accuracy, Loss, Precision, Recall, F1-Score, and a Confusion Matrix.

## Technologies & Libraries Used
* **Deep Learning Framework**: TensorFlow, Keras
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Data Visualization**: Matplotlib, Seaborn
* **Environment**: Jupyter / Google Colab

## Model Architecture
1. **Base Model**: `EfficientNetB3` (Include Top = False, Pooling = Max)
2. **Flatten Layer**: To convert the 2D feature maps into a 1D vector.
3. **Dense Layer**: 128 units with `ReLU` activation.
4. **Dropout Layer**: 30% dropout rate to mitigate overfitting.
5. **Dense Layer**: 64 units with `ReLU` activation.
6. **Output Layer**: Dense layer with `Softmax` activation (output units correspond to the number of classes).

**Optimizer**: Adam (Learning Rate = 0.001)  
**Loss Function**: Categorical Crossentropy  

## Dataset
The model is trained on a Face Mask Dataset (e.g., from Kaggle). The data pipeline handles loading image paths, assigning labels dynamically based on folder structures, and streaming images in batches (Batch size = 16) with target dimensions of 256x256 pixels.

## Results & Visualizations
The script automatically generates several visual insights:
- **Class Distribution**: Bar charts and pie charts showing the balance of the dataset.
- **Data Augmentation Previews**: A grid displaying sample augmented images from the training generator.
- **Training Curves**: Line plots for Training vs. Validation Accuracy and Loss.
- **Confusion Matrix**: A Seaborn heatmap detailing true positive/negative and false positive/negative predictions on the unseen test set.
- **Classification Report**: Precision, recall, and f1-score for each class.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Abdallahkamel1/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
   ```
3. Update the dataset path in `face_mask_detection.py`:
   ```python
   data = create_data_set("path/to/your/dataset")
   ```
4. Run the script:
   ```bash
   python face_mask_detection.py
   ```

## Saved Model
After successful training, the final model architecture and learned weights are saved as `face_mask.h5`, ready to be deployed in a web or mobile application for real-time inference.
