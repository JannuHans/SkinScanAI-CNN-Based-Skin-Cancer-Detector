# SkinScanAI-CNN-Based-Skin-Cancer-Detector
A Convolutional Neural Network (CNN) based deep learning project that classifies skin lesions as benign or malignant using dermoscopic image data. This project demonstrates the practical application of deep learning in medical image classification.

📁 Dataset

Name: Melanoma Cancer Dataset

Source: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset

Total Images: ~13,000 dermoscopic images

Classes:

Benign

Malignant

🔖 Structure:

skin_cancer_dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
├── test/
│   ├── Benign/
│   └── Malignant/
└── valid/   ← (Created manually via train-validation split)
    ├── Benign/
    └── Malignant/

✨ Features

✅ Deep CNN model using Keras with TensorFlow backend

🧪 Image preprocessing and real-time data augmentation

⚖️ Handles class imbalance using class_weight

📈 Training curves, confusion matrix, and classification report

🚩 EarlyStopping and ReduceLROnPlateau for optimal training

💾 Model saved as .h5 for reuse or deployment

📊 Results

Metric

Value

Accuracy

87%

The model achieved balanced classification performance with strong generalization capability on unseen data.

📈 Visualizations

✅ Training vs Validation Accuracy

✅ Confusion Matrix Heatmap

All visualizations are generated during training inside the Colab notebook.

📌 Dependencies

Python 3.x

TensorFlow / Keras

scikit-learn

matplotlib, seaborn

Google Colab (recommended)

🛠️ Tech Stack

Language: Python

Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn

Tools: Google Colab, Jupyter Notebook

💡 Future Improvements

📦 Integrate EfficientNet / ResNet for transfer learning

🌟 Implement focal loss to handle imbalance better

🌍 Create a web interface using Streamlit or Flask

📊 Add lesion metadata for multimodal learning

