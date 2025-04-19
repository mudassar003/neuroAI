# Leveraging Machine Learning for Early Detection of Neurological Disorders

![GitHub stars](https://img.shields.io/github/stars/mudassar003/neuroAI?style=social)
![GitHub forks](https://img.shields.io/github/forks/mudassar003/neuroAI?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/mudassar003/neuroAI?style=social)

## 📌 Project Overview

This repository contains the implementation of my MS thesis research on leveraging machine learning for early detection of neurological disorders, specifically focusing on Parkinson's Disease (PD) classification using MRI scans. The project consists of two main components:

1. **Multiclass Classification of Parkinson's Disease Stages**: Unlike traditional binary classification approaches, this research implements a multi-class classification model that can identify three distinct stages:
   - Healthy Controls (HC)
   - Prodromal Stage (early-stage PD)
   - Parkinson's Disease (PD)

2. **Drug Property Prediction**: Analysis of molecular characteristics of Anti-Parkinson's drugs using topological indices and machine learning to predict physicochemical properties for drug development.

The classification model utilizes a hybrid approach combining ResNet50 and EfficientNetB0 architectures with a novel triangular dense layer structure to achieve high accuracy in classifying PD stages from MRI scans.

## 🔬 Research Abstract

Parkinson's Disease (PD) is a neurological disorder with multiple progressive stages. There are three main stages in this disease which are HC (Healthy Controls), Prodromal and PD (Parkinson Diseases). It's symptoms include stigma, worsening of cognitive function, and greater constraints in movement. Accurate stage identification and early diagnosis is essential for effective treatment, but current clinical methods face challenges at the early stages of disease.

This study aimed to propose deep learning models for multi classification of diseases and prediction of it's drugs properties for new drug's development. A new hybrid deep learning model designed to classify PD stages using MRI scans is developed. The hybrid classification model was trained using an image dataset from Parkinson's Progression Markers Initiative (PPMI) and was developed using ResNet50 and EfficientNetB0 as base model, utilized novel triangular dense layer structure to optimize feature extraction across the stages of PD. The model achieved a test accuracy of 97%, precision of 96%, recall of 97%, and an AUC score of 99%, demonstrating its potential as an effective PD stage classification model.

For the second stage in our study, we utilized machine learning and topological indices to analyze drug's structure. Molecular characteristics of thirteen Anti-Parkinson's drugs were explored. Topological indices have been calculated using a Python Program and actual physicochemical properties are collected from chemspider database using a custom python script. QSPR and SHAP analysis is performed to find which indices are best for predicting a specific property and machine learning models are trained on base of this analysis, providing a base for development of more generalized models for prediction of physicochemical properties for drugs development.

## 🚀 Key Features

- **Multi-class Classification**: Implementation of a novel approach to classify three distinct stages of Parkinson's Disease
- **Hybrid Model Architecture**: Combination of ResNet50 and EfficientNetB0 with triangular dense layer structure
- **Comprehensive Optimization**: Experiments with various batch sizes and learning rates
- **High Performance**: Achieved 97% test accuracy, 96% precision, 97% recall, and 99% AUC
- **Visualization Tools**: Includes confusion matrices, ROC curves, and training history plots
- **Drug Property Analysis**: QSPR and SHAP analysis for predicting physicochemical properties of drugs

## 💻 Technologies Used

- **Deep Learning Frameworks**: TensorFlow, Keras
- **Data Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebooks
- **Image Processing**: FSL for MRI preprocessing
- **Molecular Analysis**: Custom Python scripts for topological indices calculation

## 📊 Dataset

The data for this research is collected from the publicly available [Parkinson's Progression Marker's Initiative (PPMI) Database](https://www.ppmi-info.org/data). The dataset includes:
- 57,530 MRI images (512×512 in DICOM format)
- 40,022 images for training and 11,506 for testing (80%-20% ratio)
- Equal distribution across three classes: HC, PD, and Prodromal

## 📂 Repository Structure

This repository is organized into several main directories containing the implementation of different models and their optimization experiments:

```
neuroAI/
├── Efficient/                           # EfficientNetB0 implementation
│   ├── b32/                             # Batch size 32 experiments
│   ├── b64/                             # Batch size 64 experiments
│   ├── b128/                            # Batch size 128 experiments
│   └── b256/                            # Batch size 256 experiments
│
├── Ensemble/                            # Hybrid model implementation (ResNet50 + EfficientNetB0)
│   ├── Notebook2 - 96 b256/             # Batch size 256 implementation
│   ├── Notebook3 -128/                  # Batch size 128 implementation
│   ├── Notebook4 64/                    # Batch size 64 implementation
│   └── Notebook5 32/                    # Batch size 32 implementation
│
├── MobileNet/                           # MobileNet implementation
│
└── Resnet50/                            # ResNet50 implementation
    ├── Batch Optimization/              # Batch size optimization experiments
    │   ├── Batch 32/
    │   ├── Batch 64/
    │   ├── Batch 128/
    │   ├── Batch 256/
    │   └── Batch 256- Update/
    │
    ├── Learninig Rate Optimization/     # Learning rate optimization experiments
    │   ├── LR 0.0001/
    │   └── LR 0.001/
    │
    └── Resnet50_Final_Version/          # Final optimized ResNet50 model
```

Each experiment directory typically contains:
- Jupyter notebooks (`.ipynb`) with the implementation code
- Confusion matrices (`.png`) visualizing model performance
- ROC curves (`.png`) for model evaluation
- Classification reports (`.txt`) with precision, recall, and F1-scores
- Training history plots (`.png`) showing accuracy and loss evolution

## 🚀 Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/mudassar003/neuroAI.git
cd neuroAI
```

2. Install required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn seaborn jupyter
```

3. Working with the Jupyter notebooks:
   - Navigate to the model directory of interest (ResNet50, Efficient, or Ensemble)
   - Open the Jupyter notebook (e.g., `resnet50-ensemble-v2.ipynb`)
   - Follow the steps in the notebook for:
     - Data loading and preprocessing
     - Model training and optimization
     - Evaluation and visualization

4. Key notebooks:
   - For the hybrid model: `Ensemble/Notebook2 - 96 b256/resnet50-ensemble-v2.ipynb`
   - For batch size optimization: `ResNet50/Batch Optimization/` contains separate folders for each batch size
   - For learning rate experiments: `ResNet50/Learning Rate Optimization/`

## 📈 Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| ResNet50 | 95% | 0.95 | 0.95 | 0.95 |
| EfficientNetB0 | 96% | 0.96 | 0.96 | 0.96 |
| Hybrid Model | 97% | 0.97 | 0.97 | 0.97 |

The hybrid model achieved the highest performance with a test accuracy of 97%, demonstrating the effectiveness of model fusion in improving classification performance.

## 📃 Publications

### Related Publications

1. **A python approach for prediction of physicochemical properties of anti-arrhythmia drugs using topological descriptors**
   - Journal: Scientific Reports (2025)
   - DOI: [10.1038/s41598-025-85352-0](https://doi.org/10.1038/s41598-025-85352-0)
   - Authors: Qin, H.; Rehman, M.; Hanif, M.F.; Bhatti, M.Y.; Siddiqui, M.K.; Fiidow, M.A.

2. **Exploring the potential of artificial neural networks in predicting physicochemical characteristics of anti-biofilm compounds from 2D and 3D structural information**
   - Journal: Modern Physics Letters B (2025-04-04)
   - DOI: [10.1142/S021798492550157X](https://doi.org/10.1142/S021798492550157X)
   - Authors: Qasem M. Tawhari; Mudassar Rehman; Wakeel Ahmed; Ali Ahmad; Ali N. A. Koam

## 👨‍🎓 About the Author

### Mudassar Rehman
MS Thesis, COMSATS University Islamabad, Lahore Campus, Pakistan  
Student ID: CIIT/SP23-RMT-018/LHR

#### Thesis Supervision
- **Supervisor:** Dr. Muhammad Yousaf Bhatti, Department of Mathematics, COMSATS University Islamabad (CUI) Lahore Campus
- **External Examiner:** Prof. Dr. Muhammad Akram, Dean of Sciences, University of the Punjab, Lahore

#### Contact & Profiles
- 🔗 [LinkedIn](https://www.linkedin.com/in/mudassar-rehman-0224441b2/)
- 🎓 [Google Scholar](https://scholar.google.com/citations?user=t52lpvcAAAAJ&hl=en)
- 🔬 [ResearchGate](https://www.researchgate.net/profile/Mudassar-Rehman-5)
- 📝 [ORCID](https://orcid.org/0009-0007-8334-6777)
- 📧 Email: mudassar.rehman687@gmail.com

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
- Parkinson's Progression Markers Initiative (PPMI) for providing the dataset
- COMSATS University Islamabad, Lahore Campus for supporting this research
- Dr. Muhammad Yousaf Bhatti for his supervision and guidance
- Prof. Dr. Muhammad Akram for serving as external examiner
- All co-authors and contributors to this research