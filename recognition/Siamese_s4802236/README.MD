# Classifying ISIC 2020 Data Set with a Siamese Network

## Abstract

This project develops a Siamese neural network in PyTorch to classify images from the ISIC 2020 Kaggle Challenge into normal and melanoma categories. The model leverages a ResNet backbone combined with a 128-dimensional projection head, learning a discriminative embedding space through triplet-loss pretraining before fine-tuning for binary classification.
A key objective was to reach a test accuracy exceeding 0.8, the official benchmark for this hard-difficulty task - a target that was successfully achieved. Beyond accuracy, the project evaluates performance using AUROC and precision-recall analysis to better account for class imbalance inherent to melanoma detection. The results demonstrate that embedding-based pretraining improves generalisation and convergence stability, though the discussion highlights the limitations of relying solely on accuracy as a success criterion for medical classification problems.

## Success Criterion

### **Official Performance Benchmark**

According to the COMP3710 project specification, the success criterion for this hard-difficulty task is to achieve an accuracy ≥ 0.8 on the test set of the ISIC 2020 Kaggle Challenge.
Our final Siamese network comfortably meets this requirement, attaining:
`[TEST]  accuracy=0.9276`
### **Limitations of Accuracy as a Metric**

While accuracy exceeds the benchmark, it is not a fully reliable indicator of real-world performance for this dataset. The ISIC 2020 data has a high degree of class imbalance, with only 1.76% (584 / 32,542) of samples representing melanoma and 98.24% (31,958 / 32,542) representing normal skin. In such an imbalanced setting, a naïve classifier that predicts every image as “normal” would already achieve around 98% accuracy - masking potentially poor sensitivity to melanoma detection. 
Therefore, aditional to accuracy, metrics such as Area Under ROC Curve (AUROC), percision, and confusion matrices were also used to provide a more complete evaluation of the model's effectiveness or ineffectiveness. 


## File Structure
The file structure for the codebase is as follows:
```PatternAnalysis-2025/recognition/Siamese_s4802236/
│
├── README_images/
│   ├── readme_image_1.jpg
│   ├── readme_image_2.jpg
│   └── ...
│
├── dataset.py
├── modules.py
├── predict.py
├── train.py
├── report.py
├── requirements.txt
└── README.md
```


## Methodology


### Data Preparation

### Model Architecture

### Training Procedure

### Predictions of test data

## Results and Findings

## Limitations


