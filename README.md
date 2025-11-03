# Detection-and-Classification-of-Malicious-Android-Applications

A machine learning project for detecting and classifying malicious Android applications using a combination of static and dynamic analysis features.

## DISCLAMER: This is a Work In Progress!

I intend to refine the documentation, code and and perhaps improve some model parameters. 
After this, I would like to expand the dataset to capture call sequences, and use RNN/LSTM models.

## Project Overview

This project implements a two-stage approach to Android malware analysis
1. Binary Model Detects whether an application is malicious or benign
2. Multiclass Model Classifies malicious applications into specific malware families

The system is designed to work efficiently with static features (easily extractable from APK files) for initial detection, followed by more resource-intensive dynamic analysis for classification.

> Note: You can find my detailed thinking process, experimentation timeline, and iterative improvements in the TODO.md file.

## Motivation

With the adoption of the Digital Markets Act (DMA), alternative app stores are becoming more prevalent, increasing the risk of users encountering malicious applications. Traditional detection methods are resource-intensive, but machine learning can significantly reduce the time and computational resources needed for malware evaluation.

## Dataset

The project uses the [Kronodroid Dataset](https://github.com/aleguma/kronodroid)
- 41,382 malicious applications (240 families)
- 36,755 benign applications
- Total 78,137 samples

### Features
- 289 dynamic attributes System calls, API usage counts
- 200 static attributes Permissions, intent filters, metadata
- Key attributes 
  - `normal` Number of normal permissions requested
  - `dangerous` Number of dangerous permissions requested
  - `Malware` Target for binary classification (0 or 1)
  - `MalFamily` Target for multiclass classification (malware family name)

## Data Preprocessing

1. Data Cleaning
   - Combined both static and dynamic datasets
   - Removed all rows and columns with missing values
   - Removed empty rows and columns

2. Feature Engineering
   - Separated data into static and dynamic analysis datasets
   - Removed columns with all 0s or 1s (no predictive value)
   - Applied feature selection to remove low-variance features
   - Removed highly correlated features (correlation  0.95)
   - Used `total_perm` instead of separate `dangerous` and `normal` counts for practical deployment

3. Data Split
   - TrainValidationTest 70%15%15%

4. Malware Family Selection
   - Focused on 35 families with over 100 samples each
   - These families cover 93% of the malicious dataset
   - Remaining families grouped into Other category

## Models Tested

The following algorithms were evaluated for both tasks
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- AdaBoost
- Multi-Layer Perceptron (Neural Network)
- TabNet (for multiclass classification)

## Results

### Binary Classification Model (Malware Detection)

Best Model LightGBM

    Model                  Accuracy     Precision    Recall     F1-Score
    ----------------------------------------------------------------------
    LightGBM                 98.42%      98.83%      98.17%      98.50%


Final Model Configuration
```python
LGBMClassifier(
    random_state=42,
    learning_rate=0.3,
    max_depth=-1,
    n_estimators=400,
    num_leaves=100
)
```

### Multiclass Classification Model (Malware Family Classification)

Best Model LightGBM

    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    LightGBM                    86.14%      86.50%      86.14%      86.13%


Final Model Configuration
```python
LGBMClassifier(
    random_state=42,
    objective='multiclass',
    num_class=36,
    class_weight='balanced',
    num_leaves=64,
    max_depth=9,
    n_estimators=400,
    learning_rate=0.095,
    verbose=-1
)
```

### Complete Model Comparison (Validation Set)

#### Binary Classification
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression         93.71%      95.29%      92.69%      93.97%
    Random Forest               98.01%      98.23%      97.99%      98.11%
    XGBoost                     98.34%      98.50%      98.35%      98.42%
    LightGBM                    98.57%      98.72%      98.56%      98.64%
    Support Vector Machine      97.36%      97.64%      97.36%      97.50%
    AdaBoost                    94.37%      95.00%      94.30%      94.65%
    Neural Network              97.84%      98.34%      97.57%      97.95%

#### Multiclass Classification
    Model                  Accuracy     Precision    Recall       F1-Score
    ----------------------------------------------------------------------
    Logistic Regression         74.80%      73.43%      74.80%      73.56%
    Random Forest               84.98%      84.95%      84.98%      84.44%
    XGBoost                     85.54%      85.66%      85.54%      85.32%
    LightGBM                    86.14%      86.50%      86.14%      86.13%
    Support Vector Machine      83.14%      83.07%      83.14%      82.87%
    AdaBoost                    62.85%      63.91%      62.85%      61.95%
    TabNet                      81.36%      81.35%      81.36%      80.87%

## Key Findings

1. Feature Selection Using feature selection without PCA yielded better results than PCA alone or PCA combined with feature selection
2. Scaling Impact Minimal impact on tree-based models (XGBoost, LightGBM) but essential for Logistic Regression, SVM, and Neural Networks
3. Correlation Removal Removing features with correlation `0.95` improved generalization
4. Class Balancing SMOTE and class weighting did not improve multiclass performance
5. `LightGBM` consistently outperformed other models for both binary and multiclass tasks

## Technology Stack

- Python 3.x
- Core Libraries
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
- ML Libraries
  - XGBoost
  - LightGBM
  - pytorch-tabnet
- Potential Extensions
  - androguard (for extracting features from APK files)
  - concurrent.futures & multiprocessing (for parallel processing)

## Usage

The two-stage detection system works as follows

1. Stage 1 - Binary Detection Input an application's static features into the Binary Model
   - If classified as benign -> Application is safe
   - If classified as malicious -> Proceed to Stage 2

2. Stage 2 - Family Classification Input the application's dynamic features into the Multiclass Model
   - Classifies the malware into one of 36 families for further analysis

## Evaluation Metrics

- Binary Classification Accuracy, Precision, Recall, F1-Score
- Multiclass Classification F1-Score, Confusion Matrix

## Future Improvements

- Experiment with deep learning approaches (RNN/LSTM) for analyzing sequences of API calls
- Implement semi-supervised learning techniques
- Integrate real-time APK feature extraction using androguard
- Deploy as a web service for practical malware scanning

## References

- [Kronodroid Dataset](https://github.com/aleguma/kronodroid)
- [Dataset Paper](https://www.sciencedirect.com/science/article/pii/S0167404821002236)
- [TabNet](https://github.com/dreamquark-ai/tabnet)

## License

This project is for educational and research purposes.

---

Author Emilija Opsenica (RA 108-2022)