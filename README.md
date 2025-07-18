# Real-Time Network Intrusion Detection using Machine Learning

This project explores the application of machine learning algorithms for real-time **Network Intrusion Detection (NID)** in telecommunication networks. The aim is to mitigate system downtime by recognizing anomalies and attack patterns in network traffic. The work includes preprocessing network traffic data, training multiple models, evaluating their performance, and identifying the most suitable model for deployment.

## 📌 Table of Contents

- [Overview]
- [Dataset]
- [Project Structure]
- [Setup Instructions]
- [Models Evaluated]
- [Key Results]
- [Conclusion]
- [Future Work]

## 🧠 Overview

The primary objective is to build and evaluate machine learning models for detecting intrusions in real-time network traffic. This is critical for protecting telecommunication infrastructures from outages and service disruptions caused by cyberattacks.

The project:
- Uses the **KDD Cup 99** dataset
- Applies preprocessing techniques including feature scaling, encoding, and PCA for dimensionality reduction
- Compares classification algorithms and ensemble methods
- Explores trade-offs between **accuracy**, **prediction latency**, and **memory usage**

## 📁 Dataset

- **Name:** [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Samples:** ~126,000
- **Features:** 43 original attributes representing traffic patterns
- **Target:** Binary classification — `normal` vs `attack`

## 🗂 Project Structure

📦 network-intrusion-detection-ml
├── 📄 README.md 
├── 📓 intrusion_detection.ipynb <-- Jupyter notebook (1.5MB)
├── 📁 data/
│ └── kddcup99.csv <-- Raw dataset (not included)
├── 📁 models/ 
├── 📁 results/
│ └── plots/, metrics/ 
├── 📄 requirements.txt 



## 🚀 Models Evaluated

The following models were trained and benchmarked:

### Classification Models
- Logistic Regression
- Naive Bayes
- Linear SVC
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Isolation Forest

### Ensemble Methods
- XGBoost
- Gradient Boosting
- CatBoost
- LightGBM

## 📊 Key Results

| Model         | Test Accuracy | Detection Time (s) | Memory Usage (MB) | Remarks                                |
|---------------|---------------|---------------------|--------------------|-----------------------------------------|
| LightGBM      | 0.999         | 0.93                | 813.6              | Best model for real-time large-scale data |
| XGBoost       | 0.999         | 1.2                 | moderate           | Excellent accuracy, good generalization |
| CatBoost      | 0.998         | 0.004               | low                | Fast and highly accurate                |
| Naive Bayes   | 0.92          | 0.365               | low                | Fast, effective basic model             |
| KNN           | 0.99          | 43.5                | 5.94               | High accuracy but too slow              |

📉 **Dimensionality Reduction:** PCA reduced 122 features to 20 components, improving model speed and avoiding overfitting.

⚖️ Evaluation included:
- F1-score
- ROC-AUC
- Precision/Recall
- Memory and latency benchmarks

## ✅ Conclusion

**LightGBM and XGBoost** emerged as the most suitable models for real-time NID based on their:
- Exceptional prediction accuracy (≈99.9%)
- Real-time speed (sub-second latency)
- Balance between memory usage and generalization

## 🔄 Future Work

To enhance the performance and robustness of intrusion detection:
- Apply **SMOTE** for handling class imbalance
- Explore **deep learning models** (CNN, RNN)
- Test models on **real-time network streams**
- Increase interpretability via SHAP or LIME


