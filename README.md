# Credit Card Fraud Detection using Machine Learning

![Fraud Detection](https://your-image-link.com/fraud-detection-banner.png)

## 📌 Project Overview
Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are significantly rare compared to genuine ones. This project applies various machine learning techniques to accurately detect fraud while minimizing false positives.

## 📊 Dataset
- **Source**: Kaggle / Public Dataset
- **Size**: Large dataset with over 67,000 transactions
- **Imbalance**: Fraud cases are less than 1% of total transactions
- **Key Features**:
  - **Transaction Amount**: Amount spent in the transaction
  - **Time**: Time since the first transaction
  - **Customer Behavior Features**
  - **Anomaly Indicators**

## ⚙️ Models & Techniques Used
### 1️⃣ **Preprocessing & Feature Engineering**
   - Handling missing values
   - Feature scaling & transformation
   - Addressing data imbalance using **SMOTE** (Synthetic Minority Over-sampling)

### 2️⃣ **Machine Learning Models**
   - ✅ **Logistic Regression**
   - ✅ **Random Forest**
   - ✅ **XGBoost (Optimized with RandomizedSearchCV)**
   - ✅ **Isolation Forest (Anomaly Detection)**
   - ✅ **Ensemble Learning (Stacking Multiple Models)**

### 3️⃣ **Performance Metrics**
   - 🔍 **Precision, Recall, F1-Score** for fraud detection evaluation
   - 📈 **AUC-ROC Score** to compare model effectiveness
   - 📊 **Confusion Matrix** to analyze false positives & false negatives

## 🚀 Results & Insights
| Model               | Precision | Recall | F1-Score | AUC-ROC |
|--------------------|-----------|--------|----------|---------|
| **XGBoost**        | **0.95**  | 0.80   | 0.87     | **0.9979** |
| **Random Forest**  | **0.96**  | 0.72   | 0.82     | 0.9822 |
| **Logistic Reg.**  | 0.05      | 0.71   | 0.10     | 0.9098 |
| **Isolation Forest** | 0.02     | 0.38   | 0.05     | 0.5000 |
| **Ensemble Model** | 0.81      | 0.78   | 0.80     | 0.9809 |

📌 **Key Takeaways:**
- 🚀 **XGBoost performed the best** with the highest AUC-ROC of **0.9979**.
- 🔥 **Random Forest significantly improved after SMOTE**, boosting precision from **0.01 to 0.96**.
- 🔄 **Ensemble models enhanced fraud detection**, confirming the strength of model stacking.
- ❌ **Isolation Forest struggled with SMOTE**, proving unsupervised models don’t benefit from synthetic sampling.

### **🔹 Confusion Matrix**
![Confusion Matrix](https://your-image-link.com/confusion-matrix.png)

## 📊 SHAP Analysis (Explainability)
To interpret model decisions, **SHAP (SHapley Additive Explanations)** was used:
- **Feature Importance**: Identifies top contributing features to fraud detection.
- **Dependence & Waterfall Plots**: Explain individual predictions.
- **Force Plots**: Visualize how each feature impacts a single prediction.

### **🔹 SHAP Summary Plot**
![SHAP Summary](https://your-image-link.com/shap-summary.png)

## 🛠 How to Use the Code
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Fraud Detection Model
```python
python fraud_detection.py
```

### 3️⃣ Perform SHAP Analysis
```python
python shap_analysis.py
```
📧 **Contact:** [shahidr54gb@gmail.com / shahidr-ds]
