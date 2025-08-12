# 📊 Customer Churn Prediction using Machine Learning

This project focuses on predicting customer churn for a subscription-based service using historical customer data.  
By analyzing features such as demographics, usage behavior, and account details, the model identifies customers at risk of leaving so businesses can take proactive retention measures.

---

## 🚀 Features

### 🧹 Data Preprocessing
- Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical variables using **Label Encoding** & **One-Hot Encoding**
- Scaled numerical features with **StandardScaler**

### 🌲 Model Training
- Implemented **Gradient Boosting**, **Random Forest**, and **Logistic Regression** for binary classification

### 📊 Model Evaluation
- Measured performance using:
  - Accuracy Score
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

### 🔍 Visualization
- Used **Seaborn Heatmaps** to visualize prediction results

### 📈 Full Dataset Predictions
- Generated churn predictions for all customers in the dataset

---

## 🛠 Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Seaborn**

---

## 📂 Dataset
The dataset contains customer demographic, account, and behavioral data along with churn labels.  

**Dataset Source:** [Provided for CodSoft Internship (contains ~10,000 customer records)](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

---

## 📊 Model Workflow
1. Load and explore dataset
2. Preprocess data (encoding & scaling)
3. Train-test split for model evaluation
4. Train models: **Gradient Boosting**, **Random Forest**, and **Logistic Regression**
5. Evaluate model performance
6. Visualize confusion matrix and feature importance
7. Generate churn predictions for the entire dataset

---

## 📌 Output Example
```yaml
Model Evaluation Results:
| Model               | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
|---------------------|----------|-----------|----------|----------|----------|
| GradientBoosting    | 0.8700   | 0.792829  | 0.488943 | 0.604863 | 0.870831 |
| RandomForest        | 0.8680   | 0.823529  | 0.447174 | 0.579618 | 0.864263 |
| LogisticRegression  | 0.8085   | 0.593750  | 0.186732 | 0.284112 | 0.774801 |
