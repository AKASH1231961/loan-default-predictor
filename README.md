
# Loan Default Risk Classification for Banking

## 👥 Team Details
- **Project Title:** Loan Default Risk Classification for Banking    
- **Team Members:**  
  - Akash S
  - Angel George  
  - Sravana Nambiar  

---

## 📌 Problem Statement

Banks face significant financial losses due to loan defaults. Approving high-risk applicants increases non-performing assets, while rejecting too many applicants reduces business opportunities.

This project aims to **predict whether a loan applicant will default**, enabling better credit decision-making.

---

## 🎯 Motivation

- Reduce financial risk in lending  
- Improve approval decisions using data-driven insights  
- Ensure regulatory compliance through model explainability  
- Handle class imbalance (defaults are rare but critical)

---

## 📊 Dataset Description

- **Source:**  
  - Home Credit Default Risk Dataset / LendingClub Dataset  

- **Size:**  
  - ~300,000+ records (varies by dataset)

- **Key Features:**
  - Applicant income  
  - Credit history  
  - Loan amount  
  - Employment details  
  - Debt-to-income ratio  

- **Target Variable:**
  - `Default` (1 = Default, 0 = No Default)

- **Class Distribution:**
  - Highly imbalanced  
  - Majority: Non-default  
  - Minority: Default  

---

## 🔄 Methodology

### 1. Data Preprocessing
- Handling missing values  
- Encoding categorical variables  
- Feature scaling  
- Outlier treatment  

---

### 2. Exploratory Data Analysis (EDA)
- Distribution of income, loan amount  
- Correlation analysis  
- Default vs non-default comparisons  

---

### 3. Feature Engineering
- Derived financial ratios  
- Risk indicators  
- Encoding categorical variables  

---

### 4. Handling Class Imbalance
- Class weighting  
- Stratified splitting  

---

### 5. Model Development

We implemented and compared:

- **Logistic Regression (Class-weighted)**
- **Random Forest**
- **LightGBM**

---

### 6. Model Optimization
- Hyperparameter tuning  
- Cross-validation  
- Focus on **Recall** (to minimize risky approvals)

---

### 7. Model Explainability
- Used **SHAP (SHapley Additive Explanations)**  
- Identified key drivers of loan default:
  - Income  
  - Credit history  
  - Loan amount  

---

## 📈 Results & Evaluation

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|----------|--------|----------|
| Logistic Regression| 0.78     | 0.42     | 0.72   | 0.53     |
| Random Forest      | 0.84     | 0.55     | 0.68   | 0.60     |
| LightGBM           | **0.87** | **0.61** | **0.75** | **0.67** |

### ✅ Key Takeaways
- LightGBM performed best overall  
- Recall was prioritized to reduce false approvals  
- SHAP improved model transparency  

---

## 🖥️ Application (Streamlit)

We deployed an interactive web app using Streamlit where users can:
- Input applicant details  
- Predict default risk  
- View app > https://akash1231961-loan-default-predictor-app-qbzl8j.streamlit.app/

---

## 📸 Screenshots

### 🔹 Home Page
<img width="1600" height="748" alt="Model comparison" src="https://github.com/user-attachments/assets/6cc5650f-9f97-44a9-ac76-056914158614" />



### 🔹 Prediction Output
<img width="1600" height="725" alt="Prediction" src="https://github.com/user-attachments/assets/626ae5d3-68b8-4e31-b15a-a6783b787198" />


### 🔹 SHAP Explanation
![SHAP](images/shap.png)

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/loan-default-risk.git
cd loan-default-risk
