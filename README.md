# 🧬 Disease Risk Prediction using Machine Learning

This project uses a synthetic dataset of patient health profiles to build a machine learning model for predicting the likelihood of developing a chronic disease. It focuses on classification using XGBoost, model evaluation with standard metrics, and explainability with SHAP.

---

## Dataset

- **Source**: Kaggle  
- **Name**: Disease Risk Prediction Dataset  
- **Rows**: 4,000 simulated patients  
- **Target variable**: `Disease_Risk` (Binary: Yes/No)

**Features include**:
- Demographics: Age, Gender
- Lifestyle: Smoking, Alcohol, Physical activity
- Clinical: Blood pressure, Cholesterol, Glucose
- Genetic: Family history, Genetic risk score
- Diagnosis history

---

## 🛠️ Tools & Libraries Used

- Python (Google Colab)
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn`
- `xgboost`: final model
- `shap`: for explainable AI 

---

## Model Building Steps

1. **Data Preprocessing**  
   - Categorical encoding 
   - Train-test split  

2. **Model Training**  
   - XGBoost classifier with `scale_pos_weight` to address class imbalance

3. **Model Evaluation**  
   - Accuracy: **~90.4%**  
   - ROC AUC: **~0.95**  
   - F1-Score (Positive Class): **~0.70**  
   - Confusion Matrix, Classification Report

4. **Model Interpretability (SHAP)**  
   - Identified most influential features:  
     - BMI, Smoking Status, Cholesterol Level, Age, Blood Pressure

---

## ✅ Results Summary

| Metric         | Score     |
|----------------|-----------|
| Accuracy       | 90.4%     |
| ROC AUC        | 0.9575    |
| Precision (1)  | 0.70      |
| Recall (1)     | 0.70      |
| F1-score (1)   | 0.70      |

---

## 📊 SHAP Feature Importance

The SHAP analysis showed that **BMI**, **Smoking Status**, and **Cholesterol Level** are among the most impactful features in predicting disease risk.

![SHAP Plot](shap_summary_plot.png)

---

## 💾 Model Saving

The trained model is saved using `joblib`:

```python
import joblib
joblib.dump(model, 'xgboost_disease_model.pkl')
