# Loan-Approval-Prediction
### Project Overview

A Machine Learning project to automate loan eligibility based on customer details like credit history, income, education etc using a Random Forest Classifier. 
The goal is to build a Machine Learning model that can predict whether a loan application will be approved or rejected.

### Data Overview
The data contains 381 records and 13 features
We have 5 numerical features, 6 categorical input features and 1 categorical target (loan_status)

Numerical features: ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History

Categorical input features: Gender,Married,Dependents,Education,Self_Employed,Property_Area



## 📂 Dataset Access

👉 [Download the dataset here from kaggle](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction)

### Data Preprocessing & EDA
1. Handling Missing Values: Missing values in the dataset were handled using mode imputation. For categorical columns like Gender, Self_Employed, and Dependents, the most frequent category was used to fill missing entries. Similarly, for numerical columns such as Loan_Amount_Term and Credit_History, missing values were replaced with the most frequently occurring value.

2. Feature Engineering: I dropped the Loan_ID as it was a non-predictive identifier.
3. Encoding: I used Manual Encoding for ordinal/binary data(eg. Gender, Married, Dependents, Education, Self_Employed and Loan_Status) and One-Hot Encoding for Property_Area to avoid artificial ordering.
4. From the EDA, I observed that:
   
   a. Majority of applicants are graduates.
   
   b. Applicants with a credit history of 1 have significantly higher loan approval rates. (This suggests Credit_History is a strong predictor)
   
   c. Approved applicants tend to have slightly higher median income.
   
   d. Credit_History shows the strongest correlation with Loan_Status among numerical variables.
   
   e. From the exploratory analysis, Credit_History emerged as the strongest predictor of loan approval. Income and Loan_Amount showed skewed distributions with outliers.
   The dataset shows slight class imbalance with more approved loans than rejected ones.


   <img width="1430" height="571" alt="s1" src="https://github.com/user-attachments/assets/95b9e696-399c-4072-902a-00bf70ddc70e" />




## Model and Performance

### Model Evaluation Results: 

Below is the visualization of my model performance . I selected Random Forest as the best model based on its superior accuracy and reliable cross-validation score. Cross-validation (CV) was applied to confirm that the model performs well on new, unseen data rather than simply memorizing patterns from the training dataset.

<img width="971" height="338" alt="image" src="https://github.com/user-attachments/assets/df74b7e4-136c-4b40-a7d8-705b1e69c9f4" />






## Prediction System

loaded_model = joblib.load('loan_status_predictor.pkl')

##### Create the sample data with the correct feature names

``` Python
sample_data = pd.DataFrame({
    'Gender': [1],
    'Married': [1],
    'Dependents': [2],
    'Education': [0],
    'Self_Employed': [0],
    'ApplicantIncome': [2889],
    'CoapplicantIncome': [0.0],
    'LoanAmount': [45],
    'Loan_Amount_Term': [180],
    'Credit_History': [1], # Replace 'Property_Area' with one-hot encoded columns
    'Property_Area_Semiurban': [1],
    'Property_Area_Urban': [0]
    
})

prediction = loaded_model.predict(sample_data)

result = "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"
print(f"Prediction Result: {result}")

Prediction Result: Loan Approved
```

###
Pipelines: I used Scikit-Learn Pipeline and StandardScaler so as to prevent data leakage.





### Conclusion & Insights


The goal of this project was to build a robust predictive model to automate the loan approval process. Through rigorous data preprocessing and model selection, I achieved the following:


Analysis showed that Credit History is the most significant factor in determining loan eligibility(Primary Predictor).


The model "Random Forest Classifier" outperformed other models with an accuracy of 87% and a stable cross-validation score of 83%.


By implementing Scikit-Learn Pipelines, the project is designed for easy scaling, allowing for seamless integration of new data preprocessing steps or different scaling techniques without code duplication .


### Business Impact: This tool can significantly reduce the manual workload for credit officers by providing an instant, data-driven "pre-approval" status.
