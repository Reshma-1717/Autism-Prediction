Autism Prediction using Machine Learning
Project Overview

This project focuses on predicting Autism Spectrum Disorder (ASD) using machine learning techniques. It analyzes behavioral screening data and demographic features to provide early autism risk prediction, supporting timely intervention.

Technologies Used

Python

Jupyter Notebook

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib, Seaborn

SMOTE (Imbalanced-learn)

Dataset

The project uses a structured dataset containing:

Behavioral screening scores (A1–A10)

Demographic details such as age, gender, ethnicity

Medical history features like jaundice and family history of autism

Methodology

Data preprocessing (label encoding, outlier handling)

Handling class imbalance using SMOTE

Exploratory Data Analysis (EDA)

Model training using Decision Tree, Random Forest, and XGBoost

Hyperparameter tuning and cross-validation

Performance evaluation using accuracy, precision, recall, and F1-score

Results

Among all models, XGBoost achieved the best performance with high accuracy and balanced prediction for both ASD and Non-ASD classes.

Files Included

Autism_Prediction.ipynb – Main implementation code

train.csv – Dataset

best_model.pkl – Trained model

encoders.pkl – Saved label encoders

Autism_Prediction.pdf – Project documentation

Conclusion

This project demonstrates how machine learning can be effectively used for early autism screening using structured data. The model can serve as a foundation for building accessible and scalable healthcare screening tools.
