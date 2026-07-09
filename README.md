# Autism Spectrum Disorder (ASD) Prediction using Machine Learning

A machine learning project that predicts the likelihood of Autism Spectrum Disorder (ASD) from behavioral screening questionnaire responses and demographic/medical data, using classical ML classifiers (Decision Tree, Random Forest, XGBoost) with class-imbalance handling via SMOTE.

## Project Structure

```
Autism-Prediction-main/
├── README.md
├── data/
│   └── train.csv                                       # Screening dataset (800 records)
├── notebooks/
│   └── Autism_Prediction_using_Machine_Learning.ipynb   # Full EDA + modeling pipeline
└── reports/
    └── Autism_Prediction.pdf                            # Project write-up / documentation
```

## Problem Statement

Early screening for autism is typically done through structured questionnaires (such as the AQ-10). This project uses the responses to a 10-question behavioral screening test, together with demographic and medical history features, to build a binary classifier that predicts whether an individual is likely to be diagnosed with ASD (`Class/ASD` = 1) or not (`Class/ASD` = 0).

## Dataset

`data/train.csv` contains **800 records** and **21 columns**:

| Feature group | Columns |
|---|---|
| Screening responses | `A1_Score` – `A10_Score` (binary answers to the AQ-10 questionnaire) |
| Demographics | `age`, `gender`, `ethnicity`, `contry_of_res` |
| Medical / behavioral history | `jaundice` (born with jaundice), `austim` (family history of autism), `used_app_before` |
| Screening result | `result` (continuous score derived from the questionnaire), `age_desc`, `relation` (who completed the test) |
| Identifier | `ID` |
| Target | `Class/ASD` (0 = No ASD, 1 = ASD) |

**Class distribution:** 639 negative vs 161 positive cases — a notably imbalanced dataset (~80/20 split), which motivates the use of SMOTE during training.

## Methodology

The notebook (`notebooks/Autism_Prediction_using_Machine_Learning.ipynb`) follows this pipeline:

1. **Data Cleaning**
   - Dropped uninformative columns (`ID`, `age_desc`)
   - Fixed inconsistent category labels (e.g. merging `Hong Kong` → `China`, `AmericanSamoa` → `United States` in `contry_of_res`)
   - Consolidated rare/ambiguous categories in `ethnicity` and `relation` into an `Others` bucket
   - Cast `age` to integer type

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots (histograms, KDE) for `age` and `result`
   - Box plots and the IQR method to detect outliers in `age` and `result`
   - Count plots for every categorical feature and the target class
   - Correlation heatmap across all encoded features

3. **Preprocessing**
   - Label encoding of all categorical/object columns (encoders saved for reuse)
   - Outlier treatment: values outside the IQR bounds for `age` and `result` replaced with the column median
   - Train/test split (80/20, stratified via `random_state=42`)
   - **SMOTE** oversampling applied only to the training set to balance the two classes (515/515 after resampling)

4. **Modeling**
   - Baseline 5-fold cross-validation comparing three classifiers:
     - Decision Tree
     - Random Forest
     - XGBoost
   - Hyperparameter tuning for all three models using `RandomizedSearchCV` (20 iterations, 5-fold CV)
   - Best model selected by cross-validation accuracy and evaluated on the held-out test set

## Results

**Baseline 5-fold CV accuracy (default parameters):**

| Model | CV Accuracy |
|---|---|
| Decision Tree | 0.86 |
| Random Forest | 0.92 |
| XGBoost | 0.90 |

**After hyperparameter tuning**, the best model selected was a tuned **Random Forest** (`bootstrap=False, max_depth=20, n_estimators=50`) with a cross-validation accuracy of **0.93**.

**Final test set performance (160 held-out samples):**

| Metric | Value |
|---|---|
| Accuracy | 0.82 |
| Precision (Class 0 / Class 1) | 0.89 / 0.59 |
| Recall (Class 0 / Class 1) | 0.87 / 0.64 |
| F1-score (Class 0 / Class 1) | 0.88 / 0.61 |

The model performs strongly on the majority (Non-ASD) class, while precision/recall on the minority (ASD) class is more modest — a common outcome given the original class imbalance and the relatively small number of positive cases in the test set.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn (Decision Tree, Random Forest, RandomizedSearchCV, metrics)
- XGBoost
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```
2. Open `notebooks/Autism_Prediction_using_Machine_Learning.ipynb` in Jupyter/Colab.
3. Update the data path if needed (the notebook reads from `/content/train.csv`, a Colab path — point it to `data/train.csv` if running locally).
4. Run all cells to reproduce EDA, preprocessing, model training, and evaluation.

Running the full notebook also produces two reusable artifacts (not included in this repo folder, but generated on execution):
- `best_model.pkl` – the trained, tuned best-performing model
- `encoders.pkl` – the fitted label encoders for categorical columns, needed to preprocess new inference data consistently

## Conclusion

This project demonstrates a full applied ML workflow for a healthcare screening use case: cleaning noisy real-world survey data, addressing class imbalance with SMOTE, comparing multiple tree-based models, and tuning the best candidate. The resulting Random Forest model can serve as a starting point for a lightweight, accessible ASD pre-screening tool, with the caveat that further work on minority-class recall/precision would be valuable before any real-world deployment.
