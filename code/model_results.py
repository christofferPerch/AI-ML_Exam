import pandas as pd

data_table_no_outliers = {
    "Model": [
        "Logistic Regression (Without Resampling)", "Logistic Regression (Stratified)",
        "Logistic Regression (Random Oversampling)","Logistic Regression (SMOTE)", 
        "Random Forest (Without Resampling)", "Random Forest (Stratified)", 
        "Random Forest (Random Oversampling)", "Random Forest (SMOTE)"
    ],
    "Accuracy":[
        0.89, 0.89,
        0.72, 0.77,
        0.89, 0.89,
        0.88, 0.86 
        
    ],
    "Precision": [
        0.44, 0.43, 
        0.24, 0.26, 
        0.47, 0.47,
        0.40, 0.30
    ],
    "Recall": [
        0.10, 0.14, 
        0.77, 0.61, 
        0.07, 0.07,
        0.20, 0.22
    ],
    "F1-score": [
        0.16, 0.22, 
        0.37, 0.37, 
        0.12, 0.12,
        0.26, 0.25
    ]
}

# Create the DataFrame
classification_reports_df_without_outliers = pd.DataFrame(data_table_no_outliers)
classification_reports_df_without_outliers.reset_index(drop=True)

classification_reports_df_without_outliers


data_table_with_outliers = {
    "Model": [
        "Logistic Regression (Without Resampling)", "Logistic Regression (Stratified)",
        "Logistic Regression (Random Oversampling)","Logistic Regression (SMOTE)", 
        "Random Forest (Without Resampling)", "Random Forest (Stratified)", 
        "Random Forest (Random Oversampling)", "Random Forest (SMOTE)"
    ],
    "Accuracy":[
        0.88, 0.89,
        0.72, 0.77,
        0.88, 0.89,
        0.88, 0.86 
        
    ],
    "Precision": [
        0.47, 0.46, 
        0.26, 0.27, 
        0.47, 0.48,
        0.42, 0.32
    ],
    "Recall": [
        0.12, 0.13, 
        0.77, 0.61, 
        0.07, 0.07,
        0.21, 0.23
    ],
    "F1-score": [
        0.19, 0.20, 
        0.39, 0.38, 
        0.13, 0.13,
        0.28, 0.26
    ]
}

# Create the DataFrame
classification_reports_df_with_outliers = pd.DataFrame(data_table_with_outliers)
classification_reports_df_with_outliers.reset_index(drop=True)

classification_reports_df_with_outliers