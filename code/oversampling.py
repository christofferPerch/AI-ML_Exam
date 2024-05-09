import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def perform_random_oversampling(model, X_train, y_train, X_test, y_test, random_state=42):
  
    # Random Oversampling
    rd_oversampler = RandomOverSampler(random_state=random_state)
    X_train_oversampled, y_train_oversampled = rd_oversampler.fit_resample(X_train, y_train)

    # Train the model
    model_clone = clone(model)
    model_clone.fit(X_train_oversampled, y_train_oversampled)
    
    # Evaluate the model
    y_pred = model_clone.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Plotting the class distribution BEFORE oversampling
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # y_train.value_counts().plot(kind="bar", color=["skyblue", "salmon"])
    # plt.title("Class Distribution Before Oversampling")
    # plt.xlabel("Had Heart Disease")
    # plt.ylabel("Frequency")

    # # Plotting the class distribution AFTER oversampling
    # plt.subplot(1, 2, 2)
    # pd.Series(y_train_oversampled).value_counts().plot(kind="bar", color=["skyblue", "salmon"])
    # plt.title("Class Distribution After Oversampling")
    # plt.xlabel("Had Heart Disease")
    # plt.ylabel("Frequency")

    # plt.tight_layout()
    # plt.show()
    
    return model_clone

# Random Oversampling
#rd_oversampler = RandomOverSampler(random_state=42)
#X_train_oversampled, y_train_oversampled = rd_oversampler.fit_resample(X_train, y_train)
    

def perform_stratified_cv(model, X, y, n_splits=2, sample_frac=1.0):

    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
     
    for i, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Using a fraction of the dataset if sample_frac is less than 1.0
        if sample_frac < 1.0:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train_fold, y_train_fold, test_size=1-sample_frac, random_state=42)
        else:
            X_train_sample, y_train_sample = X_train_fold, y_train_fold

        train_counts = y_train_sample.value_counts(normalize=True)
        test_counts = y_test_fold.value_counts(normalize=True)
           
        model_clone = clone(model)
        model_clone.fit(X_train_sample, y_train_sample)
        y_pred = model_clone.predict(X_test_fold)

        print(f"\nClassification Report for Fold {i+1}:\n")
        print(classification_report(y_test_fold, y_pred))
        
def perform_smote_oversampling(model, X_train, X_test, y_train, y_test, random_state=42):
 
    # SMOTE Oversampling
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train the model
    model_clone = clone(model)
    model_clone.fit(X_train_smote, y_train_smote)
    
    # Evaluate the model
    y_pred = model_clone.predict(X_test)
    print(classification_report(y_test, y_pred))