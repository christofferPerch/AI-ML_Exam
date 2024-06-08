import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np


df_cleaned = pd.read_csv("../data/processed/heart_2022_cleaned.csv")

df_transformed = df_cleaned.copy()


# Feature Scaling / Standardization for numerical features:
numeric_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]
scaler = StandardScaler()
df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

# Initially we have decided to replace all the existing binary values to 0 and 1's:
df_transformed = df_transformed.replace({"No": 0, "Yes": 1})

# Encoding "Sex" with 0 and 1's.
df_transformed["Sex"] = df_transformed["Sex"].replace({"Female": 0, "Male": 1})
df_transformed["Sex"].value_counts()

# Encoding "HadDiabetes" with 0 and 1's, but first we replace the strings with No and Yes:
df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace(
    {
        "No, pre-diabetes or borderline diabetes": "No",
        "Yes, but only during pregnancy (female)": "Yes",
    }
)
df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace(
    {"No": 0, "Yes": 1}
)
df_transformed["HadDiabetes"].value_counts()

# Categorizing and Encoding "AgeCategory" into 13 groups.
age_ranges = df_transformed["AgeCategory"].unique()
age_codes, _ = pd.factorize(age_ranges, sort=True)
age_range_to_code = dict(zip(age_ranges, age_codes))
df_transformed["AgeCategory"] = df_transformed["AgeCategory"].replace(age_range_to_code)
df_transformed["AgeCategory"].value_counts().sort_index()

# Categorizing and Encoding "GenHGeneralHealthealth" into 5 distinct groups.
df_transformed["GeneralHealth"].unique()
df_transformed["GeneralHealth"] = df_transformed["GeneralHealth"].replace(
    {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
)
df_transformed["GeneralHealth"].value_counts().sort_index()


# One-hot Encoding "Race" into 6 groups, which will help us identify an individual's race.
df_transformed["RaceEthnicityCategory"].unique()
df_race = pd.get_dummies(df_transformed["RaceEthnicityCategory"]).astype(int)
df_transformed = pd.concat([df_transformed, df_race], axis=1)
df_transformed.drop(columns=["RaceEthnicityCategory"], inplace=True)
df_transformed

# Categorizing and Encoding "SmokerStatus" into 4 distinct groups.
df_transformed["SmokerStatus"].unique()
df_transformed["SmokerStatus"] = df_transformed["SmokerStatus"].replace(
    {
        "Never smoked": 0,
        "Former smoker": 1,
        "Current smoker - now smokes some days": 2,
        "Current smoker - now smokes every day": 3,
    }
)
df_transformed["SmokerStatus"].value_counts().sort_index()

# Categorizing and Encoding "ECigaretteUsage" into 4 distinct groups.
df_transformed["ECigaretteUsage"].unique()
df_transformed["ECigaretteUsage"] = df_transformed["ECigaretteUsage"].replace(
    {
        "Never used e-cigarettes in my entire life": 0,
        "Not at all (right now)": 1,
        "Use them some days": 2,
        "Use them every day": 3,
    }
)
df_transformed["ECigaretteUsage"].value_counts().sort_index()

"""
Overall, this transformed data will help us in the next step, as it ensures standardized inputs,
and simplifies the categorical variables into numerical formats. The encoding techniques
applied make the dataset more accessible and manageable for us in the next phase, which is
data modeling.
"""

# Saving the transformed data into a csv file:
df_transformed.to_csv("../data/processed/heart_2022_transformed.csv", index=False)
