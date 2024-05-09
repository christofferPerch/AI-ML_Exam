import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np


df_transformed = pd.read_csv("../data/heart_2022_cleaned_removed_outliers.csv")

pd.set_option('display.max_columns', None)


df_transformed.head()

df_transformed.info()

# Feature Scaling / Standardization for numerical features:


numeric_cols = [
    "PhysicalHealthDays",
    "MentalHealthDays",
    "SleepHours",
    "BMI"
]
scaler = StandardScaler()
df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

df_transformed.head()

df_transformed = df_transformed.replace({"No": 0, "Yes": 1})

# Encoding "Sex" with 0/1.
df_transformed["Sex"] = df_transformed["Sex"].replace({"Female": 0, "Male": 1})
# df["Sex"].value_counts()
df_transformed['HadDiabetes'].unique()

# Encoding "Diabetic" with 0/1, but first we replace everything with No / Yes.
df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace(
    {
        "No, pre-diabetes or borderline diabetes": "No",
        "Yes, but only during pregnancy (female)": "Yes",
    }
)
df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace({"No": 0, "Yes": 1})
# df_transformed["Diabetic"].value_counts()


### ORDINAL VARIABLES ###
# Categorizing and Encoding "AgeCategory" into 13 groups.
age_ranges = df_transformed["AgeCategory"].unique()
age_codes, _ = pd.factorize(age_ranges, sort=True)
age_range_to_code = dict(zip(age_ranges, age_codes))
df_transformed["AgeCategory"] = df_transformed["AgeCategory"].replace(age_range_to_code)

# df_transformed["AgeCategory"].value_counts().sort_index()


# Categorizing and Encoding "BMI" into 4 different groups.
#bmi_categories = ['Underweight (< 18.5)', 'Normal weight (18.5 - 25.0)', 'Overweight (25.0 - 30.0)', 'Obese (30 <)']
#bmi_bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
#df_transformed['BMI'] = pd.cut(df_transformed['BMI'], bins=bmi_bins, labels=bmi_categories)

#dict_BMI = {category: code for code, category in enumerate(bmi_categories)}
#df_transformed['BMI'] = df_transformed['BMI'].map(dict_BMI)
#df_transformed["BMI"].value_counts()

# Categorizing and Encoding "GenHealth" into 5 different groups.
df_transformed['GeneralHealth'].unique()
df_transformed["GeneralHealth"] = df_transformed["GeneralHealth"].replace(
    {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
)
# df_transformed["GenHealth"].value_counts()


### NOMINAL VARIABLES ###
# One-hot Encoding "Race" into 6 groups.
df_transformed['RaceEthnicityCategory'].unique()
df_race = pd.get_dummies(df_transformed["RaceEthnicityCategory"]).astype(int)
df_transformed = pd.concat([df_transformed,df_race],axis=1)
df_transformed.drop(columns=["RaceEthnicityCategory"],inplace=True)
df_transformed

#df_transformed = df_transformed.replace({"False": 0, "True": 1})
# race_columns = [col for col in df_transformed.columns if col.startswith("Race_")]
# race_value_counts = df[race_columns].sum().sort_values(ascending=False)
# race_value_counts
df_transformed['SmokerStatus'].unique()
df_transformed["SmokerStatus"] = df_transformed["SmokerStatus"].replace(
    {"Never smoked": 0, "Former smoker": 1,
     "Current smoker - now smokes some days": 2,
     "Current smoker - now smokes every day": 3}
)
df_transformed.head()


df_transformed['ECigaretteUsage'].unique()
df_transformed["ECigaretteUsage"] = df_transformed["ECigaretteUsage"].replace(
    {"Never used e-cigarettes in my entire life": 0, "Not at all (right now)": 1,
     "Use them some days": 2,
     "Use them every day": 3}
)
df_transformed.head()

df_transformed.to_csv("../data/heart_2022_transformed_no_outliers.csv",index=False)



