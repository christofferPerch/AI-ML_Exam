import pickle
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler


def transform_data(data):
    # make a dataframe out of testdata
    df_transformed = pd.DataFrame(data)
    # Feature Scaling / Standardization for numerical features:
    #numeric_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]
    #scaler = StandardScaler()
    #df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    numeric_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]
    scaler = StandardScaler()
    df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
    df_transformed.head()

    df_transformed = df_transformed.replace({"No": 0, "Yes": 1})

    # Encoding "Sex" with 0/1.
    df_transformed["Sex"] = df_transformed["Sex"].replace({"Female": 0, "Male": 1})
    df_transformed["HadDiabetes"].unique()

    # Encoding "Diabetic" with 0/1, but first we replace everything with No / Yes.
    df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace(
        {
            "No, pre-diabetes or borderline diabetes": "No",
            "Yes, but only during pregnancy (female)": "Yes",
        }
    )
    df_transformed["HadDiabetes"] = df_transformed["HadDiabetes"].replace(
        {"No": 0, "Yes": 1}
    )

    ### ORDINAL VARIABLES ###
    # Categorizing and Encoding "AgeCategory" into 13 groups.
    expected_age_categories = [
    'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39',
    'Age 40 to 44', 'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59',
    'Age 60 to 64', 'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79',
    'Age 80 or older'
    ]
    
    # Create a dictionary mapping each age category to a unique integer
    age_range_to_code = {age: idx for idx, age in enumerate(expected_age_categories)}

# Encode the 'AgeCategory' with this mapping
    df_transformed['AgeCategory'] = df_transformed['AgeCategory'].replace(age_range_to_code)

    df_transformed["GeneralHealth"].unique()
    df_transformed["GeneralHealth"] = df_transformed["GeneralHealth"].replace(
        {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    )

    ### NOMINAL VARIABLES ###
    # One-hot Encoding "RaceEthnicityCategory" into 5 groups.
    # "Black only, Non-Hispanic",Hispanic,"Multiracial, Non-Hispanic","Other race only, Non-Hispanic","White only, Non-Hispanic"
    df_transformed["RaceEthnicityCategory"].unique()
    df_race = pd.get_dummies(df_transformed["RaceEthnicityCategory"]).astype(int)

    # Add missing columns that the model expects
    expected_columns = [
        "Black only, Non-Hispanic",
        "Hispanic",
        "Multiracial, Non-Hispanic",
        "Other race only, Non-Hispanic",
        "White only, Non-Hispanic",
    ]
    # Add missing column with default value 0
    for col in expected_columns:
        if col not in df_race.columns:
            df_race[col] = 0
    # Reorder columns to match the model's expectations
    df_race = df_race[expected_columns]

    df_transformed = pd.concat([df_transformed, df_race], axis=1)
    df_transformed.drop(columns=["RaceEthnicityCategory"], inplace=True)
    df_transformed

    df_transformed["SmokerStatus"].unique()
    df_transformed["SmokerStatus"] = df_transformed["SmokerStatus"].replace(
        {
            "Never smoked": 0,
            "Former smoker": 1,
            "Current smoker - now smokes some days": 2,
            "Current smoker - now smokes every day": 3,
        }
    )
    df_transformed.head()

    df_transformed["ECigaretteUsage"].unique()
    df_transformed["ECigaretteUsage"] = df_transformed["ECigaretteUsage"].replace(
        {
            "Never used e-cigarettes in my entire life": 0,
            "Not at all (right now)": 1,
            "Use them some days": 2,
            "Use them every day": 3,
        }
    )
    return df_transformed


# Create a dictionary with the test data to test models
test_data_no_heart_disease = {
    "Sex": ["Female"],
    "GeneralHealth": ["Excellent"],
    "PhysicalHealthDays": [0],
    "MentalHealthDays": [0],
    "PhysicalActivities": ["No"],
    "SleepHours": [6],
    "HadAsthma": ["No"],
    "HadDepressiveDisorder": ["No"],
    "HadKidneyDisease": ["No"],
    "HadDiabetes": ["No"],
    "DifficultyWalking": ["No"],
    "SmokerStatus": ["Never smoked"],
    "ECigaretteUsage": ["Never used e-cigarettes in my entire life"],
    "RaceEthnicityCategory": ["White only, Non-Hispanic"],
    "AgeCategory": ["Age 80 or older"],
    "BMI": [26.57],
    "AlcoholDrinkers": ["No"],
    "HIVTesting": ["No"],
}
test_data_heart_disease = {
    "Sex": ["Male"],
    "GeneralHealth": ["Poor"],
    "PhysicalHealthDays": [30],
    "MentalHealthDays": [30],
    "PhysicalActivities": ["No"],
    "SleepHours": [6],
    "HadAsthma": ["Yes"],
    "HadDepressiveDisorder": ["Yes"],
    "HadKidneyDisease": ["Yes"],
    "HadDiabetes": ["Yes"],
    "DifficultyWalking": ["Yes"],
    "SmokerStatus": ["Current smoker - now smokes every day"],
    "ECigaretteUsage": ["Never used e-cigarettes in my entire life"],
    "RaceEthnicityCategory": ["White only, Non-Hispanic"],
    "AgeCategory": ["Age 80 or older"],
    "BMI": [30],
    "AlcoholDrinkers": ["No"],
    "HIVTesting": ["No"],
}
# Transform the test data
test_data_transformed = transform_data(test_data_no_heart_disease)
# test_data_transformed = transform_data(test_data_heart_disease)

# test models with the transformed data
# Load the trained model from the file
### logistic regression
#with open("model.pkl", "rb") as file:
#    model = pickle.load(file)
#prediction = model.predict_proba(test_data_transformed)
#percentage = prediction[:, 1] * 100
#percentage.item()

### random forest
# with open("model_forest.pkl", "rb") as file:
#     model = pickle.load(file)
# prediction = model.predict_proba(test_data_transformed)
# percentage = prediction[:, 1] * 100
# percentage.item()

# ### tensorflow model
# with open("model_neural.pkl", "rb") as file:
#     model = pickle.load(file)
# prediction = model.predict(test_data_transformed)
# percentage = prediction * 100
# percentage.item()
