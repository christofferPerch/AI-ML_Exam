## DATA TRANSFORMATION ##
import pandas as pd
from sklearn.preprocessing import StandardScaler


def transform_data(df, predict=False):

    ## NUMERICAL VARIABLES ##
    # TRANSFORMING RATIO VARIABLES #

    # Feature Scaling:
    # numeric_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]
    # scaler = StandardScaler()
    # df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

    """
    By feature scaling we came to the conclusion that they had less influence on the overall prediction,
    so we decided to remove it after observing the differences.
    We decided to keep this example to demonstrate how we would have feature scaled our numeric variables.
    """

    ## CATEGORICAL VARIABLES ##
    # ENCODING BINARY VARIABLES #

    # Initially we have decided to replace all the existing binary values to 0 and 1's:
    df = df.replace({"No": 0, "Yes": 1})

    # Encoding "Sex" with 0 and 1's.:
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})
    df["Sex"].value_counts()

    # Encoding "HadDiabetes" with 0 and 1's, but first we replace the strings with No and Yes:
    df["HadDiabetes"] = df["HadDiabetes"].replace(
        {
            "No, pre-diabetes or borderline diabetes": "No",
            "Yes, but only during pregnancy (female)": "Yes",
        }
    )
    df["HadDiabetes"] = df["HadDiabetes"].replace({"No": 0, "Yes": 1})
    df["HadDiabetes"].value_counts()

    # ENCODING NOMINAL VARIABLES #

    # One-hot Encoding "Race" into 6 groups, which will help us identify an individual's race:
    df["RaceEthnicityCategory"].unique()
    df_race = pd.get_dummies(df["RaceEthnicityCategory"]).astype(int)
    if predict:
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
    df = pd.concat([df, df_race], axis=1)
    df.drop(columns=["RaceEthnicityCategory"], inplace=True)

    # Encoding Ordinal Variables #

    # Categorizing and Encoding "AgeCategory" into 13 groups:
    age_ranges = [
        "Age 18 to 24",
        "Age 25 to 29",
        "Age 30 to 34",
        "Age 35 to 39",
        "Age 40 to 44",
        "Age 45 to 49",
        "Age 50 to 54",
        "Age 55 to 59",
        "Age 60 to 64",
        "Age 65 to 69",
        "Age 70 to 74",
        "Age 75 to 79",
        "Age 80 or older",
    ]
    age_codes, _ = pd.factorize(age_ranges, sort=True)
    age_range_to_code = dict(zip(age_ranges, age_codes))
    df["AgeCategory"] = df["AgeCategory"].replace(age_range_to_code)
    df["AgeCategory"].value_counts().sort_index()

    # Categorizing and Encoding "GenHGeneralHealthealth" into 5 distinct groups:
    df["GeneralHealth"].unique()
    df["GeneralHealth"] = df["GeneralHealth"].replace(
        {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    )
    df["GeneralHealth"].value_counts().sort_index()

    # Categorizing and Encoding "SmokerStatus" into 4 distinct groups:
    df["SmokerStatus"].unique()
    df["SmokerStatus"] = df["SmokerStatus"].replace(
        {
            "Never smoked": 0,
            "Former smoker": 1,
            "Current smoker - now smokes some days": 2,
            "Current smoker - now smokes every day": 3,
        }
    )
    df["SmokerStatus"].value_counts().sort_index()

    # Categorizing and Encoding "ECigaretteUsage" into 4 distinct groups:
    df["ECigaretteUsage"].unique()
    df["ECigaretteUsage"] = df["ECigaretteUsage"].replace(
        {
            "Never used e-cigarettes in my entire life": 0,
            "Not at all (right now)": 1,
            "Use them some days": 2,
            "Use them every day": 3,
        }
    )
    df["ECigaretteUsage"].value_counts().sort_index()

    # Saving the transformed data into a csv file:
    df.to_csv("../../data/processed/heart_2022_transformed.csv", index=False)
    return df


if __name__ == "__main__":
    # Load data
    df_cleaned = pd.read_csv("../../data/processed/heart_2022_cleaned.csv")

    # Apply transformations
    df_transformed = transform_data(df_cleaned)

    # Save transformed data
    df_transformed.to_csv(
        "../../data/processed/heart_2022_transformed.csv", index=False
    )


"""
Overall, this transformed data will help us in the next phase, 
as it ensures standardized inputs, and simplifies the categorical variables into numerical formats. 
The encoding techniques applied makes the dataset more accessible and 
manageable for us in the next phase, which is data modeling.
"""
