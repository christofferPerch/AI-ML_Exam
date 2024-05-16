import pandas as pd

# from sqlalchemy import create_engine


def load_data_from_sql():
    df = pd.read_csv("../../data/processed/temp.csv")

    # # Create a connection to the SQL Server database using Windows Authentication
    # server_name = 'localhost'
    # database_name = 'HeartDisease'
    # engine = create_engine(f'mssql+pyodbc://@{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes')

    # # Define the query to select all columns except Id and CreatedAt
    # query = """
    # SELECT Sex, GeneralHealth, PhysicalHealthDays, MentalHealthDays, PhysicalActivities, SleepHours,
    #        HadAsthma, HadDepressiveDisorder, HadKidneyDisease, HadDiabetes, DifficultyWalking,
    #        SmokerStatus, ECigaretteUsage, RaceEthnicityCategory, AgeCategory, BMI,
    #        AlcoholDrinkers, HIVTesting, HadHeartDisease
    # FROM HealthSurvey
    # """

    # # Load data into a pandas DataFrame
    # df = pd.read_sql(query, engine)

    return df


# Usage
df = load_data_from_sql()
print(df.head())
