import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Read the CSV file
df = pd.read_csv('heart_2022_cleaned.csv')

# Add the CreatedAt column with the current date and time
df['CreatedAt'] = datetime.now()

# Create a connection to the SQL Server database using Windows Authentication
server_name = 'localhost'
database_name = 'HeartDisease'
engine = create_engine(f'mssql+pyodbc://@{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes')

# Insert the data into the HealthSurvey table
df.to_sql('HealthSurvey', con=engine, if_exists='append', index=False)
