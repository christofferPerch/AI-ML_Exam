import pandas as pd
import pyodbc

# Load the CSV file
file_path = 'Medicine_Details.csv'
df = pd.read_csv(file_path)

# Database connection setup using Windows authentication
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;DATABASE=HeartDisease;Trusted_Connection=yes;')
cursor = conn.cursor()

# Insert Manufacturer data
manufacturer_map = {}
for manufacturer in df['Manufacturer'].unique():
    cursor.execute("INSERT INTO Manufacturer (Name) OUTPUT INSERTED.ManufacturerID VALUES (?)", manufacturer)
    manufacturer_id = cursor.fetchone()[0]
    manufacturer_map[manufacturer] = manufacturer_id
conn.commit()

# Insert Product, SideEffect, and Review data
for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO Product (ProductName, GenericName, Description, ImageURL, ManufacturerID)
        OUTPUT INSERTED.ProductID
        VALUES (?, ?, ?, ?, ?)
    """, row['Medicine Name'], row['Composition'], row['Uses'], row['Image URL'], manufacturer_map[row['Manufacturer']])
    product_id = cursor.fetchone()[0]

    # Insert side effects
    side_effects = row['Side_effects'].split(', ')
    for effect in side_effects:
        cursor.execute("INSERT INTO SideEffect (ProductID, Effect) VALUES (?, ?)", product_id, effect.strip())

    # Insert reviews
    cursor.execute("""
        INSERT INTO Review (ProductID, ExcellentReviewPercent, AverageReviewPercent, PoorReviewPercent)
        VALUES (?, ?, ?, ?)
    """, product_id, row['Excellent Review %'], row['Average Review %'], row['Poor Review %'])

conn.commit()
conn.close()
