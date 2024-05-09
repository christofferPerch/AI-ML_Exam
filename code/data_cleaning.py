import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/heart_2022_removed_columns.csv")

pd.set_option('display.max_columns', None)

df.info()

df.head()

df_cleaned = df.dropna()

df_cleaned.info()

df_cleaned.head()



# Assuming the presence of heart disease is indicated by any 'Yes' in these three columns
df_cleaned['HadHeartDisease'] = ((df_cleaned['HadHeartAttack'] == 'Yes') | 
                          (df_cleaned['HadAngina'] == 'Yes') | 
                          (df_cleaned['HadStroke'] == 'Yes')).astype(int)

df_cleaned.drop(columns=["HeightInMeters","WeightInKilograms","HadHeartAttack","HadAngina","HadStroke"],inplace=True)

df_cleaned

# Columns suitable for box plots based on their data type
columns_to_plot = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours','BMI']

# Plotting
plt.figure(figsize=(15, 5))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(1, len(columns_to_plot), i)
    sns.boxplot(y=df[column])
    plt.title(f"Box plot of {column}")

plt.tight_layout()
plt.show()



# Removing Outliers using the IQR Range (function used in OLA-1):
def remove_outliers_iqr(dataset, col):
    """
    Function to mark values as outliers using the IQR (Interquartile Range) method.

    Explanation:
    A common method is to use the Interquartile Range (IQR),
    which is the range between the first quartile (25%) and the third quartile (75%)
    of the data. Outliers are often considered as data points that lie outside 1.5 times
    the IQR below the first quartile and above the third quartile.

    Args:
        dataset (pd.DataFrame): The dataset.
        col (string): The column you want apply outlier detection to.
    """

    Q1 = dataset[col].quantile(0.15)
    Q3 = dataset[col].quantile(0.85)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering the DataFrame to remove outliers:
    df_filtered = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    return df_filtered


# Removing outliers for each column:
columns_to_remove_outliers = ["BMI", "SleepHours", "PhysicalHealthDays","MentalHealthDays"]
for column in columns_to_remove_outliers:
    df_no_outliers = remove_outliers_iqr(df_cleaned, column)
    
df_no_outliers.info()

df_no_outliers.to_csv("../data/heart_2022_cleaned_removed_outliers.csv",index=False)

# Checking the NaN Count again to assure the outliers has been removed.
df_cleaned.info()

df_cleaned.to_csv("../data/heart_2022_cleaned_with_outliers.csv",index=False)



# Continue from the filtering step as before
#heavy_individuals = df_cleaned[df_cleaned['PhysicalHealthDays'] > 20]
#heart_disease_count = heavy_individuals['HadHeartDisease'].value_counts()

#print("Distribution of heart disease among individuals weighing more than 250kg:")
#print(heart_disease_count)


