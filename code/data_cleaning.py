import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/processed/heart_2022.csv")

# Making a copy from the original dataframe:
df_cleaned = df.copy()

# We decided to drop all the NaN values, since they're only a fraction of the dataset:
df_cleaned.info()
df_cleaned = df.dropna()


""" 
Assuming the presence of Heart Disease is indicated 
by any "Yes" in these three Heart conditions.

We create a new column called "HadHeartDisease" derived from the three Heart conditions.
"""
df_cleaned["HadHeartDisease"] = (
    (df_cleaned["HadHeartAttack"] == "Yes")
    | (df_cleaned["HadAngina"] == "Yes")
    | (df_cleaned["HadStroke"] == "Yes")
).astype(int)

""" 
We decided to drop "HeightInMeters" and "WeightInKilograms" since "BMI" will give
a better overall representation of those two columns.

We also drop the three previous Heart conditions from the data,
since we now have "HadHeartDisease" instead.
"""
df_cleaned.drop(
    columns=[
        "HeightInMeters",
        "WeightInKilograms",
        "HadHeartAttack",
        "HadAngina",
        "HadStroke",
    ],
    inplace=True,
)


## Identifying and removing outliers ##

""" 
These 4 columns will help us identify outliers,
since their datatypes are suitable.

"""
columns_to_plot = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]

# These box plots will display the outliers outside the whiskers:
plt.figure(figsize=(15, 5))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(1, len(columns_to_plot), i)
    sns.boxplot(y=df[column])
    plt.title(f"Box plot of {column}")

plt.tight_layout()
plt.show()


# Removing Outliers using the IQR Range:
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
columns_to_remove_outliers = [
    "BMI",
    "SleepHours",
    "PhysicalHealthDays",
    "MentalHealthDays",
]
for column in columns_to_remove_outliers:
    df_no_outliers = remove_outliers_iqr(df_cleaned, column)

df_no_outliers.info()

""" 
We have decided to keep the outliers, since they may be significant in
our project about Heart Disease. The IQR method above is the technique we would've applied,
if we decided to remove outliers.

We decided to test with and without outliers, and the model gave the best
results with outliers, which also showcase the real world, where the majority does not
have a Heart condition and therefore might the outliers help us identify the chance of
getting a Heart Disease.
"""

# Saving csv file with outliers.
df_cleaned.to_csv("../data/processed/heart_2022_cleaned", index=False)

# Saving csv file for no outliers:
# df_no_outliers.to_csv("../data/heart_2022_cleaned", index=False)
