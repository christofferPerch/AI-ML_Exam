import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Data exploration 

df = pd.read_csv("../data/heart_2022_removed_columns.csv")

pd.set_option('display.max_columns', None)

df.info()

df.head()

df.describe()

df.mode().iloc[0]

# seaborn settings

sns.set_theme(style="whitegrid")

# Creating a bar chart for the distribution of General Health Status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='GeneralHealth', order=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
plt.title('Distribution of General Health Status')
plt.xlabel('General Health')
plt.ylabel('Count')
plt.show()

# Histograms for Physical and Mental Health Days
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['PhysicalHealthDays'], bins=30, kde=True)
plt.title('Histogram of Physical Health Days')
plt.xlabel('Physical Health Days')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['MentalHealthDays'], bins=30, kde=True)
plt.title('Histogram of Mental Health Days')
plt.xlabel('Mental Health Days')
plt.ylabel('Frequency')
plt.show()

# Creating bar charts for HadHeartAttack, HadAngina, HadStroke
plt.figure(figsize=(15, 5))

# Subplot for 'HadHeartAttack'
plt.subplot(1, 3, 1)
sns.countplot(data=df, x='HadHeartAttack')
plt.title('Distribution of Heart Attacks')
plt.xlabel('Had Heart Attack')
plt.ylabel('Count')

# Subplot for 'HadAngina'
plt.subplot(1, 3, 2)
sns.countplot(data=df, x='HadAngina')
plt.title('Distribution of Angina Cases')
plt.xlabel('Had Angina')
plt.ylabel('Count')

# Subplot for 'HadStroke'
plt.subplot(1, 3, 3)
sns.countplot(data=df, x='HadStroke')
plt.title('Distribution of Stroke Cases')
plt.xlabel('Had Stroke')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Creating binary flags for each condition
df['HeartAttackFlag'] = (df['HadHeartAttack'] == 'Yes').astype(int)
df['AnginaFlag'] = (df['HadAngina'] == 'Yes').astype(int)
df['StrokeFlag'] = (df['HadStroke'] == 'Yes').astype(int)

# Creating a column for any heart condition
df['HeartDisease'] = df[['HeartAttackFlag', 'AnginaFlag', 'StrokeFlag']].max(axis=1)

# Calculating the total number of each condition
total_incidents = df['HeartDisease'].sum()
total_rows = len(df)
no_incidents = total_rows - total_incidents

# Data for plotting
conditions = ['Any Heart Condition', 'No Heart Condition']
counts = [total_incidents, no_incidents]

# Creating the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=conditions, y=counts, palette='muted')
plt.title('Total Incidents of Any Heart Condition Compared to No Incidents')
plt.xlabel('Condition')
plt.ylabel('Number of Incidents')
plt.show()

# Define the order of age categories manually
age_order = [
    'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39',
    'Age 40 to 44', 'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59',
    'Age 60 to 64', 'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79',
    'Age 80 or older'
]

# Plotting number of heart disease incidents by age category
plt.figure(figsize=(14, 7))
sns.countplot(data=df, x='AgeCategory', hue='HeartDisease', palette='coolwarm', order=age_order)
plt.title('Heart Disease Incidents by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()

df.head()