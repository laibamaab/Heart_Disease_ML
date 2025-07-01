# IMPORT NECESSARY LIBRARIES
import pandas as pd

# LOAD THE DATASET FROM COMPUTER
df = pd.read_csv('heart.csv')
print(df.head()) # Display first few rows

# Show basic info
print(df.info())

# Summary statistics
print(df.describe(include='all'))

#check for missing values
print(df.isnull().sum())

# HANDLE MISSING VALUES
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())

# Check unique values
print("Unique 'thal' values:", df['thal'].unique())

# NOISE REDUCTION — smoothing 'thalach'
df['thalach_smoothed'] = df['thalach'].rolling(window=3).mean().bfill()

print("Duplicate rows:", df.duplicated().sum()) #check duplicates

# View age distribution
df['age'].describe()
# Remove unrealistic ages
df = df[(df['age'] >= 5) & (df['age'] <= 100)]

# Before encoding, separate target variable
y = df['target']
# Drop target from features before encoding
X = df.drop('target', axis=1)

# Export pre-processed dataset
df.to_csv('heart_cleaned.csv', index=False)
