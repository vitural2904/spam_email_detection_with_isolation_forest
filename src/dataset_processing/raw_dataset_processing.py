import pandas as pd

df = pd.read_csv('Dataset/raw/processed_data.csv')

df_cleaned = df.drop('message', axis = 'columns')

df_cleaned = df_cleaned.rename(columns={'processed' : 'plaintext'})

# Sepreate ham email and spam email
df_normal = df_cleaned[df_cleaned['label'] == 0].copy()
df_anomalies = df_cleaned[df_cleaned['label'] == 1].copy()

# Create training datasets

# Dataset 1: 7500 spam / 10000 ham
num_total = 17500
num_anom1 = 7500
num_norm1 = num_total - num_anom1

train_1 = pd.concat([
    df_anomalies.iloc[:num_anom1],
    df_normal.iloc[:num_norm1]
], ignore_index=True)

train_1.to_csv('Dataset/final/training_dataset_1.csv', index=False)

# Dataset 2: 5000 spam / 12500 ham
num_anom2 = 5000
num_norm2 = num_total - num_anom2

train_30 = pd.concat([
    df_anomalies.iloc[:num_anom2],
    df_normal.iloc[:num_norm2]
], ignore_index=True)

train_30.to_csv('Dataset/final/training_dataset_2.csv', index=False)

# Create testing dataset (lines haven't used in Dataset 1)

used_anom_indices = df_anomalies.iloc[:num_anom1].index
used_norm_indices = df_normal.iloc[:num_norm1].index

remaining_anom = df_anomalies.drop(index=used_anom_indices)
remaining_norm = df_normal.drop(index=used_norm_indices)

testing_dataset = pd.concat([remaining_anom, remaining_norm], ignore_index=True)
testing_dataset.to_csv('Dataset/final/testing_dataset.csv', index=False)