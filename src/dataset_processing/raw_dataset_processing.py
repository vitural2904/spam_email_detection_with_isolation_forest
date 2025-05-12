import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset/raw/processed_data.csv')

df_cleaned = df.drop('message', axis = 'columns')

df_cleaned = df_cleaned.rename(columns={'processed' : 'plaintext'})

# Sepreate ham email and spam email
df_normal = df_cleaned[df_cleaned['label'] == 0].copy()
df_anomalies = df_cleaned[df_cleaned['label'] == 1].copy()

# creating datasets
total_samples = 20000

# --- Dataset 1: 45% spam (9000 spam, 11000 normal) ---
anom_1 = df_anomalies.sample(n=9000, random_state=42, replace=True)
norm_1 = df_normal.sample(n=11000, random_state=42, replace=True)

train_1 = pd.concat([anom_1, norm_1], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
train_1.to_csv('Dataset/final/training_dataset_1.csv', index=False)

# --- Dataset 2: 40% spam (8000 spam, 12000 normal) ---
anom_2 = df_anomalies.sample(n=8000, random_state=43, replace=True)
norm_2 = df_normal.sample(n=12000, random_state=43, replace=True)

train_2 = pd.concat([anom_2, norm_2], ignore_index=True).sample(frac=1, random_state=43).reset_index(drop=True)
train_2.to_csv('Dataset/final/training_dataset_2.csv', index=False)

# --- Testing dataset: 10000 email, 50% spam ---
anom_test = df_anomalies.sample(n=5000, random_state=44, replace=True)
norm_test = df_normal.sample(n=5000, random_state=44, replace=True)

test_set = pd.concat([anom_test, norm_test], ignore_index=True).sample(frac=1, random_state=44).reset_index(drop=True)
test_set.to_csv('Dataset/final/testing_dataset.csv', index=False)