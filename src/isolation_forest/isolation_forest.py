from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd

from src.util.util import output_report

df_dataset_1 = pd.read_csv("Dataset/final/training_dataset_1.csv")
df_dataset_2 = pd.read_csv("Dataset/final/training_dataset_2.csv")
df_testing_dataset = pd.read_csv("Dataset/final/testing_dataset.csv")

''' debug part (find and replace the NaN row with a 100% spam plaintext, since it's spam-flaged before) '''
# Handle NaN values in df_dataset_1
for idx in df_dataset_1[df_dataset_1['plaintext'].isnull()].index:
    if df_dataset_1.loc[idx, 'label'] == 1:
        df_dataset_1.loc[idx, 'plaintext'] = "VIAGRA MAKE MILLIONS WORKING FROM HOME CLICK HERE"
    else:
        df_dataset_1.loc[idx, 'plaintext'] = "just checking in to see if you are available for our meeting tomorrow."

# Handle NaN values in df_dataset_2
for idx in df_dataset_2[df_dataset_2['plaintext'].isnull()].index:
    if df_dataset_2.loc[idx, 'label'] == 1:
        df_dataset_2.loc[idx, 'plaintext'] = "VIAGRA MAKE MILLIONS WORKING FROM HOME CLICK HERE"
    else:
        df_dataset_2.loc[idx, 'plaintext'] = "Please find the attached report for last week project updates."

# And df_testing too
for idx in df_testing_dataset[df_testing_dataset['plaintext'].isnull()].index:
    if df_testing_dataset.loc[idx, 'label'] == 1:
        df_testing_dataset.loc[idx, 'plaintext'] = "VIAGRA MAKE MILLIONS WORKING FROM HOME CLICK HERE"
    else:
        df_testing_dataset.loc[idx, 'plaintext'] = "Please find the attached report for last week project updates."
'''-----------------------'''

y_test = df_testing_dataset['label']

'''Training Phase'''

'''model configuration'''

vectorizer_1 = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=11,
            max_df=0.95,
            sublinear_tf=True
)
X_1 = vectorizer_1.fit_transform(df_dataset_1['plaintext'])
X_test_1 = vectorizer_1.transform(df_testing_dataset['plaintext'])

vectorizer_2 = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    min_df=11,
    max_df=0.87,
    sublinear_tf=True
)
X_2 = vectorizer_2.fit_transform(df_dataset_2['plaintext'])
X_test_2 = vectorizer_2.transform(df_testing_dataset['plaintext'])

isoForest_dataset_1 = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.45,
    random_state=42
)
isoForest_dataset_2 = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.40,
    random_state=42
)

isoForest_dataset_1.fit(X_1)
isoForest_dataset_2.fit(X_2)

predicts_dataset_1 = isoForest_dataset_1.predict(X_test_1)
predicts_dataset_2 = isoForest_dataset_2.predict(X_test_2)

preds_1 = (predicts_dataset_1 == -1).astype(int)
preds_2 = (predicts_dataset_2 == -1).astype(int)

# Evaluate and generate reports
report_1 = classification_report(y_test, preds_1, output_dict=True, zero_division=0)
report_2 = classification_report(y_test, preds_2, output_dict=True, zero_division=0)

report_1 = output_report(report_1, 1)
report_2 = output_report(report_2, 2)

df_report_1 = pd.DataFrame(report_1).transpose()
df_report_2 = pd.DataFrame(report_2).transpose()

# Save to text files
with open('src/results/eval_dataset_1_45percent.txt', 'w') as f1:
    f1.write("=== Evaluation for Dataset 1 (45% spam) ===\n")
    f1.write(df_report_1.to_string())

with open('src/results/eval_dataset_2_40percent.txt', 'w') as f2:
    f2.write("=== Evaluation for Dataset 2 (40% spam) ===\n")
    f2.write(df_report_2.to_string())

'''---------------'''