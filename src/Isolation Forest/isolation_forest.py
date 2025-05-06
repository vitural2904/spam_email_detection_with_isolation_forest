from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib
import pandas as pd

'''Training Phase'''

'''model configuration'''

isoForest_dataset_1 = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.20,  # Dataset 1 has 20% of spam email
    random_state=42
)

isoForest_dataset_2 = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.30,  # Dataset 2 has 30% of spam email
    random_state=42
)

'''-----------------'''

# load tf-idf matrix from joblib
X_1 = joblib.load('src/joblib_temp/X_1_vectorized.joblib')
X_2 = joblib.load('src/joblib_temp/X_2_vectorized.joblib')
X_test = joblib.load('src/joblib_temp/X_test_vectorized.joblib')

# get the brief view of the dataspace
print("Dataset 1's TF-IDF matrix size:", X_1.shape)
print("Dataset 2's TF-IDF matrix size:", X_2.shape)

# fitting, transforming
isoForest_dataset_1.fit(X_1)
isoForest_dataset_2.fit(X_2)

'''-----------------'''


'''Evaluations on dataset model haven't seen before (testing_dataset.csv)'''

'''building path length, get anomaly score'''

# Dataset 1
scores_dataset_1 = isoForest_dataset_1.decision_function(X_test)
predicts_dataset_1 = isoForest_dataset_1.predict(X_test)  # -1 prediction label is anomaly

# Dataset 2
scores_dataset_2 = isoForest_dataset_2.decision_function(X_test)
predicts_dataset_2 = isoForest_dataset_2.predict(X_test)  # -1 prediction label is anomaly

'''---------------'''

# Load true labels from testing dataset
df_test = pd.read_csv('Dataset/final/testing_dataset.csv')
y_test = df_test['label']

# Convert predicts from {-1, 1} to {1, 0}
preds_1 = (predicts_dataset_1 == -1).astype(int)
preds_2 = (predicts_dataset_2 == -1).astype(int)

# Evaluate and generate reports
report_1 = classification_report(y_test, preds_1, digits=4)
report_2 = classification_report(y_test, preds_2, digits=4)

# Save to text files
with open('src/results/eval_dataset_1_20percent.txt', 'w') as f1:
    f1.write("=== Evaluation for Dataset 1 (20% spam) ===\n")
    f1.write(report_1)

with open('src/results/eval_dataset_2_30percent.txt', 'w') as f2:
    f2.write("=== Evaluation for Dataset 2 (30% spam) ===\n")
    f2.write(report_2)

'''---------------'''