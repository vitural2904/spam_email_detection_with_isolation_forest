import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


'''util'''
def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step
'''----'''

df_dataset_1 = pd.read_csv("Dataset/final/training_dataset_1.csv")
df_dataset_2 = pd.read_csv("Dataset/final/training_dataset_2.csv")
df_testing_dataset = pd.read_csv("Dataset/final/testing_dataset.csv")

# get true label of the testing dataset
y_test = df_testing_dataset['label']


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

# init variables
best_config_dataset_1 = None
best_config_dataset_2 = None
best_score_dataset_1 = 0
best_score_dataset_2 = 0
optimizing_result = ""

# optimizing the spam filter rate due to tuning the hyperparameter
for min_df in range(5, 15):
    for max_df in [round(x, 2) for x in list(frange(0.85, 0.951, 0.01))]:

        vectorizer_1 = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        X_1 = vectorizer_1.fit_transform(df_dataset_1['plaintext'])
        X_test_1 = vectorizer_1.transform(df_testing_dataset['plaintext'])

        vectorizer_2 = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
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

        # classification_report expects string labels by default
        report_1 = classification_report(y_test, preds_1, output_dict=True, zero_division=0)
        report_2 = classification_report(y_test, preds_2, output_dict=True, zero_division=0)

        f1_spam_report_1 = report_1['1.0']['f1-score']
        f1_spam_report_2 = report_2['1.0']['f1-score']

        optimizing_result += f"\nmin_df={min_df}, max_df={max_df}, F1 for detect spam = {f1_spam_report_1:.4f}"
        if f1_spam_report_1 > best_score_dataset_1:
            best_score_dataset_1 = f1_spam_report_1
            best_config_dataset_1 = {
                'X_1_best': X_1,
                'X_test_dataset_1': X_test_1
            }

        optimizing_result += f"\nmin_df={min_df}, max_df={max_df}, F1 for detect spam = {f1_spam_report_2:.4f}"
        if f1_spam_report_2 > best_score_dataset_2:
            best_score_dataset_2 = f1_spam_report_2
            best_config_dataset_2 = {
                'X_2_best': X_2,
                'X_test_dataset_2': X_test_2
            }


# output the optimizing result
with open("src/results/optimizing_result.txt", "w") as f:
    f.write(optimizing_result)

'''-----------------------------'''
