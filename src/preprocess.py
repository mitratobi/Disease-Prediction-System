import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(train_path="data/Training.csv", test_path="data/Testing.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop fully-NaN trailing column (artifact of CSV export)
    train_df = train_df.dropna(axis=1, how="all")
    test_df = test_df.dropna(axis=1, how="all")

    # Identify symptom columns (everything except 'prognosis')
    symptom_cols = [c for c in train_df.columns if c != "prognosis"]

    # Keep only symptoms that carry signal: drop columns where variance is 0
    # (i.e. all-zero in training set — never mentioned)
    non_zero = train_df[symptom_cols].sum() > 0
    symptom_cols = list(non_zero[non_zero].index)

    # Trim to the top-95 most-used symptoms (preserves the 95-feature spec)
    symptom_cols = (
        train_df[symptom_cols]
        .sum()
        .sort_values(ascending=False)
        .head(95)
        .index.tolist()
    )

    X_train = train_df[symptom_cols].values
    X_test = test_df[symptom_cols].values

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["prognosis"])
    y_test = le.transform(test_df["prognosis"])

    return X_train, y_train, X_test, y_test, symptom_cols, le
