import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0])
    df = df[df["age"] > 0]
    df["age"] = df["age"].clip(upper=df["age"].quantile(0.99))
    return df

def preprocess_data(input_path, output_dir):
    df = load_data(input_path)
    df = clean_data(df)

    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    os.makedirs(output_dir, exist_ok=True)

    train_final = X_train_scaled.copy()
    train_final["SeriousDlqin2yrs"] = y_train.values
    train_final.to_csv(os.path.join(output_dir, "credit_scoring_preprocessed.csv"), index=False)

    test_final = X_test_scaled.copy()
    test_final["SeriousDlqin2yrs"] = y_test.values
    test_final.to_csv(os.path.join(output_dir, "credit_scoring_test.csv"), index=False)

    return train_final, test_final

if __name__ == "__main__":
    preprocess_data(
        input_path="credit_scoring_raw/credit_scoring_raw.csv",
        output_dir="preprocessing/credit_scoring_preprocessing"
    )
    print("Preprocessing selesai.")
