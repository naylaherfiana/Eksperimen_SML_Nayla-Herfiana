import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Breast cancer dataset dari sklearn sudah bersih, tapi kita pastikan tidak ada duplikat
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def preprocess_data(input_path, output_dir):
    df = load_data(input_path)
    df = clean_data(df)

    # Memisahkan fitur dan target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratify split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    os.makedirs(output_dir, exist_ok=True)

    # Menyimpan data train
    train_final = X_train_scaled.copy()
    train_final["target"] = y_train.values
    train_final.to_csv(os.path.join(output_dir, "breast_cancer_preprocessed.csv"), index=False)

    # Menyimpan data test
    test_final = X_test_scaled.copy()
    test_final["target"] = y_test.values
    test_final.to_csv(os.path.join(output_dir, "breast_cancer_test.csv"), index=False)

    return train_final, test_final

if __name__ == "__main__":
    preprocess_data(
        input_path="breast_cancer_raw/breast_cancer.csv",
        output_dir="preprocessing/breast_cancer_preprocessing"
    )
    print("Preprocessing selesai.")

