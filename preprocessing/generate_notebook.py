import nbformat as nbf

nb = nbf.v4.new_notebook()

code_cells = [
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler""",

    """df = pd.read_csv("../credit_scoring_raw/credit_scoring_raw.csv")
df.head()""",

    """df.info()
df.describe()
df.isnull().sum()          # cek missing value
df["SeriousDlqin2yrs"].value_counts(normalize=True)   # cek distribusi target""",

    """plt.figure(figsize=(6,4))
sns.countplot(x="SeriousDlqin2yrs", data=df)
plt.title("Distribusi Target")
plt.show()""",

    """plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Korelasi Antar Fitur")
plt.show()""",

    """# Hapus kolom index bawaan
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Tangani missing value
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0])

# Tangani outlier ekstrem sederhana (capping usia)
df = df[df["age"] > 0]
df["age"] = df["age"].clip(upper=df["age"].quantile(0.99))

# Pisahkan fitur & target
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)""",

    """import os
os.makedirs("credit_scoring_preprocessing", exist_ok=True)

train_final = X_train_scaled.copy()
train_final["SeriousDlqin2yrs"] = y_train.values
train_final.to_csv("credit_scoring_preprocessing/credit_scoring_preprocessed.csv", index=False)

test_final = X_test_scaled.copy()
test_final["SeriousDlqin2yrs"] = y_test.values
test_final.to_csv("credit_scoring_preprocessing/credit_scoring_test.csv", index=False)"""
]

nb['cells'] = [nbf.v4.new_code_cell(code) for code in code_cells]

with open('preprocessing/Eksperimen_Nayla.ipynb', 'w') as f:
    nbf.write(nb, f)
