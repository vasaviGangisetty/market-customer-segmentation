import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Encode categorical columns and scale features.
    """
    df = df.copy()

    # Encode Gender
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    # Select features for clustering
    features = df[
        ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ]

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return df, scaled_features