import pandas as pd


def create_rfm_features(df):
    """
    Create simplified RFM-style features.
    """

    rfm = pd.DataFrame()

    rfm["CustomerID"] = df["CustomerID"]

    # Approximation only
    rfm["Recency"] = df["Age"]
    rfm["Frequency"] = df["Annual Income (k$)"]
    rfm["Monetary"] = df["Spending Score (1-100)"]

    return rfm