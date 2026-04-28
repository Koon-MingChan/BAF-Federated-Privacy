import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = "data/Base.csv"
SENSITIVITY_OUT = "results/sensitivity/feature_mi_report.csv"
PAIRWISE_OUT = "results/clustering/pairwise_mi_matrix.csv"

SAMPLE_SIZE = 100000
RANDOM_STATE = 42


def load_sample():
    print("Loading Base.csv...")
    df = pd.read_csv(DATA_PATH)

    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    y = df["fraud_bool"].astype(int)
    X = df.drop(columns=["fraud_bool"])

    # For now, use numeric features only.
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

    return X_numeric, y


def compute_entropy(series):
    prob_dist = series.value_counts(normalize=True)
    return entropy(prob_dist)


def compute_feature_mi_report(X, y):
    print("Computing MI with fraud label...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mi_scores = mutual_info_classif(
        X_scaled,
        y,
        discrete_features=False,
        random_state=RANDOM_STATE
    )

    results = []
    for feature, mi in zip(X.columns, mi_scores):
        results.append({
            "Feature": feature,
            "MI_with_label": mi,
            "Entropy_Risk": compute_entropy(X[feature]),
            "Mean": X[feature].mean(),
            "Std": X[feature].std()
        })

    report = pd.DataFrame(results)
    report = report.sort_values(by="MI_with_label", ascending=False)

    os.makedirs(os.path.dirname(SENSITIVITY_OUT), exist_ok=True)
    report.to_csv(SENSITIVITY_OUT, index=False)

    print(f"\nSaved feature MI report to: {SENSITIVITY_OUT}")
    print("\nTop 10 features by MI_with_label:")
    print(report.head(10))

    return report


def compute_pairwise_mi_matrix(X):
    print("\nComputing pairwise MI matrix between features...")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    features = list(X.columns)
    mi_matrix = pd.DataFrame(
        np.zeros((len(features), len(features))),
        index=features,
        columns=features
    )

    for target_feature in features:
        y_target = X_scaled[target_feature].values

        other_mi = mutual_info_regression(
            X_scaled,
            y_target,
            discrete_features=False,
            random_state=RANDOM_STATE
        )

        mi_matrix.loc[target_feature, :] = other_mi

    # Symmetrize because MI should be symmetric, but estimates may differ.
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0

    # Set diagonal to 0 for clustering.
    mi_values = mi_matrix.to_numpy(copy=True)
    np.fill_diagonal(mi_values, 0.0)
    mi_matrix = pd.DataFrame(mi_values, index=features, columns=features)

    os.makedirs(os.path.dirname(PAIRWISE_OUT), exist_ok=True)
    mi_matrix.to_csv(PAIRWISE_OUT)

    print(f"Saved pairwise MI matrix to: {PAIRWISE_OUT}")
    print("\nPairwise MI matrix preview:")
    print(mi_matrix.iloc[:5, :5])

    return mi_matrix


def analyze_baf():
    X, y = load_sample()

    print(f"Using {len(X)} samples and {X.shape[1]} numeric features.")
    print(f"Fraud rate in sample: {y.mean():.4f}")

    compute_feature_mi_report(X, y)
    compute_pairwise_mi_matrix(X)

    print("\nFLSS-Dyn Phase 1 analysis complete.")


if __name__ == "__main__":
    analyze_baf()