import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

MI_REPORT_PATH = "results/sensitivity/feature_mi_report.csv"
PAIRWISE_MI_PATH = "results/clustering/pairwise_mi_matrix.csv"
CLUSTER_OUT = "results/clustering/feature_clusters.csv"

N_CLUSTERS = 12
LAMBDA_CORR = 0.5


def normalize_matrix(mi_matrix: pd.DataFrame) -> pd.DataFrame:
    max_val = mi_matrix.to_numpy().max()
    if max_val == 0:
        return mi_matrix.copy()
    return mi_matrix / max_val


def build_distance_matrix(pairwise_mi: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pairwise MI similarity into distance.
    Higher MI = closer features.
    """
    mi_norm = normalize_matrix(pairwise_mi)
    distance = 1.0 - mi_norm

    # Ensure diagonal is exactly zero
    values = distance.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)

    return pd.DataFrame(values, index=pairwise_mi.index, columns=pairwise_mi.columns)


def cluster_features(distance_matrix: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    features = list(distance_matrix.index)

    condensed_dist = squareform(distance_matrix.to_numpy(), checks=False)
    Z = linkage(condensed_dist, method="average")

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return pd.DataFrame({
        "Feature": features,
        "Cluster_ID": labels
    })


def compute_cluster_scores(cluster_df: pd.DataFrame, mi_report: pd.DataFrame, pairwise_mi: pd.DataFrame) -> pd.DataFrame:
    merged = cluster_df.merge(
        mi_report[["Feature", "MI_with_label", "Entropy_Risk"]],
        on="Feature",
        how="left"
    )

    cluster_scores = []

    for cluster_id, group in merged.groupby("Cluster_ID"):
        features = group["Feature"].tolist()

        max_label_mi = group["MI_with_label"].max()
        avg_label_mi = group["MI_with_label"].mean()

        if len(features) > 1:
            sub_matrix = pairwise_mi.loc[features, features].copy()
            values = sub_matrix.to_numpy(copy=True)
            upper_vals = values[np.triu_indices_from(values, k=1)]
            avg_pairwise_mi = float(np.mean(upper_vals)) if len(upper_vals) > 0 else 0.0
        else:
            avg_pairwise_mi = 0.0

        cluster_score = max_label_mi + LAMBDA_CORR * avg_pairwise_mi

        cluster_scores.append({
            "Cluster_ID": cluster_id,
            "Num_Features": len(features),
            "Features": ", ".join(features),
            "Max_MI_with_label": max_label_mi,
            "Avg_MI_with_label": avg_label_mi,
            "Avg_Pairwise_MI": avg_pairwise_mi,
            "Cluster_Sensitivity_Score": cluster_score
        })

    score_df = pd.DataFrame(cluster_scores)
    return merged.merge(
        score_df[["Cluster_ID", "Cluster_Sensitivity_Score"]],
        on="Cluster_ID",
        how="left"
    ), score_df


def main():
    print("Loading FLSS-Dyn sensitivity and pairwise MI reports...")

    mi_report = pd.read_csv(MI_REPORT_PATH)
    pairwise_mi = pd.read_csv(PAIRWISE_MI_PATH, index_col=0)

    # Make sure feature order is consistent
    features = mi_report["Feature"].tolist()
    pairwise_mi = pairwise_mi.loc[features, features]

    print(f"Loaded {len(features)} features.")

    distance_matrix = build_distance_matrix(pairwise_mi)
    cluster_df = cluster_features(distance_matrix, n_clusters=N_CLUSTERS)

    feature_cluster_df, cluster_score_df = compute_cluster_scores(
        cluster_df,
        mi_report,
        pairwise_mi
    )

    os.makedirs(os.path.dirname(CLUSTER_OUT), exist_ok=True)
    feature_cluster_df.to_csv(CLUSTER_OUT, index=False)

    cluster_score_out = "results/clustering/cluster_scores.csv"
    cluster_score_df.to_csv(cluster_score_out, index=False)

    print(f"\nSaved feature clusters to: {CLUSTER_OUT}")
    print(f"Saved cluster scores to: {cluster_score_out}")

    print("\nCluster summary:")
    print(cluster_score_df.sort_values("Cluster_Sensitivity_Score", ascending=False))

    print("\nFeature-to-cluster mapping:")
    print(feature_cluster_df.sort_values(["Cluster_ID", "MI_with_label"], ascending=[True, False]))


if __name__ == "__main__":
    main()