import os
import pandas as pd
import numpy as np

MI_REPORT_PATH = "results/sensitivity/feature_mi_report.csv"
PAIRWISE_MI_PATH = "results/clustering/pairwise_mi_matrix.csv"

FEATURE_CLUSTER_OUT = "results/clustering/hybrid_feature_clusters.csv"
CLUSTER_SCORE_OUT = "results/clustering/hybrid_cluster_scores.csv"

LAMBDA_CORR = 0.5

HYBRID_GROUPS = {
    "Identity_Contact": [
        "name_email_similarity",
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "date_of_birth_distinct_emails_4w",
        "customer_age",
    ],
    "Device_Session": [
        "keep_alive_session",
        "device_distinct_emails_8w",
        "session_length_in_minutes",
    ],
    "Financial_Credit": [
        "income",
        "proposed_credit_limit",
        "credit_risk_score",
        "has_other_cards",
    ],
    "Address_History": [
        "prev_address_months_count",
        "current_address_months_count",
        "zip_count_4w",
    ],
    "Behaviour_Velocity": [
        "velocity_4w",
        "velocity_6h",
        "velocity_24h",
        "days_since_request",
    ],
    "Application_Metadata": [
        "month",
        "bank_months_count",
        "bank_branch_count_8w",
        "intended_balcon_amount",
        "foreign_request",
        "device_fraud_count",
    ],
}


def compute_avg_pairwise_mi(features, pairwise_mi):
    if len(features) <= 1:
        return 0.0

    sub_matrix = pairwise_mi.loc[features, features].to_numpy(copy=True)
    upper_vals = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]

    if len(upper_vals) == 0:
        return 0.0

    return float(np.mean(upper_vals))


def main():
    print("Loading MI reports...")

    mi_report = pd.read_csv(MI_REPORT_PATH)
    pairwise_mi = pd.read_csv(PAIRWISE_MI_PATH, index_col=0)

    known_features = set(mi_report["Feature"].tolist())

    feature_rows = []
    cluster_rows = []

    used_features = set()

    for cluster_id, (cluster_name, features) in enumerate(HYBRID_GROUPS.items(), start=1):
        valid_features = [f for f in features if f in known_features]
        missing_features = [f for f in features if f not in known_features]

        if missing_features:
            print(f"Warning: missing features in {cluster_name}: {missing_features}")

        if not valid_features:
            continue

        used_features.update(valid_features)

        group_df = mi_report[mi_report["Feature"].isin(valid_features)].copy()

        max_mi = float(group_df["MI_with_label"].max())
        avg_mi = float(group_df["MI_with_label"].mean())
        avg_entropy = float(group_df["Entropy_Risk"].mean())
        avg_pairwise_mi = compute_avg_pairwise_mi(valid_features, pairwise_mi)

        cluster_score = max_mi + LAMBDA_CORR * avg_pairwise_mi

        for _, row in group_df.iterrows():
            feature_rows.append({
                "Feature": row["Feature"],
                "Cluster_ID": cluster_id,
                "Cluster_Name": cluster_name,
                "MI_with_label": row["MI_with_label"],
                "Entropy_Risk": row["Entropy_Risk"],
                "Cluster_Sensitivity_Score": cluster_score,
            })

        cluster_rows.append({
            "Cluster_ID": cluster_id,
            "Cluster_Name": cluster_name,
            "Num_Features": len(valid_features),
            "Features": ", ".join(valid_features),
            "Max_MI_with_label": max_mi,
            "Avg_MI_with_label": avg_mi,
            "Avg_Entropy_Risk": avg_entropy,
            "Avg_Pairwise_MI": avg_pairwise_mi,
            "Cluster_Sensitivity_Score": cluster_score,
        })

    # Catch any feature not assigned to a semantic group.
    unassigned = sorted(list(known_features - used_features))
    if unassigned:
        print(f"\nWarning: unassigned features detected: {unassigned}")
        for f in unassigned:
            row = mi_report[mi_report["Feature"] == f].iloc[0]
            cluster_id = len(cluster_rows) + 1
            cluster_name = "Unassigned"

            feature_rows.append({
                "Feature": f,
                "Cluster_ID": cluster_id,
                "Cluster_Name": cluster_name,
                "MI_with_label": row["MI_with_label"],
                "Entropy_Risk": row["Entropy_Risk"],
                "Cluster_Sensitivity_Score": row["MI_with_label"],
            })

            cluster_rows.append({
                "Cluster_ID": cluster_id,
                "Cluster_Name": cluster_name,
                "Num_Features": 1,
                "Features": f,
                "Max_MI_with_label": row["MI_with_label"],
                "Avg_MI_with_label": row["MI_with_label"],
                "Avg_Entropy_Risk": row["Entropy_Risk"],
                "Avg_Pairwise_MI": 0.0,
                "Cluster_Sensitivity_Score": row["MI_with_label"],
            })

    feature_cluster_df = pd.DataFrame(feature_rows)
    cluster_score_df = pd.DataFrame(cluster_rows)

    feature_cluster_df = feature_cluster_df.sort_values(
        ["Cluster_ID", "MI_with_label"],
        ascending=[True, False]
    )

    cluster_score_df = cluster_score_df.sort_values(
        "Cluster_Sensitivity_Score",
        ascending=False
    )

    os.makedirs("results/clustering", exist_ok=True)

    feature_cluster_df.to_csv(FEATURE_CLUSTER_OUT, index=False)
    cluster_score_df.to_csv(CLUSTER_SCORE_OUT, index=False)

    print(f"\nSaved hybrid feature clusters to: {FEATURE_CLUSTER_OUT}")
    print(f"Saved hybrid cluster scores to: {CLUSTER_SCORE_OUT}")

    print("\nHybrid cluster summary:")
    print(cluster_score_df)

    print("\nFeature-to-cluster mapping:")
    print(feature_cluster_df)


if __name__ == "__main__":
    main()