import os
import pandas as pd

RESULT_DIR = "results/experiments"

INPUT_FILES = {
    "Clean FL + Early Stopping": "clean_fl_earlystop_round_metrics.csv",
    "Uniform DP + Early Stopping": "uniform_dp_earlystop_round_metrics.csv",
    "FLSS-Dyn + Early Stopping": "flss_dyn_earlystop_round_metrics.csv",
}

SUMMARY_OUT = os.path.join(RESULT_DIR, "summary_comparison_table.csv")
LATEX_OUT = os.path.join(RESULT_DIR, "summary_comparison_table.tex")


def load_test_row(method_name, filename):
    path = os.path.join(RESULT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if "phase" not in df.columns:
        raise ValueError(f"{filename} does not contain a 'phase' column.")

    test_rows = df[df["phase"] == "test"]

    if test_rows.empty:
        raise ValueError(f"{filename} does not contain a test row.")

    # Usually there is only one test row.
    row = test_rows.iloc[-1]

    return {
        "Method": method_name,
        "Selected Round": int(row["round"]),
        "AUC-PR": float(row["auc_pr"]),
        "Precision": float(row["fraud_precision"]),
        "Recall": float(row["fraud_recall"]),
        "F1": float(row["fraud_f1"]),
        "Best Threshold": float(row["best_threshold"]) if "best_threshold" in row else None,
    }


def main():
    rows = []

    for method_name, filename in INPUT_FILES.items():
        rows.append(load_test_row(method_name, filename))

    summary_df = pd.DataFrame(rows)

    # Sort by F1 or AUC-PR if desired.
    # For now, keep logical method order.

    os.makedirs(RESULT_DIR, exist_ok=True)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\nSummary comparison table:")
    print(summary_df.to_string(index=False))

    # Create a LaTeX table for paper use.
    latex_df = summary_df.copy()

    for col in ["AUC-PR", "Precision", "Recall", "F1", "Best Threshold"]:
        if col in latex_df.columns:
            latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

    latex_table = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrr",
        caption="Performance comparison of clean FL, uniform DP-FL, and FLSS-Dyn under validation-based checkpoint selection.",
        label="tab:main_comparison"
    )

    with open(LATEX_OUT, "w") as f:
        f.write(latex_table)

    print(f"\nSaved CSV summary to: {SUMMARY_OUT}")
    print(f"Saved LaTeX table to: {LATEX_OUT}")


if __name__ == "__main__":
    main()