import copy
import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
from data_loader import get_dataloader, get_train_val_dataloaders
from sklearn.metrics import classification_report, average_precision_score, precision_recall_fscore_support
import csv
import os

# Hyperparameters
INPUT_DIM = 26
LR = 0.01
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 256

# Simple DP settings for first experiment
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.5  # try 0.1, 0.3, 0.5, 1.0 later

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_dp_noise_to_gradients(model, clip_norm=CLIP_NORM, noise_multiplier=NOISE_MULTIPLIER):
    """
    Simple baseline DP-style gradient clipping and Gaussian noise.
    This is not yet FLSS-Dyn cluster-aware DP.
    We first create a uniform DP-gradient baseline.
    """
    total_norm = torch.norm(
        torch.stack([
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]),
        2
    )

    clip_coef = clip_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef.item(), 1.0)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(clip_coef)

            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * clip_norm,
                size=p.grad.shape,
                device=p.grad.device
            )

            p.grad.add_(noise)


def train_client_dp(model, dataloader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pos_weight = torch.tensor([90.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()

            add_dp_noise_to_gradients(model)

            optimizer.step()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def aggregate_weights(global_model, client_weights):
    avg_weights = {}

    for key in client_weights[0].keys():
        avg_weights[key] = torch.stack(
            [client_weight[key].float() for client_weight in client_weights],
            dim=0
        ).mean(dim=0)

    global_model.load_state_dict(avg_weights)


def evaluate(model, dataloader):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            logits = model(data)
            probs = torch.sigmoid(logits)

            all_targets.extend(target.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    auc_pr = average_precision_score(all_targets, all_probs)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = []

    print("\nThreshold analysis:")
    for th in thresholds:
        preds = [1 if p > th else 0 for p in all_probs]

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets,
            preds,
            average="binary",
            zero_division=0
        )

        threshold_results.append((th, precision, recall, f1))

        print(
            f"Threshold={th:.1f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

    best = max(threshold_results, key=lambda x: x[3])
    best_threshold, best_precision, best_recall, best_f1 = best

    best_preds = [1 if p > best_threshold else 0 for p in all_probs]

    report = classification_report(
        all_targets,
        best_preds,
        target_names=["No Fraud", "Fraud"],
        zero_division=0
    )

    return report, auc_pr, best_precision, best_recall, best_f1, best_threshold

def save_round_metrics(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def main():
    print(f"Starting Uniform DP-FL with validation-based early stopping on {device}...")
    print(f"CLIP_NORM={CLIP_NORM}, NOISE_MULTIPLIER={NOISE_MULTIPLIER}")

    log_path = "results/experiments/uniform_dp_earlystop_round_metrics.csv"

    if os.path.exists(log_path):
        os.remove(log_path)

    global_model = get_model(INPUT_DIM).to(device)

    # Same split as FLSS-Dyn:
    # Bank A: training
    # Bank B: train + validation split
    # Bank C: final test only
    bank_a_loader = get_dataloader("a", batch_size=BATCH_SIZE)
    bank_b_train_loader, val_loader = get_train_val_dataloaders(
        "b",
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
        seed=42
    )
    test_loader = get_dataloader("c", batch_size=BATCH_SIZE)

    train_loaders = [
        ("a", bank_a_loader),
        ("b_train", bank_b_train_loader),
    ]

    best_val_auc_pr = -1.0
    best_state_dict = None
    best_round = 0

    for r in range(ROUNDS):
        print(f"\n--- Round {r + 1}/{ROUNDS} ---")

        client_updates = []

        for bank_name, train_loader in train_loaders:
            print(f"Training on Bank {bank_name.upper()} with uniform DP gradients...")

            local_model = get_model(INPUT_DIM).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            local_weights = train_client_dp(local_model, train_loader)
            client_updates.append(local_weights)

        aggregate_weights(global_model, client_updates)
        print(f"Global model updated for round {r + 1}")

        print("Evaluating Global Model on Bank B validation set...")
        val_report, val_auc_pr, val_precision, val_recall, val_f1, val_threshold= evaluate(
            global_model,
            val_loader
        )

        print(val_report)
        print(f"Validation AUC-PR: {val_auc_pr:.4f}")
        print(f"Validation Fraud Precision: {val_precision:.4f}")
        print(f"Validation Fraud Recall: {val_recall:.4f}")
        print(f"Validation Fraud F1: {val_f1:.4f}")

        save_round_metrics(log_path, {
            "method": "uniform_dp_earlystop",
            "phase": "validation",
            "round": r + 1,
            "auc_pr": val_auc_pr,
            "fraud_precision": val_precision,
            "fraud_recall": val_recall,
            "fraud_f1": val_f1,
            "best_threshold": val_threshold,
            "clip_norm": CLIP_NORM,
            "noise_multiplier": NOISE_MULTIPLIER,
            "best_val_auc_pr_so_far": max(best_val_auc_pr, val_auc_pr),
        })

        if val_auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_auc_pr
            best_round = r + 1
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }
            print(f"New best validation model found at round {best_round}")

    print(f"\nLoading best validation model from round {best_round}")
    global_model.load_state_dict(best_state_dict)

    print("Final evaluation on Bank C test set...")
    test_report, test_auc_pr, test_precision, test_recall, test_f1, test_threshold = evaluate(
        global_model,
        test_loader
    )

    print(test_report)
    print(f"Test AUC-PR: {test_auc_pr:.4f}")
    print(f"Test Fraud Precision: {test_precision:.4f}")
    print(f"Test Fraud Recall: {test_recall:.4f}")
    print(f"Test Fraud F1: {test_f1:.4f}")

    save_round_metrics(log_path, {
        "method": "uniform_dp_earlystop",
        "phase": "test",
        "round": best_round,
        "auc_pr": test_auc_pr,
        "fraud_precision": test_precision,
        "fraud_recall": test_recall,
        "fraud_f1": test_f1,
        "best_threshold": test_threshold,
        "clip_norm": CLIP_NORM,
        "noise_multiplier": NOISE_MULTIPLIER,
        "best_val_auc_pr_so_far": best_val_auc_pr,
    })

    torch.save(global_model.state_dict(), "global_baf_uniform_dp_earlystop_model.pth")
    print("\nUniform DP-FL with validation-based early stopping complete. Model saved.")


if __name__ == "__main__":
    main()