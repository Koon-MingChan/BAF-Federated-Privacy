import copy
import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
from data_loader import get_dataloader
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
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            logits = model(data)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(
        all_targets,
        all_preds,
        target_names=["No Fraud", "Fraud"],
        zero_division=0
    )

    auc_pr = average_precision_score(all_targets, all_probs)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="binary",
        zero_division=0
    )

    return report, auc_pr, precision, recall, f1

def save_round_metrics(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def main():
    print(f"Starting Uniform DP-FL baseline on {device}...")
    print(f"CLIP_NORM={CLIP_NORM}, NOISE_MULTIPLIER={NOISE_MULTIPLIER}")

    log_path = "results/experiments/uniform_dp_round_metrics.csv"

    if os.path.exists(log_path):
        os.remove(log_path)

    global_model = get_model(INPUT_DIM).to(device)

    train_banks = ["a", "b"]
    test_bank = "c"

    test_loader = get_dataloader(test_bank, batch_size=BATCH_SIZE)

    for r in range(ROUNDS):
        print(f"\n--- Round {r + 1}/{ROUNDS} ---")

        client_updates = []

        for bank in train_banks:
            print(f"Training on Bank {bank.upper()} with DP gradients...")

            train_loader = get_dataloader(bank, batch_size=BATCH_SIZE)

            local_model = get_model(INPUT_DIM).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            local_weights = train_client_dp(local_model, train_loader)
            client_updates.append(local_weights)

        aggregate_weights(global_model, client_updates)
        print(f"Global model updated for round {r + 1}")

        print("Evaluating Global Model on Bank C...")
        report, auc_pr, precision, recall, f1 = evaluate(global_model, test_loader)

        print(report)
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"Fraud Precision: {precision:.4f}")
        print(f"Fraud Recall: {recall:.4f}")
        print(f"Fraud F1: {f1:.4f}")

        save_round_metrics(log_path, {
            "method": "uniform_dp",
            "phase": "test",
            "round": r + 1,
            "auc_pr": auc_pr,
            "fraud_precision": precision,
            "fraud_recall": recall,
            "fraud_f1": f1,
            "clip_norm": CLIP_NORM,
            "noise_multiplier": NOISE_MULTIPLIER,
        })

    torch.save(global_model.state_dict(), "global_baf_dp_model.pth")
    print("\nUniform DP-FL baseline training complete. Model saved.")


if __name__ == "__main__":
    main()