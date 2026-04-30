import copy
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
from data_loader import get_dataloader
from sklearn.metrics import classification_report, average_precision_score, precision_recall_fscore_support

# Hyperparameters
INPUT_DIM = 26
LR = 0.01
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 256

# DP settings
CLIP_NORM = 1.0
BASE_NOISE_MULTIPLIER = 0.5
EMA_ALPHA = 0.7
MIN_NOISE_FACTOR = 0.9
MAX_NOISE_FACTOR = 1.1

# FLSS-Dyn settings
CLUSTER_PATH = "results/clustering/hybrid_feature_clusters.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cluster_noise_scales():
    """
    Convert cluster sensitivity scores into noise multipliers.

    Higher cluster sensitivity score = more useful/risky cluster.
    For this first FLSS-Dyn version:
      - high score -> lower noise
      - low score -> higher noise
    """
    cluster_df = pd.read_csv(CLUSTER_PATH)

    # Sort by feature order as used in data_loader.py numeric columns.
    feature_to_cluster = dict(zip(cluster_df["Feature"], cluster_df["Cluster_Name"]))

    cluster_scores = (
        cluster_df[["Cluster_Name", "Cluster_Sensitivity_Score"]]
        .drop_duplicates()
        .set_index("Cluster_Name")["Cluster_Sensitivity_Score"]
        .to_dict()
    )

    max_score = max(cluster_scores.values())
    min_score = min(cluster_scores.values())

    cluster_noise = {}

    for cluster_name, score in cluster_scores.items():
        # Normalize score to [0,1]
        norm_score = (score - min_score) / (max_score - min_score + 1e-8)

        # High sensitivity/utility gets less noise.
        # Range: 0.5x to 1.5x BASE_NOISE_MULTIPLIER
        noise_multiplier = BASE_NOISE_MULTIPLIER * (1.2 - 0.4 * norm_score)

        cluster_noise[cluster_name] = noise_multiplier

    print("\nFLSS-Dyn cluster noise multipliers:")
    for k, v in cluster_noise.items():
        print(f"{k}: {v:.4f}")

    return feature_to_cluster, cluster_noise, cluster_df

def initialize_cluster_ema(cluster_noise):
    """
    Initialize EMA gradient norms for each cluster.
    Start equally to avoid unstable first-round allocation.
    """
    return {cluster_name: 1.0 for cluster_name in cluster_noise.keys()}


def compute_cluster_gradient_norms(model, feature_names, feature_to_cluster):
    """
    Compute gradient norm for each feature cluster using fc1.weight gradients.

    fc1.weight shape:
        [hidden_dim, input_dim]

    Each input feature corresponds to one column.
    """
    cluster_norms = {}

    if model.fc1.weight.grad is None:
        return cluster_norms

    grad = model.fc1.weight.grad.detach()

    for feature_idx, feature_name in enumerate(feature_names):
        cluster_name = feature_to_cluster.get(feature_name, None)

        if cluster_name is None:
            continue

        feature_grad_norm = grad[:, feature_idx].norm(2).item()

        if cluster_name not in cluster_norms:
            cluster_norms[cluster_name] = []

        cluster_norms[cluster_name].append(feature_grad_norm)

    # Average feature norms inside each cluster
    cluster_norms = {
        cluster_name: float(sum(values) / len(values))
        for cluster_name, values in cluster_norms.items()
        if len(values) > 0
    }

    return cluster_norms


def update_cluster_ema(cluster_ema, observed_norms):
    """
    EMA update:
        EMA_t = alpha * EMA_{t-1} + (1-alpha) * observed_norm
    """
    for cluster_name in cluster_ema.keys():
        observed = observed_norms.get(cluster_name, 0.0)
        cluster_ema[cluster_name] = (
            EMA_ALPHA * cluster_ema[cluster_name]
            + (1.0 - EMA_ALPHA) * observed
        )

    return cluster_ema


def compute_dynamic_cluster_noise(base_cluster_noise, cluster_ema):
    """
    Convert EMA gradient norms into dynamic noise.

    Higher gradient norm means the model is still learning from that cluster,
    so we reduce noise for that cluster.

    Lower gradient norm means the cluster is less active/saturated,
    so we increase noise.
    """
    max_ema = max(cluster_ema.values())
    min_ema = min(cluster_ema.values())

    dynamic_noise = {}

    for cluster_name, base_noise in base_cluster_noise.items():
        norm_score = (
            (cluster_ema[cluster_name] - min_ema)
            / (max_ema - min_ema + 1e-8)
        )

        # High EMA -> lower noise factor
        # Low EMA -> higher noise factor
        noise_factor = MAX_NOISE_FACTOR - (MAX_NOISE_FACTOR - MIN_NOISE_FACTOR) * norm_score

        dynamic_noise[cluster_name] = base_noise * noise_factor

    return dynamic_noise


def average_cluster_norms(norm_list):
    """
    Average cluster gradient norms across clients.
    """
    if not norm_list:
        return {}

    all_clusters = set()
    for norms in norm_list:
        all_clusters.update(norms.keys())

    avg_norms = {}

    for cluster_name in all_clusters:
        vals = [norms[cluster_name] for norms in norm_list if cluster_name in norms]
        avg_norms[cluster_name] = float(sum(vals) / len(vals)) if vals else 0.0

    return avg_norms


def add_uniform_dp_noise_to_non_input_layers(model):
    """
    Apply uniform DP noise to layers after the first layer.
    FLSS-Dyn is applied only to input-feature-connected fc1.weight columns.
    """
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        if name == "fc1.weight":
            continue

        noise = torch.normal(
            mean=0.0,
            std=BASE_NOISE_MULTIPLIER * CLIP_NORM,
            size=p.grad.shape,
            device=p.grad.device
        )
        p.grad.add_(noise)


def clip_gradients(model, clip_norm=CLIP_NORM):
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


def add_flss_dyn_noise_to_fc1(model, feature_names, feature_to_cluster, cluster_noise):
    """
    Apply cluster-aware Gaussian noise to fc1.weight.

    fc1.weight has shape:
        [hidden_dim, input_dim]

    Each input feature corresponds to one column.
    """
    if model.fc1.weight.grad is None:
        return

    grad = model.fc1.weight.grad

    for feature_idx, feature_name in enumerate(feature_names):
        cluster_name = feature_to_cluster.get(feature_name, None)

        if cluster_name is None:
            noise_multiplier = BASE_NOISE_MULTIPLIER
        else:
            noise_multiplier = cluster_noise[cluster_name]

        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * CLIP_NORM,
            size=grad[:, feature_idx].shape,
            device=grad.device
        )

        grad[:, feature_idx].add_(noise)


def train_client_flss_dyn(model, dataloader, feature_names, feature_to_cluster, cluster_noise):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    pos_weight = torch.tensor([90.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    collected_cluster_norms = []

    for _ in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()

            # Collect cluster gradient norms BEFORE clipping/noise
            batch_cluster_norms = compute_cluster_gradient_norms(
                model,
                feature_names,
                feature_to_cluster
            )
            collected_cluster_norms.append(batch_cluster_norms)

            # Step 1: clip gradients
            clip_gradients(model)

            # Step 2: add FLSS-Dyn cluster-aware noise to input layer
            add_flss_dyn_noise_to_fc1(
                model,
                feature_names,
                feature_to_cluster,
                cluster_noise
            )

            # Step 3: add uniform DP noise to remaining layers
            add_uniform_dp_noise_to_non_input_layers(model)

            optimizer.step()

    client_avg_norms = average_cluster_norms(collected_cluster_norms)

    return (
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        client_avg_norms
    )

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


def main():
    print(f"Starting FLSS-Dyn DP-FL on {device}...")
    print(f"CLIP_NORM={CLIP_NORM}, BASE_NOISE_MULTIPLIER={BASE_NOISE_MULTIPLIER}")

    feature_to_cluster, base_cluster_noise, cluster_df = load_cluster_noise_scales()
    cluster_ema = initialize_cluster_ema(base_cluster_noise)

    # Feature order must match data_loader.py numeric column order.
    sample_loader = get_dataloader("a", batch_size=BATCH_SIZE)
    feature_names = list(sample_loader.dataset.feature_names)

    print("\nFeature order:")
    for i, f in enumerate(feature_names):
        print(f"{i}: {f}")

    global_model = get_model(INPUT_DIM).to(device)

    train_banks = ["a", "b"]
    test_bank = "c"

    test_loader = get_dataloader(test_bank, batch_size=BATCH_SIZE)

    for r in range(ROUNDS):
        print(f"\n--- Round {r + 1}/{ROUNDS} ---")

        dynamic_cluster_noise = compute_dynamic_cluster_noise(base_cluster_noise, cluster_ema)

        print("\nDynamic cluster noise for this round:")
        for k, v in dynamic_cluster_noise.items():
            print(f"{k}: {v:.4f} | EMA={cluster_ema[k]:.6f}")

        client_updates = []
        client_norms_list = []

        for bank in train_banks:
            print(f"Training on Bank {bank.upper()} with FLSS-Dyn gradients...")

            train_loader = get_dataloader(bank, batch_size=BATCH_SIZE)

            local_model = get_model(INPUT_DIM).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            local_weights, client_norms = train_client_flss_dyn(
                local_model,
                train_loader,
                feature_names,
                feature_to_cluster,
                dynamic_cluster_noise
            )

        client_updates.append(local_weights)
        client_norms_list.append(client_norms)

        aggregate_weights(global_model, client_updates)
        round_avg_norms = average_cluster_norms(client_norms_list)
        cluster_ema = update_cluster_ema(cluster_ema, round_avg_norms)

        print("\nUpdated cluster EMA gradient norms:")
        for k, v in cluster_ema.items():
            print(f"{k}: {v:.6f}")

        print(f"Global model updated for round {r + 1}")

        print("Evaluating Global Model on Bank C...")
        report, auc_pr, precision, recall, f1, best_threshold = evaluate(global_model, test_loader)

        print(report)
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"Best Threshold: {best_threshold:.2f}")
        print(f"Fraud Precision: {precision:.4f}")
        print(f"Fraud Recall: {recall:.4f}")
        print(f"Fraud F1: {f1:.4f}")

    torch.save(global_model.state_dict(), "global_baf_flss_dyn_model.pth")
    print("\nFLSS-Dyn DP-FL training complete. Model saved.")


if __name__ == "__main__":
    main()