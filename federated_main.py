import torch
import torch.optim as optim
import torch.nn as nn
from models import get_model
from data_loader import get_dataloader
from data_loader import get_noisedataloader
from sklearn.metrics import f1_score, classification_report

# Hyperparameters
INPUT_DIM = 26 # Adjust this if your BAF columns differ
LR = 0.01
ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_client(model, dataloader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    #criterion = nn.BCELoss()
    # Use a weight (Roughly 1:90 ratio for BAF)
    weight = torch.tensor([90.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    
    for epoch in range(LOCAL_EPOCHS):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def aggregate_weights(global_model, client_weights):
    # Standard FedAvg: Average the weights
    avg_weights = client_weights[0]
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
    global_model.load_state_dict(avg_weights)

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            # Apply sigmoid here since it's no longer in the model
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # zero_division=0 prevents errors if the model doesn't predict any fraud yet
    report = classification_report(all_targets, all_preds, target_names=['No Fraud', 'Fraud'], zero_division=0)
    return report

def main():
    print(f"Starting Federated Learning on {device}...")
    global_model = get_model(INPUT_DIM).to(device)
    
    banks = ['a', 'b', 'c']
    
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")
        client_updates = []
        
        for bank in banks:
            print(f"Training on Bank {bank.upper()}...")
            #loader = get_dataloader(bank, batch_size=BATCH_SIZE)
            loader = get_noisedataloader(bank, batch_size=BATCH_SIZE)
            
            # Create a local copy of the global model
            local_model = get_model(INPUT_DIM).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            weights = train_client(local_model, loader)
            client_updates.append(weights)
        
        # 1. Server aggregates the updates
        aggregate_weights(global_model, client_updates)
        print(f"Global model updated for round {r+1}")

        # 2. ADD THIS: Evaluate the new global model
        print(f"Evaluating Global Model on Bank C (Test Set)...")
        #test_loader = get_dataloader('c', batch_size=BATCH_SIZE)
        test_loader = get_noisedataloader('c', batch_size=BATCH_SIZE)
        metrics_report = evaluate(global_model, test_loader)
        print(metrics_report)

    torch.save(global_model.state_dict(), "global_baf_model.pth")
    print("\nBaseline Training Complete. Model saved.")

if __name__ == "__main__":
    main()