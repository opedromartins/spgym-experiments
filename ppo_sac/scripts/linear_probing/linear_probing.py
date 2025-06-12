import glob
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import wandb

# Set device to CUDA if available
device = "cpu"
dataset_dir = "encoding_datasets"

# Define a simple linear probe model.
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        # Output dimension is num_cells * num_classes
        self.linear = nn.Linear(input_dim, num_classes ** 2)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.linear(x)  # Shape: [batch_size, num_cells * num_classes]
        # Reshape to [batch_size, num_cells, num_classes]
        out = out.view(-1, self.num_classes, self.num_classes)
        return out
    
config = {
    "max_dataset_size": 43500,
    "train_split": 0.8,
    "val_split": 0.1,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 1000,
    "patience": 10
}

with open("linear_probing_results.json", "r") as f:
    probe_results = json.load(f)

for filename in tqdm(glob.glob(f"{dataset_dir}/*.jsonl"), desc="Processing files"):
    if filename in probe_results:
        continue

    # Initialize wandb run
    run_name = os.path.basename(filename).split('.')[0]
    wandb.init(project="spgym_linear_probing", name=run_name, config=config)

    # latent_vectors: [num_samples, latent_dim]
    # target_states: [num_samples, 3, 3] where each element is an integer in {0,1,...,8}
    latent_vectors = []
    target_states = []
    with open(filename, "r") as f:
        for line in tqdm(f, total=config["max_dataset_size"], desc="Loading data", leave=False):
            data = json.loads(line)
            latent_vectors.append(data["latent"])
            target_states.append(data["state"])
            if len(latent_vectors) >= config["max_dataset_size"]:
                break
    latent_vectors = torch.tensor(latent_vectors).to(device)
    target_states = torch.tensor(target_states).to(device)
    target_states = target_states.view(target_states.size(0), -1).clamp(min=0)  # flatten to [num_samples, 9]
    num_classes = target_states.size(1)
    print(latent_vectors.shape, target_states.shape)

    # Create a dataset and split it into training (60%), validation (20%), and testing (20%) sets
    dataset = TensorDataset(latent_vectors, target_states)
    train_size = int(config["train_split"] * len(dataset))
    val_size = int(config["val_split"] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    input_dim = latent_vectors.size(1)
    model = LinearProbe(input_dim, num_classes).to(device)

    # Use CrossEntropyLoss for the classification task.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    pbar = tqdm(range(config["num_epochs"]), desc="Epochs")
    for epoch in pbar:
        model.train()
        running_loss = 0.0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data, batch_target
            optimizer.zero_grad()
            outputs = model(batch_data)  # Shape: [batch_size, 9, 9]
            # Flatten outputs and targets for loss computation.
            # outputs: [batch_size * 9, 9]
            outputs = outputs.view(-1, num_classes)
            # batch_target: [batch_size, 9] -> [batch_size * 9]
            batch_target = batch_target.view(-1)
            
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)
        
        epoch_loss = running_loss / train_size
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                batch_data, batch_target = batch_data, batch_target
                outputs = model(batch_data)
                outputs = outputs.view(-1, num_classes)
                batch_target = batch_target.view(-1)
                val_loss += criterion(outputs, batch_target).item() * batch_data.size(0)
                
                # Calculate accuracy for validation set
                predicted = outputs.argmax(dim=1)
                correct += (predicted == batch_target).sum().item()
                total += batch_target.size(0)
        
        val_loss /= val_size
        val_acc = correct / total * 100
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": best_val_loss
        })
        
        pbar.set_postfix_str(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n\tEarly stopping at epoch {epoch+1}")
                break

    # Evaluate the probe on the test set.
    model.eval()
    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), f"linear_probing_models/{run_name}.pth")
        wandb.save(f"linear_probing_models/{run_name}.pth")

    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
            batch_data, batch_target = batch_data, batch_target
            outputs = model(batch_data)
            outputs = outputs.view(-1, num_classes)
            batch_target = batch_target.view(-1)
            loss = criterion(outputs, batch_target)
            test_loss += loss.item() * batch_data.size(0)
            
            # Calculate accuracy for each cell
            predicted = outputs.argmax(dim=1)
            correct += (predicted == batch_target).sum().item()
            total += batch_target.size(0)

    test_loss /= test_size
    accuracy = correct / total * 100
    
    # Log final metrics
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_accuracy": accuracy,
        "final_val_loss": val_loss,
        "final_val_accuracy": val_acc,
        "final_train_loss": epoch_loss
    })
    
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    probe_results[run_name] = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "train_loss": epoch_loss
    }
    with open(f"linear_probing_results.json", "w") as f:
        json.dump(probe_results, f)
    
    # Finish wandb run
    wandb.finish()
