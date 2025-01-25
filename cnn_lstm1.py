import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import optuna
import numpy as np
from data_utils import prepare_train_test_data  # Ensure this imports correctly

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the CNN module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=2),

            nn.Conv1d(256, 128, kernel_size=3, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=2)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (batch_size, channels, sequence_length)
        return self.feature_reduction(x)

# Define the combined CNN-LSTM module
class CNN_LSTM(nn.Module):
    def __init__(self, sequence_length=20, hidden_size=100, num_classes=4):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.cnn_output_size = 128

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # Prepare for LSTM input
        lstm_out, _ = self.lstm(cnn_out)
        x = self.classifier(lstm_out[:, -1, :])  # Only take the output from the last time step
        return x

def train_model(trial, X_train, y_train, X_test, y_test, device, class_weights=None):
    # Hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 50, 200)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = CNN_LSTM(sequence_length=20, hidden_size=hidden_size, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 100
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            all_preds = []
            all_labels = []

            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

            if f1 > best_f1:
                best_f1 = f1

    return best_f1

def objective(trial):
    # Load and prepare data
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()

    # Convert data to numpy arrays first, then to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    # Train model
    return train_model(trial, X_train, y_train, X_test, y_test, device, class_weights)

def main():
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Run 20 trials

    # Display the best trial
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
