import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import optuna
import numpy as np
from data_utils import prepare_train_test_data
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, activation='ReLU'):
        super(CNN, self).__init__()
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=3, padding=2),
            self.get_activation(activation),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),  # แก้ไขจาก padding=2 เป็น padding=0

            nn.Conv1d(256, 128, kernel_size=3, padding=2),
            nn.BatchNorm1d(128),
            self.get_activation(activation),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)  # แก้ไขจาก padding=2 เป็น padding=0
        )

    def get_activation(self, activation):
        activations = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return activations[activation]

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.feature_reduction(x)


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=100, num_lstm_layers=2, activation='ReLU', num_classes=2):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(activation)
        self.hidden_size = hidden_size
        self.cnn_output_size = 128

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            self.get_activation(activation),
            nn.Linear(32, num_classes)
        )

    def get_activation(self, activation):
        activations = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return activations[activation]

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)
        lstm_out, _ = self.lstm(cnn_out)
        x = self.classifier(lstm_out[:, -1, :])
        return x


def plot_training_progress(losses, f1_scores, num_epochs, trial_number):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses, 'b-', label='Training Loss')
    plt.title(f'Training Loss - Trial {trial_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), f1_scores, 'r-', label='F1 Score')
    plt.title(f'F1 Score Progress - Trial {trial_number}')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_progress_trial_{trial_number}.png')
    plt.close()


def plot_confusion_matrix(true_labels, predictions, trial_number):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Trial {trial_number}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_trial_{trial_number}.png')
    plt.close()


def train_model(trial, X_train, y_train, X_test, y_test, device, class_weights=None):
    hidden_size = trial.suggest_int("lstm_hidden_units", 50, 150)
    num_lstm_layers = trial.suggest_int("lstm_hidden_layers", 1, 5)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ['ReLU', 'Sigmoid', 'tanh'])
    num_epochs = trial.suggest_int("epochs", 50, 200)

    model = CNN_LSTM(
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        activation=activation
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_f1 = 0
    epoch_losses = []
    epoch_f1_scores = []
    best_predictions = None
    best_true_labels = None

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
        epoch_losses.append(avg_loss)

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
            epoch_f1_scores.append(f1)

            print(f'Trial {trial.number}, Epoch [{epoch + 1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')

            if f1 > best_f1:
                best_f1 = f1
                best_predictions = all_preds
                best_true_labels = all_labels
                print("\nCurrent Best Parameters:")
                print(f"F1 Score: {best_f1:.4f}")
                for key, value in trial.params.items():
                    print(f"{key}: {value}")

    plot_training_progress(epoch_losses, epoch_f1_scores, num_epochs, trial.number)
    plot_confusion_matrix(best_true_labels, best_predictions, trial.number)

    return best_f1


def objective(trial):
    X_train, X_test, y_train, y_test, class_weights = prepare_train_test_data()

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)

    return train_model(trial, X_train, y_train, X_test, y_test, device, class_weights)


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"Best F1 Score: {trial.value:.4f}")
    print("Best Parameters:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    print("\nAll trials summary:")
    trials_df = study.trials_dataframe()
    print(trials_df.sort_values('value', ascending=False))


if __name__ == "__main__":
    main()
