import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

# กำหนด path ของไฟล์ข้อมูล
DATA_FILE_PATH = r"C:\\CNNLSTM\\scripts\\data_normal_abnormal.csv"

def load_thermal_data(file_path, start_row, end_row):
    data = pd.read_csv(file_path)
    features = data.iloc[start_row - 1:end_row, 1:-2].values
    labels = data.iloc[start_row - 1:end_row, -1].values
    return features, labels

def prepare_data(features, labels, window_size=20, stride=1):
    """
    Prepare data with sliding window approach and consistent sampling
    """
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, stride):
        window_features = features[i:i + window_size]
        window_label = labels[i + window_size - 1]
        X.append(window_features)
        y.append(window_label)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def prepare_train_test_data(file_path=DATA_FILE_PATH):
    """
    Load and prepare data for training and testing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    data = pd.read_csv(file_path)
    total_rows = data.shape[0]

    # Process data
    features, labels = load_thermal_data(file_path, 2, total_rows)
    X, y = prepare_data(features, labels, window_size=20, stride=1)

    # Separate data by class
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    y_class0 = y[y == 0]
    y_class1 = y[y == 1]

    # Split each class separately
    X_train0, X_test0, y_train0, y_test0 = train_test_split(
        X_class0, y_class0,
        test_size=0.2,
        random_state=42
    )

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_class1, y_class1,
        test_size=0.2,
        random_state=42
    )

    # Concatenate the splits
    X_train = torch.cat([X_train0, X_train1])
    X_test = torch.cat([X_test0, X_test1])
    y_train = torch.cat([y_train0, y_train1])
    y_test = torch.cat([y_test0, y_test1])

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y.numpy()),
        y=y.numpy()
    )

    # Print data distribution
    print("\nClass distribution in training set:")
    unique_labels_train, counts_train = np.unique(y_train.numpy(), return_counts=True)
    for label, count in zip(unique_labels_train, counts_train):
        print(f"Class {int(label)}: {count} samples ({count / len(y_train) * 100:.2f}%)")

    print("\nClass distribution in test set:")
    unique_labels_test, counts_test = np.unique(y_test.numpy(), return_counts=True)
    for label, count in zip(unique_labels_test, counts_test):
        print(f"Class {int(label)}: {count} samples ({count / len(y_test) * 100:.2f}%)")

    print("\nX_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nFeatures dimension:", X_train.shape[-1])

    print("\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    return X_train, X_test, y_train, y_test, class_weights
