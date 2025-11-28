# Q2/q2_train.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

RANDOM_STATE = 42
TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"
MODEL_SAVE_PATH = "models/mnist_best_model.joblib"

def load_mnist(csv_path):
    """
    Assumes first column is 'label', remaining columns are pixels.
    """
    df = pd.read_csv(csv_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32)
    X /= 255.0
    return X, y


def main():
    print("Loading training data...")
    X, y = load_mnist(TRAIN_CSV)
    print("Train shape:", X.shape, y.shape)

    print("Loading test data...")
    X_test, y_test = load_mnist(TEST_CSV)
    print("Test shape:", X_test.shape, y_test.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "log_reg": LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            C=1.0,
            max_iter=100,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "knn_k3": KNeighborsClassifier(n_neighbors=3),
        "mlp_128": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=5,
        ),
    }

    best_model = None
    best_name = None
    best_val_acc = 0.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)

        y_val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"{name} validation accuracy: {val_acc:.4f}")
        print("Validation classification report:")
        print(classification_report(y_val, y_val_pred))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_name = name

    print(f"\nBest model on validation: {best_name} (val acc = {best_val_acc:.4f})")

    y_test_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy of best model: {test_acc:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, y_test_pred))

    if test_acc < 0.90:
        print("WARNING: Test accuracy < 90%. Consider tuning hyperparameters.")

    bundle = {
        "scaler": scaler,
        "model": best_model,
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    dump(bundle, MODEL_SAVE_PATH)
    print(f"\nSaved best model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
