# Q2/q2_infer.py

import sys
import pandas as pd
import numpy as np
from joblib import load

TEST_CSV = "mnist_test.csv"
MODEL_PATH = "models/mnist_best_model.joblib"


def main():
    if len(sys.argv) != 2:
        print("Usage: python q2_infer.py <row_index>")
        sys.exit(1)

    row_idx = int(sys.argv[1])
    df = pd.read_csv(TEST_CSV)
    n_rows = len(df)

    if row_idx < 0 or row_idx >= n_rows:
        print(f"Row index must be between 0 and {n_rows - 1}")
        sys.exit(1)

    row = df.iloc[row_idx]
    true_label = row.iloc[0]
    pixels = row.iloc[1:].values.astype(np.float32) / 255.0

    X = pixels.reshape(1, -1)

    bundle = load(MODEL_PATH)
    scaler = bundle["scaler"]
    model = bundle["model"]

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    print(f"Row index: {row_idx}")
    print(f"True label:      {true_label}")
    print(f"Predicted label: {pred}")


if __name__ == "__main__":
    main()
