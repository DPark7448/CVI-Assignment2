# Q1/q1_train.py

import os
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

RANDOM_STATE = 42
DATA_ROOT = "."         
IMG_SIZE = (128, 128)     
MODEL_SAVE_PATH = "models/catdog_best_model.joblib"

def load_images_and_labels(split_dir):
    classes = ["Cat", "Dog"]
    X = []
    y = []

    for label, cls in enumerate(classes):
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder}")
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                img = imread(fpath)
                if img.ndim == 3:
                    img = rgb2gray(img)
                img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
                X.append(img_resized)
                y.append(label)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

    return np.array(X), np.array(y)


def extract_hog_features(images):
    features = []
    for img in images:
        feat = hog(
            img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm="L2-Hys"
        )
        features.append(feat)
    return np.array(features)


def build_models():
    models = {
        "knn_k3": KNeighborsClassifier(n_neighbors=3),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),

        "linear_svm_C1": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=2000),
        "linear_svm_C10": LinearSVC(C=10.0, random_state=RANDOM_STATE, max_iter=2000),

        "rbf_svm_C1": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        "rbf_svm_C10": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE),

        "rf_100": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=RANDOM_STATE
        ),
        "rf_300": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE
        ),
    }
    return models


def main():
    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "test")

    print("Loading training images...")
    X_train_img, y_train = load_images_and_labels(train_dir)
    print(f"Loaded {len(X_train_img)} training images")

    print("Loading test images...")
    X_test_img, y_test = load_images_and_labels(test_dir)
    print(f"Loaded {len(X_test_img)} test images")

    print("Extracting HOG features for training...")
    X_train_feat = extract_hog_features(X_train_img)
    print("Training features shape:", X_train_feat.shape)

    print("Extracting HOG features for test...")
    X_test_feat = extract_hog_features(X_test_img)
    print("Test features shape:", X_test_feat.shape)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_feat,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    models = build_models()
    best_model = None
    best_name = None
    best_val_acc = 0.0

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"{name} validation accuracy: {val_acc:.4f}")
        print("Validation classification report:")
        print(classification_report(y_val, y_val_pred, target_names=["Cat", "Dog"]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_name = name

    print(f"\nBest model on validation: {best_name} (val acc = {best_val_acc:.4f})")
    y_test_pred = best_model.predict(X_test_feat)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy of best model: {test_acc:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, y_test_pred, target_names=["Cat", "Dog"]))

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    dump(
        {
            "model": best_model,
            "classes": ["Cat", "Dog"],
            "img_size": IMG_SIZE,
        },
        MODEL_SAVE_PATH,
    )
    print(f"\nSaved best model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
