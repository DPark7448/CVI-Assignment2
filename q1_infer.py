# Q1/q1_infer.py

import sys
import os
import glob
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from joblib import load

MODEL_PATH = "models/catdog_best_model.joblib"


def preprocess_image(path, img_size):
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img_resized = resize(img, img_size, anti_aliasing=True)
    feat = hog(
        img_resized,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
    )
    return feat


def collect_image_paths(args):
    path_set = set()

    for arg in args:
        if os.path.isdir(arg):
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for p in glob.glob(os.path.join(arg, pattern)):
                    path_set.add(os.path.normpath(p))
        else:
            path_set.add(os.path.normpath(arg))

    return sorted(path_set)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python q1_infer.py image1.jpg [image2.jpg ...]")
        print("  python q1_infer.py folder_with_images")
        sys.exit(1)

    bundle = load(MODEL_PATH)
    model = bundle["model"]
    classes = bundle["classes"]
    img_size = bundle["img_size"]

    image_paths = collect_image_paths(sys.argv[1:])

    if not image_paths:
        print("No image files found for the given paths.")
        sys.exit(1)

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"{img_path}: file not found")
            continue

        feat = preprocess_image(img_path, img_size).reshape(1, -1)
        pred_idx = model.predict(feat)[0]
        label = classes[pred_idx]
        print(f"{img_path} -> predicted: {label}")


if __name__ == "__main__":
    main()
