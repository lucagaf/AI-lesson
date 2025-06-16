"""Evaluate submitted Teachable Machine models on a common test set."""

import json
import os
import time
import zipfile

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

TEST_DIR = "./Data/Test"
SUBMISSION_DIR = "./Student_Submissions"


start = time.time()

def load_model_function(model_path):
    """
    Load the Keras model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path, compile=False)

def prepare_image(image_path):
    """Load and normalize an image for prediction."""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    return data


def predict(model, class_names, image_path):
    """Return predicted class name and confidence for a single image."""
    data = prepare_image(image_path)
    prediction = model.predict(data, verbose=0)[0]
    index = int(np.argmax(prediction))
    class_name = class_names[index].strip().lower()
    confidence_score = float(prediction[index])
    return class_name, confidence_score

def unzip_submissions(directory: str) -> None:
    """Extract zip files placed in the submissions directory."""
    for fname in os.listdir(directory):
        if fname.endswith(".zip"):
            zip_path = os.path.join(directory, fname)
            extract_dir = os.path.join(directory, fname[:-4])
            if not os.path.exists(extract_dir):
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)


def evaluate_student(student_dir: str, test_images: list[str]) -> dict:
    """Return accuracy and mean confidence for a single student model."""
    model_path = os.path.join(student_dir, "keras_model.h5")
    labels_path = os.path.join(student_dir, "labels.txt")

    model = load_model_function(model_path)
    with open(labels_path, "r") as f:
        class_names = f.read().splitlines()

    correct = 0
    confidences = []

    for img_name in tqdm(test_images, desc=os.path.basename(student_dir), leave=False):
        gt_label = img_name.split("_")[0].lower()
        image_path = os.path.join(TEST_DIR, img_name)
        pred_name, confidence = predict(model, class_names, image_path)
        confidences.append(confidence)
        if pred_name == gt_label:
            correct += 1

    accuracy = correct / len(test_images) if test_images else 0
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
    }


def main() -> None:
    unzip_submissions(SUBMISSION_DIR)

    test_images = [img for img in os.listdir(TEST_DIR) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    results = []
    for student in os.listdir(SUBMISSION_DIR):
        student_path = os.path.join(SUBMISSION_DIR, student)
        if student.startswith(".") or not os.path.isdir(student_path):
            continue
        metrics = evaluate_student(student_path, test_images)
        results.append({"student": student, **metrics})

    # Sort by accuracy then mean confidence
    leaderboard = sorted(results, key=lambda x: (x["accuracy"], x["mean_confidence"]), reverse=True)

    print("\nLeaderboard:")
    for rank, entry in enumerate(leaderboard, start=1):
        print(f"{rank:2d}. {entry['student']:20s} Acc: {entry['accuracy']:.2%}  MeanConf: {entry['mean_confidence']:.2f}")

    with open(os.path.join(TEST_DIR, "results.json"), "w") as f:
        json.dump(leaderboard, f, indent=2)


if __name__ == "__main__":
    main()
    end = time.time()
    print(f"Evaluation finished in {end - start:.2f} seconds")

