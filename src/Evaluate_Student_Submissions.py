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
MUSTER_LABEL_TXT = "./Student_Submissions/Max_Mustermann/labels.txt"

#TODO: Testset erweitern, um mehr Bilder pro Klasse zu haben

start = time.time()

def check_labelsTXT(student, mustermann_dir=MUSTER_LABEL_TXT):
    """
    Check if the labels.txt file exists in the student directory.
    It also checks whether the label file matches the sample file.
    """
    label_path = os.path.join(student, "labels.txt")
    if not os.path.exists(label_path):
        print(f"labels.txt not found in {student}")
    with open(label_path, "r") as f:
        student_labels = f.readlines()
    with open(mustermann_dir, "r") as f:
        mustermann_labels = f.readlines()
    if student_labels != mustermann_labels:
        #print(f"Labels do not match for {student}. Expected: {mustermann_labels}, Found: {student_labels}")
        raise ValueError(f"Labels in {label_path} do not match the expected labels in {mustermann_dir}. Please check your labels.txt file.")


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
            check_labelsTXT(extract_dir)


def evaluate_student(student_dir: str, test_images: list[str]) -> dict:
    """Return accuracy and mean confidence for a single student model."""
    english_to_german = {
        "cat": "katze",
        "dog": "hund",
        "horse": "pferd",
        "cow": "kuh",
        "chicken": "huhn"
    }

    model_path = os.path.join(student_dir, "keras_model.h5")
    labels_path = os.path.join(student_dir, "labels.txt")

    model = load_model_function(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        entries = [line.strip().split(maxsplit=1) for line in f if line.strip()]

    # 2) Dictionary bauen: index → Name
    index_to_name = {int(idx): name for idx, name in entries}
    # 3) class_names-Liste erzeugen (geordnet nach index)
    class_names = [index_to_name[i] for i in sorted(index_to_name)]

    correct = 0
    confidences = []

    for img_name in tqdm(test_images, desc=os.path.basename(student_dir), leave=False):
        gt_label = img_name.split("_")[0].lower()
        gt_label = english_to_german.get(gt_label,)  # Convert to German
        image_path = os.path.join(TEST_DIR, img_name)
        pred_name, confidence = predict(model, class_names, image_path)
        #print(f'Image: {img_name}, Predicted: {pred_name}, Confidence: {confidence:.2f}, GT: {gt_label}')
        if pred_name == gt_label:
            correct += 1
            confidences.append(confidence)

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
        end_student = time.time()
        print(f"Evaluation for {student} completed in {end_student - start:.2f} seconds")

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

