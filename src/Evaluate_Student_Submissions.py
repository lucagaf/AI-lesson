import zipfile
import os
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import time

TEST_DIR = "./Data/Test"
TEST_IMAGE  = os.path.join(TEST_DIR, "cat_0.jpeg")

# TODO: Implement TQDM
# TODO: Create .txt file in the Test directory with the ground truth labels --> In the predictions function I can use them
# TODO: Implement a way to handle multiple images in the test directory
# TODO: Implement JSON document to store results

start = time.time()

def load_model_function(model_path):
    """
    Load the Keras model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path, compile=False)


def make_predictions(student_dir, image_path, model):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model and labels
    label_path = os.path.join(student_dir, "labels.txt")
    # print(f'Label Path: {os.path.exists(label_path)}')
    class_names = open(label_path, "r").readlines()
    print(f'Class Names: {class_names}')
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)     # resizing the image to be at least 224x224 and then cropping from the center
    image_array = np.asarray(image)    # turn the image into a numpy array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1     # Normalize the image
    data[0] = normalized_image_array     # Load the image into the array

    # Predicts the model
    prediction = model.predict(data)
    # print(prediction)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)
    return index, class_name.lower(), confidence_score

# Unzip all student submissions
for zip in os.listdir("./Student_Submissions"):
    if zip.endswith(".zip"):
        with zipfile.ZipFile(os.path.join("./Student_Submissions", zip), "r") as zip_ref:
            zip_ref.extractall("./Student_Submissions/" + zip[:-4])

for student in os.listdir("./Student_Submissions"):
    if student.startswith('.') or not os.path.isdir(os.path.join("./Student_Submissions", student)):
        continue
    print(student)
    student_dir = os.path.join("./Student_Submissions", student)
    # student_model = os.path.join(student_dir, "keras_model.h5")
    # print(f'Student Model: {os.path.exists(student_model)}')
    model = load_model(os.path.join(student_dir, "keras_model.h5"))

    for image in os.listdir("./Data/Test"):
        label = image.split("_")[0].lower()
        #print(f'Label: {label} Image: {image}')
        test_image = os.path.join("./Data/Test", image)
        #print(os.path.exists(test_image))
        pred_index, pred_name, confidence_score = make_predictions(student_dir, TEST_IMAGE, model)
        print(f"Ground Truth: {label} / Prediction: {pred_name}, Confidence: {confidence_score:.2f} for {image} by {student}")

end = time.time()
print(end - start)

