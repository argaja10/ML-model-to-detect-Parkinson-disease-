import numpy as np
import cv2 as cv
import os
import argparse
from imutils import build_montages
from imutils import paths
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="Path to the input dataset")
parser.add_argument("-t", "--trials", type=int, default=5, help="Number of trials to run")
args = vars(parser.parse_args())

def extract_features(image):
    """
    Extract HOG features from an image.
    """
    hog_features = feature.hog(image, orientations=9, 
                               pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                               transform_sqrt=True, block_norm="L1")
    return hog_features

def load_data_and_labels(dataset_path):
    """
    Load images, extract features, and get labels from the dataset.
    """
    image_paths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        image = cv.imread(image_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        resized_image = cv.resize(gray_image, (200, 200))
        threshold_image = cv.threshold(resized_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        features = extract_features(threshold_image)
        data.append(features)
        labels.append(label)

    return np.array(data), np.array(labels)

# Paths to the training and testing datasets
train_path = os.path.sep.join([args["dataset"], "training"])
test_path = os.path.sep.join([args["dataset"], "testing"])

# Load training and testing datasets
print("Loading data...")
train_data, train_labels = load_data_and_labels(train_path)
test_data, test_labels = load_data_and_labels(test_path)

# Encode the labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Initialize the trials dictionary
trials = {}

# Run the specified number of trials
for trial in range(args["trials"]):
    print(f"Training model {trial + 1} of {args['trials']}...")
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)
    metrics = {}

    # Compute the confusion matrix
    cm = confusion_matrix(test_labels, predictions).flatten()
    tn, fp, fn, tp = cm
    metrics["accuracy"] = (tp + tn) / float(cm.sum())
    metrics["sensitivity"] = tp / float(tp + fn)
    metrics["specificity"] = tn / float(tn + fp)

    # Update the trials dictionary with the list of values for the current metric
    for key, value in metrics.items():
        trials[key] = trials.get(key, []) + [value]

# Print the mean and standard deviation for each metric
for metric in ("accuracy", "sensitivity", "specificity"):
    values = trials[metric]
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric}\n{'=' * len(metric)}\nMean: {mean:.4f}, Std Dev: {std:.4f}\n")

# Randomly select a few images for visualization
test_image_paths = list(paths.list_images(test_path))
random_indices = np.random.choice(len(test_image_paths), size=25, replace=False)
visualization_images = []

for idx in random_indices:
    image_path = test_image_paths[idx]
    original_image = cv.imread(image_path)
    display_image = cv.resize(original_image.copy(), (128, 128))
    gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    resized_image = cv.resize(gray_image, (200, 200))
    threshold_image = cv.threshold(resized_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    features = extract_features(threshold_image)
    prediction = classifier.predict([features])
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    color = (0, 255, 0) if predicted_label == "healthy" else (0, 0, 255)
    cv.putText(display_image, predicted_label, (3, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    visualization_images.append(display_image)

# Create a montage of the selected images
montage = build_montages(visualization_images, (128, 128), (5, 5))[0]

# Show the output montage
cv.imshow("Output", montage)
cv.waitKey(0)
