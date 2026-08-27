import os
import cv2
import joblib
import numpy as np

from ml_features import extract_ml_features


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = (
    "data/CRACK500/traincrop/"
    "20160222_081011_1281_721.jpg"
)

MODEL_PATH = (
    "models/crack_random_forest_v2.joblib"
)

OUTPUT_DIR = "outputs/ml_v2"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print()
print("==========================================")
print("RANDOM FOREST V2 PREDICTION")
print("==========================================")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Random Forest V2...")

model_data = joblib.load(
    MODEL_PATH
)

# V2 model was saved as a dictionary
if isinstance(model_data, dict):
    clf = model_data["model"]
    threshold = model_data["threshold"]
else:
    clf = model_data
    threshold = 0.80

print("Model loaded successfully.")

print(
    f"Threshold: {threshold:.2f}"
)


# ============================================================
# EXTRACT 9 FEATURES
# ============================================================

print()
print("Extracting 9 classical features...")

features = extract_ml_features(
    image
)

h, w, feature_count = features.shape

print(
    "Feature shape:",
    features.shape
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("Predicting crack probability...")

X = features.reshape(
    -1,
    feature_count
)

probabilities = clf.predict_proba(
    X
)[:, 1]

probability_map = probabilities.reshape(
    h,
    w
)


# ============================================================
# SAVE PROBABILITY MAP
# ============================================================

probability_image = (
    probability_map * 255
).clip(
    0,
    255
).astype(
    np.uint8
)

probability_path = os.path.join(
    OUTPUT_DIR,
    "crack2_rf_probability_v2.png"
)

cv2.imwrite(
    probability_path,
    probability_image
)


# ============================================================
# THRESHOLD AT 0.80
# ============================================================

binary_mask = (
    probability_map >= threshold
)


mask = (
    binary_mask.astype(
        np.uint8
    ) * 255
)


# ============================================================
# LIGHT MORPHOLOGICAL CLEANING
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

cleaned_mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

cleaned_mask = cv2.morphologyEx(
    cleaned_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


# ============================================================
# SAVE MASK
# ============================================================

mask_path = os.path.join(
    OUTPUT_DIR,
    "crack2_rf_mask_v2.png"
)

cv2.imwrite(
    mask_path,
    cleaned_mask
)


# ============================================================
# COVERAGE
# ============================================================

crack_pixels = np.count_nonzero(
    cleaned_mask
)

total_pixels = (
    h * w
)

coverage = (
    crack_pixels /
    total_pixels
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("------------------------------------------")

print(
    "Image shape:",
    image.shape
)

print(
    "Predicted crack pixels:",
    crack_pixels
)

print(
    f"Predicted crack coverage: "
    f"{coverage:.2f}%"
)

print()
print(
    "Probability map:"
)

print(
    probability_path
)

print()
print(
    "Final RF mask:"
)

print(
    mask_path
)

print()
print("==========================================")
print("RF V2 PREDICTION COMPLETED")
print("==========================================")