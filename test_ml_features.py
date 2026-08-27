import cv2
import numpy as np

from ml_features import (
    extract_ml_features,
    feature_names
)


IMAGE_PATH = (
    "data/CRACK500/traincrop/"
    "20160222_081011_1281_721.jpg"
)


image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {IMAGE_PATH}"
    )


features = extract_ml_features(
    image
)


print()
print("==========================================")
print("RANDOM FOREST FEATURE EXTRACTION TEST")
print("==========================================")
print()

print("Image shape:")
print(image.shape)

print()

print("Feature shape:")
print(features.shape)

print()

print("Number of features:")
print(features.shape[-1])

print()

print("Feature names:")

for index, name in enumerate(
    feature_names(),
    start=1
):
    print(
        f"{index}. {name}"
    )

print()

print("Minimum feature value:")
print(features.min())

print()

print("Maximum feature value:")
print(features.max())

print()

print(
    "ML feature extraction completed successfully."
)