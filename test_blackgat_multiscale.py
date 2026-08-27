import cv2

from segmentation import (
    create_multiscale_blackhat_response
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = "outputs/gabor/crack2_gabor.png"

OUTPUT_PATH = (
    "outputs/segmentation/"
    "crack2_blackhat_multiscale.png"
)


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {INPUT_PATH}"
    )


# --------------------------------------------------
# Multi-scale black-hat response
# --------------------------------------------------

response = create_multiscale_blackhat_response(
    image,
    scales=(7, 13, 19, 25)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

success = cv2.imwrite(
    OUTPUT_PATH,
    response
)

if not success:
    raise RuntimeError(
        f"Could not save: {OUTPUT_PATH}"
    )


# --------------------------------------------------
# Result
# --------------------------------------------------

print("==========================================")
print("MULTI-SCALE BLACK-HAT TEST")
print("==========================================")
print()
print("Input:")
print(INPUT_PATH)
print()
print("Scales:")
print("7, 13, 19, 25")
print()
print("Output:")
print(OUTPUT_PATH)
print()
print(
    "Multi-scale black-hat response "
    "created successfully."
)