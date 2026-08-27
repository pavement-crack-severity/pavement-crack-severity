import os
import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "outputs/ml_v2/crack2_rf_mask_v2.png"

OUTPUT_DIR = "outputs/ml_v2"

SKELETON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "crack2_rf_skeleton.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RF MASK
# ============================================================

mask = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        f"Could not load: {INPUT_PATH}"
    )


print()
print("======================================================")
print("RF MASK → SKELETONIZATION")
print("======================================================")

print()
print("Input:")
print(INPUT_PATH)

print()
print("Mask shape:")
print(mask.shape)


# ============================================================
# CONVERT TO BINARY
# ============================================================

binary_mask = mask > 127

print()
print("Input crack pixels:")
print(np.count_nonzero(binary_mask))


# ============================================================
# SKELETONIZATION
# ============================================================

print()
print("Skeletonizing...")

skeleton = skeletonize(
    binary_mask
)


# ============================================================
# CONVERT TO UINT8
# ============================================================

skeleton_image = (
    skeleton.astype(np.uint8) * 255
)


# ============================================================
# SAVE SKELETON
# ============================================================

cv2.imwrite(
    SKELETON_OUTPUT,
    skeleton_image
)


# ============================================================
# CALCULATE SKELETON STATISTICS
# ============================================================

skeleton_pixels = np.count_nonzero(
    skeleton
)

total_pixels = (
    skeleton.shape[0] *
    skeleton.shape[1]
)

skeleton_coverage = (
    skeleton_pixels /
    total_pixels
) * 100


# ============================================================
# RESULTS
# ============================================================

print()
print("------------------------------------------")

print(
    "Skeleton pixels:",
    skeleton_pixels
)

print(
    f"Skeleton coverage: "
    f"{skeleton_coverage:.4f}%"
)

print()
print("Skeleton output:")
print(SKELETON_OUTPUT)

print()
print("======================================================")
print("SKELETONIZATION COMPLETED SUCCESSFULLY")
print("======================================================")