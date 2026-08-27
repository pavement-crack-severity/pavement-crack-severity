import cv2

from segmentation import (
    apply_adaptive_response_threshold,
    clean_binary_mask,
    remove_small_components
)


# ============================================================
# INPUT
# ============================================================

INPUT_PATH = (
    "outputs/gabor/crack2_gabor.png"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = (
    "outputs/segmentation/crack2_adaptive.png"
)


# ============================================================
# LOAD GABOR IMAGE
# ============================================================

gabor_image = cv2.imread(
    INPUT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if gabor_image is None:
    raise FileNotFoundError(
        f"Could not load: {INPUT_PATH}"
    )


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================

adaptive_mask = apply_adaptive_response_threshold(
    gabor_image,
    block_size=31,
    C=-2
)


# ============================================================
# MORPHOLOGICAL CLEANING
# ============================================================

cleaned_mask = clean_binary_mask(
    adaptive_mask
)


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

final_mask = remove_small_components(
    cleaned_mask,
    min_area=300
)


# ============================================================
# SAVE
# ============================================================

success = cv2.imwrite(
    OUTPUT_PATH,
    final_mask
)

if not success:
    raise RuntimeError(
        f"Could not save: {OUTPUT_PATH}"
    )


# ============================================================
# RESULT
# ============================================================

crack_pixels = cv2.countNonZero(
    final_mask
)

total_pixels = final_mask.size

coverage = (
    crack_pixels /
    total_pixels *
    100
)

print("==========================================")
print("ADAPTIVE BASELINE CREATED")
print("==========================================")
print()

print("Input:")
print(INPUT_PATH)
print()

print("Output:")
print(OUTPUT_PATH)
print()

print("Crack pixels:")
print(crack_pixels)
print()

print("Crack coverage:")
print(f"{coverage:.2f}%")
print()

print(
    "Adaptive baseline created successfully."
)