import cv2
import numpy as np

from segmentation import (
    apply_adaptive_response_threshold,
    clean_binary_mask,
    remove_small_components
)


# ==================================================
# INPUTS
# ==================================================

GABOR_PATH = (
    "outputs/gabor/crack2_gabor.png"
)

BLACKHAT_PATH = (
    "outputs/gabor/crack2_blackhat.png"
)


# ==================================================
# OUTPUTS
# ==================================================

FUSED_RESPONSE_PATH = (
    "outputs/segmentation/crack2_fused_gabor_blackhat.png"
)

FUSED_MASK_PATH = (
    "outputs/segmentation/crack2_fused_mask_new.png"
)

CLEAN_MASK_PATH = (
    "outputs/segmentation/crack2_fused_clean_new.png"
)

FINAL_MASK_PATH = (
    "outputs/segmentation/crack2_fused_final_new.png"
)


# ==================================================
# LOAD RESPONSES
# ==================================================

gabor = cv2.imread(
    GABOR_PATH,
    cv2.IMREAD_GRAYSCALE
)

blackhat = cv2.imread(
    BLACKHAT_PATH,
    cv2.IMREAD_GRAYSCALE
)


if gabor is None:
    raise FileNotFoundError(
        f"Could not load: {GABOR_PATH}"
    )


if blackhat is None:
    raise FileNotFoundError(
        f"Could not load: {BLACKHAT_PATH}"
    )


# ==================================================
# SIZE CHECK
# ==================================================

if gabor.shape != blackhat.shape:
    raise ValueError(
        "Gabor and Black-hat images must have "
        "the same dimensions."
    )


# ==================================================
# NORMALIZATION
# ==================================================

gabor_float = cv2.normalize(
    gabor.astype(np.float32),
    None,
    0.0,
    1.0,
    cv2.NORM_MINMAX
)

blackhat_float = cv2.normalize(
    blackhat.astype(np.float32),
    None,
    0.0,
    1.0,
    cv2.NORM_MINMAX
)


# ==================================================
# CONTROLLED FUSION
# ==================================================

GABOR_WEIGHT = 0.70
BLACKHAT_WEIGHT = 0.30


fused = (
    GABOR_WEIGHT * gabor_float
    +
    BLACKHAT_WEIGHT * blackhat_float
)


# ==================================================
# NORMALIZE FUSED RESPONSE
# ==================================================

fused = cv2.normalize(
    fused,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

fused = fused.astype(np.uint8)


# ==================================================
# SAVE FUSED RESPONSE
# ==================================================

if not cv2.imwrite(
    FUSED_RESPONSE_PATH,
    fused
):
    raise RuntimeError(
        "Could not save fused response."
    )


# ==================================================
# ADAPTIVE THRESHOLD
# ==================================================

fused_mask = apply_adaptive_response_threshold(
    fused,
    block_size=31,
    C=-2
)


cv2.imwrite(
    FUSED_MASK_PATH,
    fused_mask
)


# ==================================================
# MORPHOLOGICAL CLEANING
# ==================================================

cleaned_mask = clean_binary_mask(
    fused_mask
)


cv2.imwrite(
    CLEAN_MASK_PATH,
    cleaned_mask
)


# ==================================================
# REMOVE SMALL COMPONENTS
# ==================================================

final_mask = remove_small_components(
    cleaned_mask,
    min_area=300
)


cv2.imwrite(
    FINAL_MASK_PATH,
    final_mask
)


# ==================================================
# STATISTICS
# ==================================================

total_pixels = final_mask.size

crack_pixels = np.count_nonzero(
    final_mask
)

coverage = (
    crack_pixels /
    total_pixels *
    100
)


# ==================================================
# OUTPUT
# ==================================================

print("==========================================")
print("GABOR + BLACK-HAT FUSION")
print("==========================================")
print()

print("Gabor weight:")
print(GABOR_WEIGHT)
print()

print("Black-hat weight:")
print(BLACKHAT_WEIGHT)
print()

print("Fused response:")
print(FUSED_RESPONSE_PATH)
print()

print("Fused mask:")
print(FUSED_MASK_PATH)
print()

print("Cleaned mask:")
print(CLEAN_MASK_PATH)
print()

print("Final mask:")
print(FINAL_MASK_PATH)
print()

print("Final crack pixels:")
print(crack_pixels)
print()

print("Final crack coverage:")
print(
    f"{coverage:.2f}%"
)
print()

print(
    "Gabor + Black-hat fusion "
    "completed successfully."
)