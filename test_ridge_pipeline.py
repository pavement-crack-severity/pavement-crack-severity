import cv2
import numpy as np

from segmentation_ridge_pipeline import (
    segment_crack_ridge_pipeline
)


# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = (
    "data/CRACK500/traincrop/"
    "20160222_081011_1281_721.jpg"
)

GT_PATH = (
    "data/CRACK500/traincrop/"
    "20160222_081011_1281_721.png"
)

OUTPUT_DIR = "outputs/ridge_pipeline"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

import os

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_COLOR
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# ============================================================
# GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

gray_clahe = clahe.apply(
    gray
)


cv2.imwrite(
    f"{OUTPUT_DIR}/01_clahe.png",
    gray_clahe
)


# ============================================================
# RUN RIDGE PIPELINE
# ============================================================

final_mask, debug = (
    segment_crack_ridge_pipeline(
        gray_clahe,

        ridge_scales=(1, 2, 3, 4),

        tile_size=32,

        k_high=3.0,

        k_low=1.5,

        min_size=25,

        min_elongation=3.0,

        max_solidity=0.6
    )
)


# ============================================================
# SAVE RIDGE RESPONSE
# ============================================================

ridge_response = debug[
    "ridge_response"
]

ridge_uint8 = (
    ridge_response * 255
).clip(
    0,
    255
).astype(
    np.uint8
)


cv2.imwrite(
    f"{OUTPUT_DIR}/02_ridge_response.png",
    ridge_uint8
)


# ============================================================
# SAVE RAW HYSTERESIS MASK
# ============================================================

raw_mask = (
    debug["raw_mask"]
    .astype(np.uint8)
    * 255
)

cv2.imwrite(
    f"{OUTPUT_DIR}/03_raw_hysteresis.png",
    raw_mask
)


# ============================================================
# SAVE RECONSTRUCTION MASK
# ============================================================

cleaned_mask = (
    debug["cleaned_mask"]
    .astype(np.uint8)
    * 255
)

cv2.imwrite(
    f"{OUTPUT_DIR}/04_reconstruction.png",
    cleaned_mask
)


# ============================================================
# SAVE SHAPE-FILTERED MASK
# ============================================================

shape_mask = (
    debug["shape_filtered"]
    .astype(np.uint8)
    * 255
)

cv2.imwrite(
    f"{OUTPUT_DIR}/05_shape_filtered.png",
    shape_mask
)


# ============================================================
# SAVE FINAL MASK
# ============================================================

final_mask_uint8 = (
    final_mask.astype(np.uint8)
    * 255
)

cv2.imwrite(
    f"{OUTPUT_DIR}/06_final_mask.png",
    final_mask_uint8
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = cv2.imread(
    GT_PATH,
    cv2.IMREAD_GRAYSCALE
)

if ground_truth is None:
    raise FileNotFoundError(
        f"Could not load ground truth: {GT_PATH}"
    )


ground_truth = (
    ground_truth > 0
)


# ============================================================
# CHECK SIZE
# ============================================================

if final_mask.shape != ground_truth.shape:

    raise ValueError(
        "Final mask and ground-truth mask "
        "have different dimensions."
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

predicted = final_mask.astype(bool)

tp = np.logical_and(
    predicted,
    ground_truth
).sum()

fp = np.logical_and(
    predicted,
    ~ground_truth
).sum()

fn = np.logical_and(
    ~predicted,
    ground_truth
).sum()

tn = np.logical_and(
    ~predicted,
    ~ground_truth
).sum()


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

union = tp + fp + fn

iou = (
    tp / union
    if union > 0
    else 0.0
)


# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------

precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0.0
)


# ------------------------------------------------------------
# Recall
# ------------------------------------------------------------

recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0.0
)


# ------------------------------------------------------------
# F1 / Dice
# ------------------------------------------------------------

f1 = (
    2 * precision * recall
    /
    (precision + recall)
    if precision + recall > 0
    else 0.0
)


# ------------------------------------------------------------
# Specificity
# ------------------------------------------------------------

specificity = (
    tn / (tn + fp)
    if tn + fp > 0
    else 0.0
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

gt_coverage = (
    ground_truth.sum()
    /
    ground_truth.size
    *
    100
)

pred_coverage = (
    predicted.sum()
    /
    predicted.size
    *
    100
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("======================================================")
print("RIDGE-BASED JASSI SEGMENTATION")
print("======================================================")
print()

print("Image:")
print(IMAGE_PATH)
print()

print("Ground-truth crack pixels:")
print(int(ground_truth.sum()))
print()

print(
    "Ground-truth coverage:",
    f"{gt_coverage:.2f}%"
)
print()

print("Predicted crack pixels:")
print(int(predicted.sum()))
print()

print(
    "Predicted coverage:",
    f"{pred_coverage:.2f}%"
)
print()

print("------------------------------------------------------")
print("SEGMENTATION METRICS")
print("------------------------------------------------------")

print(
    "IoU:",
    f"{iou:.4f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall:",
    f"{recall:.4f}"
)

print(
    "F1 / Dice:",
    f"{f1:.4f}"
)

print(
    "Specificity:",
    f"{specificity:.4f}"
)

print()

print("------------------------------------------------------")
print("OUTPUT FILES")
print("------------------------------------------------------")

print(
    "1.",
    f"{OUTPUT_DIR}/01_clahe.png"
)

print(
    "2.",
    f"{OUTPUT_DIR}/02_ridge_response.png"
)

print(
    "3.",
    f"{OUTPUT_DIR}/03_raw_hysteresis.png"
)

print(
    "4.",
    f"{OUTPUT_DIR}/04_reconstruction.png"
)

print(
    "5.",
    f"{OUTPUT_DIR}/05_shape_filtered.png"
)

print(
    "6.",
    f"{OUTPUT_DIR}/06_final_mask.png"
)

print()

print(
    "Ridge-based segmentation test "
    "completed successfully."
)