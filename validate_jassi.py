import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

GROUND_TRUTH_PATH = (
    "data/CRACK500/traincrop/"
    "20160222_081011_1281_721.png"
)

OLD_MASK_PATH = (
    "outputs/segmentation/"
    "crack2_adaptive.png"
)

NEW_MASK_PATH = (
    "outputs/segmentation/"
    "crack2_fused_final_new.png"
)


# ============================================================
# LOAD MASK
# ============================================================

def load_binary_mask(path):
    """
    Load an image and convert it into a binary mask.

    Any non-zero pixel is treated as crack.
    """

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {path}"
        )

    mask = image > 0

    return mask


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    predicted,
    ground_truth
):
    """
    Calculate segmentation metrics.
    """

    predicted = predicted.astype(bool)
    ground_truth = ground_truth.astype(bool)

    true_positive = np.logical_and(
        predicted,
        ground_truth
    ).sum()

    false_positive = np.logical_and(
        predicted,
        np.logical_not(ground_truth)
    ).sum()

    false_negative = np.logical_and(
        np.logical_not(predicted),
        ground_truth
    ).sum()

    intersection = np.logical_and(
        predicted,
        ground_truth
    ).sum()

    union = np.logical_or(
        predicted,
        ground_truth
    ).sum()

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = (
            true_positive /
            (true_positive + false_positive)
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = (
            true_positive /
            (true_positive + false_negative)
        )

    # --------------------------------------------------------
    # F1 / Dice
    # --------------------------------------------------------

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True Positive": true_positive,
        "False Positive": false_positive,
        "False Negative": false_negative
    }


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = load_binary_mask(
    GROUND_TRUTH_PATH
)


# ============================================================
# LOAD OLD SEGMENTATION
# ============================================================

old_mask = load_binary_mask(
    OLD_MASK_PATH
)


# ============================================================
# LOAD NEW SEGMENTATION
# ============================================================

new_mask = load_binary_mask(
    NEW_MASK_PATH
)


# ============================================================
# CHECK IMAGE SIZES
# ============================================================

print("==========================================")
print("JASSI SEGMENTATION VALIDATION")
print("==========================================")
print()

print(
    "Ground-truth shape:",
    ground_truth.shape
)

print(
    "Old mask shape:",
    old_mask.shape
)

print(
    "New mask shape:",
    new_mask.shape
)

print()


if (
    old_mask.shape != ground_truth.shape
    or
    new_mask.shape != ground_truth.shape
):
    raise ValueError(
        "Ground-truth and predicted masks "
        "must have identical dimensions."
    )


# ============================================================
# COVERAGE
# ============================================================

total_pixels = ground_truth.size

ground_truth_pixels = np.count_nonzero(
    ground_truth
)

old_pixels = np.count_nonzero(
    old_mask
)

new_pixels = np.count_nonzero(
    new_mask
)


ground_truth_coverage = (
    ground_truth_pixels /
    total_pixels *
    100
)

old_coverage = (
    old_pixels /
    total_pixels *
    100
)

new_coverage = (
    new_pixels /
    total_pixels *
    100
)


# ============================================================
# CALCULATE METRICS
# ============================================================

old_metrics = calculate_metrics(
    old_mask,
    ground_truth
)

new_metrics = calculate_metrics(
    new_mask,
    ground_truth
)


# ============================================================
# PRINT GROUND-TRUTH INFORMATION
# ============================================================

print("------------------------------------------")
print("GROUND TRUTH")
print("------------------------------------------")

print(
    "Crack pixels:",
    ground_truth_pixels
)

print(
    "Crack coverage:",
    f"{ground_truth_coverage:.2f}%"
)

print()


# ============================================================
# OLD METHOD
# ============================================================

print("------------------------------------------")
print("OLD ADAPTIVE SEGMENTATION")
print("------------------------------------------")

print(
    "Predicted crack pixels:",
    old_pixels
)

print(
    "Predicted coverage:",
    f"{old_coverage:.2f}%"
)

print(
    "IoU:",
    f"{old_metrics['IoU']:.4f}"
)

print(
    "Precision:",
    f"{old_metrics['Precision']:.4f}"
)

print(
    "Recall:",
    f"{old_metrics['Recall']:.4f}"
)

print(
    "F1 / Dice:",
    f"{old_metrics['F1']:.4f}"
)

print()


# ============================================================
# NEW METHOD
# ============================================================

print("------------------------------------------")
print("NEW GABOR + BLACK-HAT SEGMENTATION")
print("------------------------------------------")

print(
    "Predicted crack pixels:",
    new_pixels
)

print(
    "Predicted coverage:",
    f"{new_coverage:.2f}%"
)

print(
    "IoU:",
    f"{new_metrics['IoU']:.4f}"
)

print(
    "Precision:",
    f"{new_metrics['Precision']:.4f}"
)

print(
    "Recall:",
    f"{new_metrics['Recall']:.4f}"
)

print(
    "F1 / Dice:",
    f"{new_metrics['F1']:.4f}"
)

print()


# ============================================================
# COMPARISON
# ============================================================

print("==========================================")
print("COMPARISON")
print("==========================================")

iou_change = (
    new_metrics["IoU"]
    -
    old_metrics["IoU"]
)

f1_change = (
    new_metrics["F1"]
    -
    old_metrics["F1"]
)

precision_change = (
    new_metrics["Precision"]
    -
    old_metrics["Precision"]
)

recall_change = (
    new_metrics["Recall"]
    -
    old_metrics["Recall"]
)


print(
    "IoU change:",
    f"{iou_change:+.4f}"
)

print(
    "Precision change:",
    f"{precision_change:+.4f}"
)

print(
    "Recall change:",
    f"{recall_change:+.4f}"
)

print(
    "F1 change:",
    f"{f1_change:+.4f}"
)

print()


# ============================================================
# BEST METHOD
# ============================================================

if new_metrics["F1"] > old_metrics["F1"]:

    print(
        "RESULT: New Gabor + Black-hat method "
        "has better F1 score."
    )

elif new_metrics["F1"] < old_metrics["F1"]:

    print(
        "RESULT: Old adaptive method "
        "has better F1 score."
    )

else:

    print(
        "RESULT: Both methods have the same F1 score."
    )

print()
print(
    "Ground-truth validation completed successfully."
)