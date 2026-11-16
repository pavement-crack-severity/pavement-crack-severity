import os
import cv2
import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/CRACK500/traincrop"
MODEL_PATH = "models/crack_random_forest_v2.joblib"
OUTPUT_DIR = "outputs"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_feature(x):

    x = x.astype(np.float32)

    minimum = np.min(x)
    maximum = np.max(x)

    if maximum - minimum < 1e-8:
        return np.zeros_like(x)

    return (
        x - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# EXACT V2 FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    from skimage.filters import frangi

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_float = gray.astype(np.float32)

    # 1. Intensity
    intensity = normalize_feature(
        gray_float
    )

    # 2. CLAHE
    clahe_operator = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe = clahe_operator.apply(gray)

    clahe = normalize_feature(
        clahe
    )

    # 3. Gabor
    gabor_responses = []

    orientations = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    for theta in orientations:

        kernel = cv2.getGaborKernel(
            ksize=(31, 31),
            sigma=5.0,
            theta=theta,
            lambd=20.0,
            gamma=0.5,
            psi=np.pi / 2,
            ktype=cv2.CV_32F
        )

        response = cv2.filter2D(
            gray_float,
            cv2.CV_32F,
            kernel
        )

        response = np.maximum(
            -response,
            0
        )

        gabor_responses.append(
            response
        )

    gabor = np.max(
        np.array(gabor_responses),
        axis=0
    )

    gabor = normalize_feature(
        gabor
    )

    # 4. Black-hat
    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel
    )

    blackhat = normalize_feature(
        blackhat
    )

    # 5. Frangi
    gray_normalized = (
        gray.astype(np.float32) / 255.0
    )

    frangi_response = frangi(
        gray_normalized,
        sigmas=(1, 2, 3, 4),
        alpha=0.5,
        beta=0.5,
        black_ridges=True
    )

    frangi_response = np.nan_to_num(
        frangi_response
    )

    frangi_response = normalize_feature(
        frangi_response
    )

    # 6. Gradient
    gx = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        gx,
        gy
    )

    gradient = normalize_feature(
        gradient
    )

    # 7. Local mean
    local_mean_raw = cv2.GaussianBlur(
        gray_float,
        (9, 9),
        0
    )

    local_mean = normalize_feature(
        local_mean_raw
    )

    # 8. Local standard deviation
    mean_square = cv2.GaussianBlur(
        gray_float ** 2,
        (9, 9),
        0
    )

    variance = (
        mean_square -
        local_mean_raw ** 2
    )

    variance = np.maximum(
        variance,
        0
    )

    local_std_raw = np.sqrt(
        variance
    )

    local_std = normalize_feature(
        local_std_raw
    )

    # 9. Local contrast
    morphology_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    local_min = cv2.erode(
        gray,
        morphology_kernel
    )

    local_max = cv2.dilate(
        gray,
        morphology_kernel
    )

    local_contrast_raw = (
        local_max.astype(np.float32)
        -
        local_min.astype(np.float32)
    )

    local_contrast = normalize_feature(
        local_contrast_raw
    )

    # Stack
    features = np.stack(
        [
            intensity,
            clahe,
            gabor,
            blackhat,
            frangi_response,
            gradient,
            local_mean,
            local_std,
            local_contrast
        ],
        axis=-1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# FIND IMAGE-MASK PAIRS
# ============================================================

def get_image_mask_pairs():

    pairs = []

    files = sorted(
        os.listdir(DATA_DIR)
    )

    for filename in files:

        if not filename.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(
            DATA_DIR,
            filename
        )

        stem = os.path.splitext(
            filename
        )[0]

        mask_path = os.path.join(
            DATA_DIR,
            stem + ".png"
        )

        if os.path.exists(mask_path):

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

    return pairs


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(pairs):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = np.arange(
        len(pairs)
    )

    rng.shuffle(indices)

    total = len(indices)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = (
        train_end +
        int(total * VAL_RATIO)
    )

    train_pairs = [
        pairs[i]
        for i in indices[:train_end]
    ]

    val_pairs = [
        pairs[i]
        for i in indices[
            train_end:val_end
        ]
    ]

    test_pairs = [
        pairs[i]
        for i in indices[val_end:]
    ]

    return (
        train_pairs,
        val_pairs,
        test_pairs
    )


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise FileNotFoundError(
            f"Could not load mask: {path}"
        )

    return (
        mask > 0
    ).astype(
        np.uint8
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, tn, fp, fn):

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
        if tp + tn + fp + fn > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    iou = (
        tp /
        (tp + fp + fn)
        if tp + fp + fn > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# MAIN
# ============================================================

print()
print("======================================================")
print("RANDOM FOREST V2 — SAVED MODEL TEST EVALUATION")
print("======================================================")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print()
print("Loading saved model...")

model_data = joblib.load(
    MODEL_PATH
)

if isinstance(model_data, dict):

    model = model_data["model"]

    threshold = model_data.get(
        "threshold",
        0.80
    )

else:

    model = model_data

    threshold = 0.80


print(
    f"Trees: {model.n_estimators}"
)

print(
    f"Maximum depth: {model.max_depth}"
)

print(
    f"Threshold: {threshold:.2f}"
)


# ------------------------------------------------------------
# Find dataset
# ------------------------------------------------------------

pairs = get_image_mask_pairs()

print()
print(
    f"Image-mask pairs found: {len(pairs)}"
)

if len(pairs) == 0:

    raise RuntimeError(
        "No JPG/PNG image-mask pairs found."
    )


# ------------------------------------------------------------
# Recreate exact split
# ------------------------------------------------------------

train_pairs, val_pairs, test_pairs = (
    split_dataset(pairs)
)

print()
print("DATASET SPLIT")

print(
    f"Training:   {len(train_pairs)}"
)

print(
    f"Validation: {len(val_pairs)}"
)

print(
    f"Testing:    {len(test_pairs)}"
)


# ------------------------------------------------------------
# Evaluate TEST SET
# ------------------------------------------------------------

total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0

print()
print("======================================================")
print("EVALUATING UNSEEN TEST IMAGES")
print("======================================================")

total = len(test_pairs)

for index, (
    image_path,
    mask_path
) in enumerate(
    test_pairs,
    start=1
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Skipping: {image_path}"
        )

        continue

    gt = load_mask(
        mask_path
    )

    features = extract_features(
        image
    )

    h, w, feature_count = (
        features.shape
    )

    X = features.reshape(
        -1,
        feature_count
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    prediction = (
        probabilities >= threshold
    )

    gt_flat = (
        gt.reshape(-1)
        .astype(bool)
    )

    total_tp += np.count_nonzero(
        prediction & gt_flat
    )

    total_tn += np.count_nonzero(
        (~prediction) & (~gt_flat)
    )

    total_fp += np.count_nonzero(
        prediction & (~gt_flat)
    )

    total_fn += np.count_nonzero(
        (~prediction) & gt_flat
    )

    if (
        index % 10 == 0
        or index == total
    ):

        print(
            f"Progress: {index}/{total}"
        )


# ------------------------------------------------------------
# Calculate final metrics
# ------------------------------------------------------------

metrics = calculate_metrics(
    total_tp,
    total_tn,
    total_fp,
    total_fn
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("======================================================")
print("FINAL TEST-SET RESULTS")
print("======================================================")

print(
    f"Accuracy:    {metrics['accuracy']:.6f}"
)

print(
    f"Precision:   {metrics['precision']:.6f}"
)

print(
    f"Recall:      {metrics['recall']:.6f}"
)

print(
    f"F1-score:    {metrics['f1']:.6f}"
)

print(
    f"IoU:         {metrics['iou']:.6f}"
)

print(
    f"Specificity:  {metrics['specificity']:.6f}"
)

print()
print("Confusion counts:")

print(
    f"TP: {metrics['tp']}"
)

print(
    f"TN: {metrics['tn']}"
)

print(
    f"FP: {metrics['fp']}"
)

print(
    f"FN: {metrics['fn']}"
)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "rf_test_metrics.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Metric,Value\n"
    )

    f.write(
        f"Accuracy,{metrics['accuracy']:.6f}\n"
    )

    f.write(
        f"Precision,{metrics['precision']:.6f}\n"
    )

    f.write(
        f"Recall,{metrics['recall']:.6f}\n"
    )

    f.write(
        f"F1-score,{metrics['f1']:.6f}\n"
    )

    f.write(
        f"IoU,{metrics['iou']:.6f}\n"
    )

    f.write(
        f"Specificity,{metrics['specificity']:.6f}\n"
    )

    f.write(
        f"Threshold,{threshold:.6f}\n"
    )

print()
print(
    "Metrics saved to:"
)

print(
    csv_path
)

print()
print("======================================================")
print("EVALUATION COMPLETED")
print("======================================================")