import os
import cv2
import joblib
import warnings
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# Suppress the repeated sklearn parallel warning
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.*"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/CRACK500/traincrop"

MODEL_PATH = "models/crack_random_forest_v2.joblib"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SAMPLES_PER_IMAGE = 2000

N_ESTIMATORS = 200
MAX_DEPTH = 14

# Thresholds to test
THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.05
)


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
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    """
    Nine classical image features.

    1. intensity
    2. clahe
    3. gabor
    4. blackhat
    5. frangi
    6. gradient
    7. local_mean
    8. local_std
    9. local_contrast

    No deep learning.
    """

    from skimage.filters import frangi

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_float = gray.astype(
        np.float32
    )

    # --------------------------------------------------------
    # 1. INTENSITY
    # --------------------------------------------------------

    intensity = normalize_feature(
        gray_float
    )

    # --------------------------------------------------------
    # 2. CLAHE
    # --------------------------------------------------------

    clahe_operator = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe = clahe_operator.apply(
        gray
    )

    clahe = normalize_feature(
        clahe
    )

    # --------------------------------------------------------
    # 3. GABOR
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. BLACK-HAT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. FRANGI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. GRADIENT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. LOCAL MEAN
    # --------------------------------------------------------

    local_mean_raw = cv2.GaussianBlur(
        gray_float,
        (9, 9),
        0
    )

    local_mean = normalize_feature(
        local_mean_raw
    )

    # --------------------------------------------------------
    # 8. LOCAL STANDARD DEVIATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 9. LOCAL CONTRAST
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STACK FEATURES
    # --------------------------------------------------------

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

    rng.shuffle(
        indices
    )

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
# LOAD GROUND TRUTH
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
# CREATE TRAINING DATA
# ============================================================

def create_training_data(
    pairs,
    samples_per_image
):

    X_list = []
    y_list = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    processed = 0

    print()
    print(
        "Creating balanced pixel samples..."
    )

    for image_path, mask_path in pairs:

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        mask = load_mask(
            mask_path
        )

        features = extract_features(
            image
        )

        h, w, f = features.shape

        X = features.reshape(
            -1,
            f
        )

        y = mask.reshape(
            -1
        )

        crack_indices = np.where(
            y == 1
        )[0]

        noncrack_indices = np.where(
            y == 0
        )[0]

        if len(crack_indices) == 0:
            continue

        n_each = min(
            samples_per_image // 2,
            len(crack_indices),
            len(noncrack_indices)
        )

        if n_each == 0:
            continue

        crack_sample = rng.choice(
            crack_indices,
            size=n_each,
            replace=False
        )

        noncrack_sample = rng.choice(
            noncrack_indices,
            size=n_each,
            replace=False
        )

        selected = np.concatenate(
            [
                crack_sample,
                noncrack_sample
            ]
        )

        X_list.append(
            X[selected]
        )

        y_list.append(
            y[selected]
        )

        processed += 1

        if processed % 100 == 0:

            print(
                f"Processed: "
                f"{processed}/{len(pairs)}"
            )

    if len(X_list) == 0:

        raise RuntimeError(
            "No training samples were created."
        )

    X_train = np.concatenate(
        X_list,
        axis=0
    )

    y_train = np.concatenate(
        y_list,
        axis=0
    )

    return (
        X_train,
        y_train
    )


# ============================================================
# FAST METRICS FROM CONFUSION COUNTS
# ============================================================

def calculate_metrics(
    tp,
    tn,
    fp,
    fn
):

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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "specificity": specificity
    }


# ============================================================
# ONE-PASS VALIDATION
# ============================================================

def evaluate_thresholds_once(
    model,
    pairs,
    thresholds
):

    threshold_count = len(
        thresholds
    )

    tp = np.zeros(
        threshold_count,
        dtype=np.int64
    )

    tn = np.zeros(
        threshold_count,
        dtype=np.int64
    )

    fp = np.zeros(
        threshold_count,
        dtype=np.int64
    )

    fn = np.zeros(
        threshold_count,
        dtype=np.int64
    )

    print()
    print(
        "Predicting validation images ONCE..."
    )

    total = len(pairs)

    for index, (
        image_path,
        mask_path
    ) in enumerate(pairs, start=1):

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        gt = load_mask(
            mask_path
        )

        features = extract_features(
            image
        )

        h, w, f = features.shape

        X = features.reshape(
            -1,
            f
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # RF prediction happens ONLY ONCE here.
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X
        )[:, 1]

        gt_flat = gt.reshape(
            -1
        ).astype(
            bool
        )

        # ----------------------------------------------------
        # Evaluate all thresholds using the SAME
        # probability array.
        # ----------------------------------------------------

        for t_index, threshold in enumerate(
            thresholds
        ):

            prediction = (
                probabilities >= threshold
            )

            tp[t_index] += np.count_nonzero(
                prediction & gt_flat
            )

            tn[t_index] += np.count_nonzero(
                (~prediction) & (~gt_flat)
            )

            fp[t_index] += np.count_nonzero(
                prediction & (~gt_flat)
            )

            fn[t_index] += np.count_nonzero(
                (~prediction) & gt_flat
            )

        if (
            index % 10 == 0
            or index == total
        ):

            print(
                f"Validation progress: "
                f"{index}/{total}"
            )

    results = []

    for i, threshold in enumerate(
        thresholds
    ):

        metrics = calculate_metrics(
            int(tp[i]),
            int(tn[i]),
            int(fp[i]),
            int(fn[i])
        )

        results.append(
            (
                float(threshold),
                metrics
            )
        )

    return results


# ============================================================
# ONE-PASS TEST
# ============================================================

def evaluate_test_once(
    model,
    pairs,
    threshold
):

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    print()
    print(
        "Predicting TEST images..."
    )

    total = len(pairs)

    for index, (
        image_path,
        mask_path
    ) in enumerate(pairs, start=1):

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        gt = load_mask(
            mask_path
        )

        features = extract_features(
            image
        )

        h, w, f = features.shape

        X = features.reshape(
            -1,
            f
        )

        probabilities = model.predict_proba(
            X
        )[:, 1]

        prediction = (
            probabilities >= threshold
        )

        gt_flat = gt.reshape(
            -1
        ).astype(
            bool
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
                f"Test progress: "
                f"{index}/{total}"
            )

    return calculate_metrics(
        total_tp,
        total_tn,
        total_fp,
        total_fn
    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "======================================================"
)

print(
    "RANDOM FOREST V2 — OPTIMIZED VALIDATION"
)

print(
    "======================================================"
)


# ============================================================
# FIND DATA
# ============================================================

pairs = get_image_mask_pairs()

print()
print(
    f"Image-mask pairs found: {len(pairs)}"
)

if len(pairs) == 0:

    raise RuntimeError(
        "No JPG/PNG pairs found."
    )


# ============================================================
# SPLIT
# ============================================================

train_pairs, val_pairs, test_pairs = (
    split_dataset(pairs)
)

print()
print(
    "DATASET SPLIT"
)

print(
    f"Training:   {len(train_pairs)}"
)

print(
    f"Validation: {len(val_pairs)}"
)

print(
    f"Testing:    {len(test_pairs)}"
)


# ============================================================
# TRAINING DATA
# ============================================================

print()
print(
    "======================================================"
)

print(
    "CREATING TRAINING DATA"
)

print(
    "======================================================"
)

X_train, y_train = create_training_data(
    train_pairs,
    SAMPLES_PER_IMAGE
)

print()
print(
    "Feature matrix:"
)

print(
    X_train.shape
)

print(
    f"Crack samples:     "
    f"{np.sum(y_train == 1)}"
)

print(
    f"Non-crack samples: "
    f"{np.sum(y_train == 0)}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print(
    "======================================================"
)

print(
    "TRAINING RANDOM FOREST"
)

print(
    "======================================================"
)

print(
    f"Trees: {N_ESTIMATORS}"
)

print(
    f"Maximum depth: {MAX_DEPTH}"
)

print(
    "Training started..."
)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "intensity",
    "clahe",
    "gabor",
    "blackhat",
    "frangi",
    "gradient",
    "local_mean",
    "local_std",
    "local_contrast"
]

print()
print(
    "======================================================"
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "======================================================"
)

importance = model.feature_importances_

for name, value in sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{name:<18} {value:.4f}"
    )


# ============================================================
# VALIDATION
# ============================================================

print()
print(
    "======================================================"
)

print(
    "VALIDATION THRESHOLD TUNING"
)

print(
    "======================================================"
)

print(
    "IMPORTANT: Random Forest will predict each "
    "validation image only ONCE."
)

validation_results = evaluate_thresholds_once(
    model,
    val_pairs,
    THRESHOLDS
)


# ============================================================
# PRINT THRESHOLD RESULTS
# ============================================================

print()
print(
    "------------------------------------------------------"
)

print(
    "VALIDATION RESULTS"
)

print(
    "------------------------------------------------------"
)

best_threshold = 0.50
best_iou = -1
best_metrics = None

for threshold, metrics in validation_results:

    print(
        f"Threshold={threshold:.2f} | "
        f"IoU={metrics['iou']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}"
    )

    if metrics["iou"] > best_iou:

        best_iou = metrics["iou"]

        best_threshold = threshold

        best_metrics = metrics


print()
print(
    f"BEST VALIDATION THRESHOLD: "
    f"{best_threshold:.2f}"
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print()
print(
    "======================================================"
)

print(
    "BEST VALIDATION RESULT"
)

print(
    "======================================================"
)

print(
    f"IoU:         "
    f"{best_metrics['iou']:.4f}"
)

print(
    f"Precision:   "
    f"{best_metrics['precision']:.4f}"
)

print(
    f"Recall:      "
    f"{best_metrics['recall']:.4f}"
)

print(
    f"F1 / Dice:   "
    f"{best_metrics['f1']:.4f}"
)

print(
    f"Specificity: "
    f"{best_metrics['specificity']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print(
    "======================================================"
)

print(
    "FINAL TEST — UNSEEN IMAGES"
)

print(
    "======================================================"
)

print(
    f"Using threshold: "
    f"{best_threshold:.2f}"
)

print(
    "The test images were NOT used for threshold tuning."
)

test_metrics = evaluate_test_once(
    model,
    test_pairs,
    best_threshold
)


# ============================================================
# TEST RESULT
# ============================================================

print()
print(
    "======================================================"
)

print(
    "FINAL TEST RESULT"
)

print(
    "======================================================"
)

print(
    f"IoU:         "
    f"{test_metrics['iou']:.4f}"
)

print(
    f"Precision:   "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall:      "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1 / Dice:   "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Specificity: "
    f"{test_metrics['specificity']:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_data = {
    "model": model,
    "threshold": best_threshold,
    "feature_names": feature_names,
    "random_state": RANDOM_STATE,
    "trees": N_ESTIMATORS,
    "max_depth": MAX_DEPTH
}

joblib.dump(
    model_data,
    MODEL_PATH
)

print()
print(
    "======================================================"
)

print(
    "RANDOM FOREST V2 COMPLETED"
)

print(
    "======================================================"
)

print(
    f"Model saved to:"
)

print(
    MODEL_PATH
)

print()
print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    "Model is ready for unseen-image prediction."
)

print()