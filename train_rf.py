import os
import glob
import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from ml_features import extract_ml_features


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/CRACK500/traincrop"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "crack_random_forest.joblib"
)

# Number of pixels sampled from each image
SAMPLES_PER_IMAGE = 2000

# Random Forest parameters
N_ESTIMATORS = 200
MAX_DEPTH = 14

RANDOM_STATE = 42


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

def get_image_mask_pairs():

    jpg_files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.jpg"
            )
        )
    )

    pairs = []

    for image_path in jpg_files:

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        mask_path = os.path.join(
            DATA_DIR,
            base_name + ".png"
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
# SAMPLE BALANCED PIXELS
# ============================================================

def sample_pixels(
    features,
    mask,
    samples_per_image
):

    h, w, num_features = (
        features.shape
    )

    feature_matrix = features.reshape(
        -1,
        num_features
    )

    labels = (
        mask.reshape(-1) > 0
    ).astype(
        np.uint8
    )

    positive_indices = np.where(
        labels == 1
    )[0]

    negative_indices = np.where(
        labels == 0
    )[0]

    # --------------------------------------------------------
    # We want approximately equal crack/non-crack samples.
    # --------------------------------------------------------

    samples_per_class = (
        samples_per_image // 2
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Positive pixels
    # --------------------------------------------------------

    n_positive = min(
        len(positive_indices),
        samples_per_class
    )

    if n_positive == 0:
        return None, None

    positive_sample = rng.choice(
        positive_indices,
        size=n_positive,
        replace=False
    )

    # --------------------------------------------------------
    # Negative pixels
    # --------------------------------------------------------

    n_negative = min(
        len(negative_indices),
        samples_per_class
    )

    negative_sample = rng.choice(
        negative_indices,
        size=n_negative,
        replace=False
    )

    selected_indices = np.concatenate(
        [
            positive_sample,
            negative_sample
        ]
    )

    X = feature_matrix[
        selected_indices
    ]

    y = labels[
        selected_indices
    ]

    return X, y


# ============================================================
# LOAD DATASET
# ============================================================

pairs = get_image_mask_pairs()


print()
print("======================================================")
print("RANDOM FOREST TRAINING")
print("======================================================")
print()

print(
    "Dataset directory:",
    DATA_DIR
)

print()

print(
    "Image-mask pairs found:",
    len(pairs)
)

print()

if len(pairs) == 0:

    raise RuntimeError(
        "No image-mask pairs were found."
    )


# ============================================================
# SHUFFLE DATASET
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

indices = np.arange(
    len(pairs)
)

rng.shuffle(
    indices
)

pairs = [
    pairs[i]
    for i in indices
]


# ============================================================
# TRAINING DATA
# ============================================================

X_list = []
y_list = []


# ============================================================
# PROCESS IMAGES
# ============================================================

for count, (
    image_path,
    mask_path
) in enumerate(
    pairs,
    start=1
):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(
            "Skipping image:",
            image_path
        )
        continue

    if mask is None:
        print(
            "Skipping mask:",
            mask_path
        )
        continue

    if image.shape != mask.shape:

        print(
            "Skipping size mismatch:",
            os.path.basename(image_path)
        )

        continue


    # --------------------------------------------------------
    # Extract classical features
    # --------------------------------------------------------

    features = extract_ml_features(
        image
    )


    # --------------------------------------------------------
    # Balanced pixel sampling
    # --------------------------------------------------------

    X, y = sample_pixels(
        features,
        mask,
        SAMPLES_PER_IMAGE
    )

    if X is None:
        continue


    X_list.append(
        X
    )

    y_list.append(
        y
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if count % 50 == 0:

        print(
            f"Processed {count}/{len(pairs)} images"
        )


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

if len(X_list) == 0:

    raise RuntimeError(
        "No training samples were generated."
    )


X_train = np.concatenate(
    X_list,
    axis=0
)

y_train = np.concatenate(
    y_list,
    axis=0
)


# ============================================================
# PRINT DATA INFORMATION
# ============================================================

print()
print("------------------------------------------------------")
print("TRAINING DATA")
print("------------------------------------------------------")

print(
    "Feature matrix:",
    X_train.shape
)

print(
    "Number of samples:",
    len(y_train)
)

print(
    "Crack samples:",
    int(np.sum(y_train == 1))
)

print(
    "Non-crack samples:",
    int(np.sum(y_train == 0))
)

print()


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

print("Creating Random Forest...")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)


# ============================================================
# TRAIN
# ============================================================

print("Training started...")

clf.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    clf,
    MODEL_PATH
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print()
print("------------------------------------------------------")
print("MODEL")
print("------------------------------------------------------")

print(
    "Trees:",
    N_ESTIMATORS
)

print(
    "Maximum depth:",
    MAX_DEPTH
)

print(
    "Feature count:",
    X_train.shape[1]
)

print()


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


print("Feature importance:")
print()

importance = clf.feature_importances_

sorted_indices = np.argsort(
    importance
)[::-1]


for index in sorted_indices:

    print(
        f"{feature_names[index]:20s}"
        f" {importance[index]:.4f}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("======================================================")
print("RANDOM FOREST TRAINING COMPLETED")
print("======================================================")
print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()
print(
    "Traditional ML model ready for testing."
)