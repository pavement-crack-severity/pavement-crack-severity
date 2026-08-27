import os
import csv
import cv2
import joblib
import numpy as np
import networkx as nx

from skimage.morphology import skeletonize

from ml_features import extract_ml_features

from geometric_features import (
    calculate_width_features,
    calculate_crack_coverage,
    calculate_crack_area,
    calculate_crack_orientation
)

from feature_extraction import (
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/CRACK500/traincrop"

MODEL_PATH = "models/crack_random_forest_v2.joblib"

OUTPUT_DIR = "outputs/rf_dataset"

OUTPUT_CSV = "outputs/features/rf_csi_dataset.csv"

# Start with 100 images.
# After confirming everything works, change to 200 or None.
MAX_IMAGES = None


# ============================================================
# CSI NORMALIZATION RANGES
# ============================================================

RANGES = {

    "crack_coverage_percent":
        (0.0, 20.0),

    "average_width_pixels":
        (0.0, 15.0),

    "maximum_width_pixels":
        (0.0, 30.0),

    "skeleton_length_pixels":
        (0.0, 3000.0),

    "branch_density":
        (0.0, 0.10),

    "endpoint_density":
        (0.0, 0.10),

    "branch_point_count":
        (0.0, 100.0),

    "connected_components":
        (1.0, 30.0)
}


# ============================================================
# CSI WEIGHTS
# ============================================================

WEIGHTS = {

    "crack_coverage_percent":
        0.30,

    "average_width_pixels":
        0.15,

    "maximum_width_pixels":
        0.10,

    "skeleton_length_pixels":
        0.15,

    "branch_density":
        0.10,

    "endpoint_density":
        0.05,

    "branch_point_count":
        0.10,

    "connected_components":
        0.05
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    value,
    minimum,
    maximum
):

    if maximum <= minimum:
        return 0.0

    value = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return max(
        0.0,
        min(
            1.0,
            value
        )
    )


# ============================================================
# SEVERITY
# ============================================================

def classify_severity(csi):

    if csi < 20:
        return "Low"

    elif csi < 40:
        return "Moderate"

    elif csi < 60:
        return "High"

    elif csi < 80:
        return "Very High"

    else:
        return "Severe"


# ============================================================
# FIND IMAGES
# ============================================================

def get_images():

    images = []

    for filename in sorted(
        os.listdir(DATA_DIR)
    ):

        if filename.lower().endswith(".jpg"):

            images.append(
                os.path.join(
                    DATA_DIR,
                    filename
                )
            )

    if MAX_IMAGES is not None:

        images = images[
            :MAX_IMAGES
        ]

    return images


# ============================================================
# MAIN
# ============================================================

print()
print("======================================================")
print("RF DATASET → FEATURES → CSI")
print("======================================================")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    "outputs/features",
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Random Forest V2...")

model_data = joblib.load(
    MODEL_PATH
)

if isinstance(
    model_data,
    dict
):

    model = model_data["model"]

    threshold = model_data["threshold"]

else:

    model = model_data

    threshold = 0.80


print(
    "Model loaded successfully."
)

print(
    f"RF threshold: {threshold:.2f}"
)


# ============================================================
# GET IMAGES
# ============================================================

images = get_images()

print()
print(
    f"Images to process: {len(images)}"
)


if len(images) == 0:

    raise RuntimeError(
        "No JPG images found."
    )


# ============================================================
# PROCESS IMAGES
# ============================================================

rows = []


for index, image_path in enumerate(
    images,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    image_name = os.path.splitext(
        filename
    )[0]


    print()
    print(
        f"[{index}/{len(images)}] "
        f"Processing {filename}"
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print(
            "WARNING: Could not load image."
        )

        continue


    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    features = extract_ml_features(
        image
    )

    h, w, feature_count = (
        features.shape
    )

    X = features.reshape(
        -1,
        feature_count
    )


    # --------------------------------------------------------
    # RF PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X
    )[:, 1]


    probability_map = probabilities.reshape(
        h,
        w
    )


    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    mask = (
        probability_map >= threshold
    )


    # --------------------------------------------------------
    # LIGHT CLEANING
    # --------------------------------------------------------

    mask_u8 = (
        mask.astype(
            np.uint8
        ) * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    mask = (
        mask_u8 > 0
    )


    # --------------------------------------------------------
    # GEOMETRIC FEATURES
    # --------------------------------------------------------

    crack_area = calculate_crack_area(
        mask_u8
    )

    coverage = calculate_crack_coverage(
        mask_u8
    )

    width_features = calculate_width_features(
        mask_u8
    )

    orientation = calculate_crack_orientation(
        mask_u8
    )


    # --------------------------------------------------------
    # SKELETON
    # --------------------------------------------------------

    skeleton = skeletonize(
        mask
    )


    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    endpoints = get_clustered_endpoints(
        skeleton,
        radius=3
    )

    branch_points = get_clustered_branch_points(
        skeleton,
        radius=3
    )

    graph = build_skeleton_graph(
        skeleton
    )


    connected_components = (
        nx.number_connected_components(
            graph
        )
        if graph.number_of_nodes() > 0
        else 0
    )


    skeleton_length = (
        calculate_skeleton_length(
            graph
        )
        if graph.number_of_nodes() > 0
        else 0.0
    )


    branch_density = (
        calculate_branch_density(
            len(branch_points),
            skeleton_length
        )
        if skeleton_length > 0
        else 0.0
    )


    endpoint_density = (
        calculate_endpoint_density(
            len(endpoints),
            skeleton_length
        )
        if skeleton_length > 0
        else 0.0
    )


    # --------------------------------------------------------
    # CSI NORMALIZATION
    # --------------------------------------------------------

    values = {

        "crack_coverage_percent":
            coverage,

        "average_width_pixels":
            width_features[
                "average_width_pixels"
            ],

        "maximum_width_pixels":
            width_features[
                "maximum_width_pixels"
            ],

        "skeleton_length_pixels":
            skeleton_length,

        "branch_density":
            branch_density,

        "endpoint_density":
            endpoint_density,

        "branch_point_count":
            len(branch_points),

        "connected_components":
            connected_components
    }


    normalized = {}

    for feature, value in values.items():

        minimum, maximum = (
            RANGES[feature]
        )

        normalized[feature] = normalize(
            value,
            minimum,
            maximum
        )


    # --------------------------------------------------------
    # CSI
    # --------------------------------------------------------

    weighted_sum = 0.0

    for feature, weight in WEIGHTS.items():

        weighted_sum += (
            normalized[feature]
            * weight
        )


    csi = (
        weighted_sum * 100
    )


    severity = classify_severity(
        csi
    )


    # --------------------------------------------------------
    # SAVE ROW
    # --------------------------------------------------------

    rows.append({

        "image":
            image_name,

        "crack_area_pixels":
            crack_area,

        "crack_coverage_percent":
            round(
                coverage,
                4
            ),

        "average_width_pixels":
            round(
                width_features[
                    "average_width_pixels"
                ],
                4
            ),

        "maximum_width_pixels":
            round(
                width_features[
                    "maximum_width_pixels"
                ],
                4
            ),

        "dominant_orientation_degrees":
            round(
                orientation,
                4
            ),

        "endpoint_count":
            len(endpoints),

        "branch_point_count":
            len(branch_points),

        "connected_components":
            connected_components,

        "graph_nodes":
            graph.number_of_nodes(),

        "graph_edges":
            graph.number_of_edges(),

        "skeleton_length_pixels":
            round(
                skeleton_length,
                4
            ),

        "branch_density":
            round(
                branch_density,
                6
            ),

        "endpoint_density":
            round(
                endpoint_density,
                6
            ),

        "CSI":
            round(
                csi,
                4
            ),

        "severity":
            severity
    })


    print(
        f"  Coverage: {coverage:.2f}%"
    )

    print(
        f"  Skeleton: {skeleton_length:.2f}px"
    )

    print(
        f"  CSI: {csi:.2f}"
    )

    print(
        f"  Severity: {severity}"
    )


# ============================================================
# SAVE CSV
# ============================================================

if len(rows) == 0:

    raise RuntimeError(
        "No images were successfully processed."
    )


fieldnames = list(
    rows[0].keys()
)


with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

csi_values = np.array(
    [
        row["CSI"]
        for row in rows
    ]
)


print()
print("======================================================")
print("DATASET PROCESSING COMPLETED")
print("======================================================")

print(
    f"Successfully processed: {len(rows)}"
)

print(
    f"Minimum CSI: {csi_values.min():.2f}"
)

print(
    f"Maximum CSI: {csi_values.max():.2f}"
)

print(
    f"Average CSI: {csi_values.mean():.2f}"
)

print()
print(
    "Output:"
)

print(
    OUTPUT_CSV
)

print()
print("======================================================")