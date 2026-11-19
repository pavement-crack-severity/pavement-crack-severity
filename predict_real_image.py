import os
import sys
import csv
import cv2
import joblib
import numpy as np

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

MODEL_PATH = "models/crack_random_forest_v2.joblib"

OUTPUT_DIR = "outputs/real_world"


# ============================================================
# FINAL CSI WEIGHTS
# ============================================================

WEIGHTS = {

    "crack_coverage_percent": 0.30,

    "skeleton_length_pixels": 0.25,

    "average_width_pixels": 0.15,

    "maximum_width_pixels": 0.10,

    "branch_point_count": 0.10,

    "branch_density": 0.05,

    "connected_components": 0.05
}


# ============================================================
# NORMALIZATION LIMITS
# ============================================================
#
# These are the P5/P95 values obtained from the full
# 1896-image CRACK500 feature distribution.
#

LIMITS = {

    "crack_coverage_percent":
        (0.5090, 16.7679),

    "skeleton_length_pixels":
        (186.6739, 5064.4534),

    "average_width_pixels":
        (3.3218, 10.8342),

    "maximum_width_pixels":
        (10.3938, 50.0829),

    "branch_point_count":
        (1.0, 118.25),

    "branch_density":
        (0.0056, 0.0326),

    "connected_components":
        (7.0, 259.0)
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value, low, high):

    if high <= low:
        return 0.0

    value = (
        value - low
    ) / (
        high - low
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
# GET IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print()
    print("Usage:")
    print(
        "python predict_real_image.py "
        "path_to_image.jpg"
    )
    print()

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


# ============================================================
# START
# ============================================================

print()
print("======================================================")
print("REAL-WORLD PAVEMENT CRACK ANALYSIS")
print("======================================================")


print()
print("Input image:")
print(IMAGE_PATH)


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

    model = model_data.get(
        "model",
        model_data
    )

    threshold = model_data.get(
        "threshold",
        0.80
    )
    

else:

    model = model_data

    threshold = 0.80


print(
    "Random Forest loaded successfully."
)

print(
    f"Threshold: {threshold:.2f}"
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)


if image is None:

    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


print()
print(
    "Image shape:",
    image.shape
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


base_name = os.path.splitext(
    os.path.basename(
        IMAGE_PATH
    )
)[0]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print()
print("Extracting classical image features...")


features = extract_ml_features(
    image
)


h, w, feature_count = (
    features.shape
)


print(
    "Feature shape:",
    features.shape
)


X = features.reshape(
    -1,
    feature_count
)


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

print()
print("Predicting crack probability...")


probabilities = model.predict_proba(
    X
)[:, 1]


probability_map = probabilities.reshape(
    h,
    w
)


# ============================================================
# SAVE PROBABILITY MAP
# ============================================================

probability_output = (
    f"{OUTPUT_DIR}/"
    f"{base_name}_probability.png"
)


probability_image = (
    probability_map
    * 255
).astype(
    np.uint8
)


cv2.imwrite(
    probability_output,
    probability_image
)


# ============================================================
# BINARY MASK
# ============================================================

mask = (
    probability_map >= 0.65
)


mask_u8 = (
    mask.astype(
        np.uint8
    ) * 255
)


# ============================================================
# LIGHT MORPHOLOGICAL CLEANING
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

""""
mask_u8 = cv2.morphologyEx(
    mask_u8,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)
"""

mask_u8 = cv2.morphologyEx(
    mask_u8,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


mask = (
    mask_u8 > 0
)


# ============================================================
# SAVE MASK
# ============================================================

mask_output = (
    f"{OUTPUT_DIR}/"
    f"{base_name}_mask.png"
)


cv2.imwrite(
    mask_output,
    mask_u8
)


# ============================================================
# GEOMETRIC FEATURES
# ============================================================

print()
print("Calculating geometric features...")


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


# ============================================================
# SKELETON
# ============================================================

print()
print("Skeletonizing...")


skeleton = skeletonize(
    mask
)


skeleton_u8 = (
    skeleton.astype(
        np.uint8
    ) * 255
)


skeleton_output = (
    f"{OUTPUT_DIR}/"
    f"{base_name}_skeleton.png"
)


cv2.imwrite(
    skeleton_output,
    skeleton_u8
)


# ============================================================
# GRAPH FEATURES
# ============================================================

print()
print("Calculating graph features...")


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

"""
if graph.number_of_nodes() > 0:

    connected_components = (
        graph.number_of_nodes()
    )

    import networkx as nx

    connected_components = (
        nx.number_connected_components(
            graph
        )
    )

    skeleton_length = (
        calculate_skeleton_length(
            graph
        )
    )

else:

    connected_components = 0

    skeleton_length = 0.0
"""
import networkx as nx

if graph.number_of_nodes() > 0:

    connected_components = (
        nx.number_connected_components(graph)
    )

    skeleton_length = (
        calculate_skeleton_length(graph)
    )

else:

    connected_components = 0
    skeleton_length = 0.0

if skeleton_length > 0:

    branch_density = (
        calculate_branch_density(
            len(branch_points),
            skeleton_length
        )
    )

    endpoint_density = (
        calculate_endpoint_density(
            len(endpoints),
            skeleton_length
        )
    )

else:

    branch_density = 0.0

    endpoint_density = 0.0


# ============================================================
# FEATURE DICTIONARY
# ============================================================

values = {

    "crack_coverage_percent":
        coverage,

    "skeleton_length_pixels":
        skeleton_length,

    "average_width_pixels":
        width_features[
            "average_width_pixels"
        ],

    "maximum_width_pixels":
        width_features[
            "maximum_width_pixels"
        ],

    "branch_point_count":
        len(branch_points),

    "branch_density":
        branch_density,

    "connected_components":
        connected_components
}


# ============================================================
# NORMALIZE + CSI
# ============================================================

weighted_sum = 0.0


for feature, weight in WEIGHTS.items():

    value = values[
        feature
    ]

    low, high = LIMITS[
        feature
    ]

    normalized = normalize(
        value,
        low,
        high
    )

    weighted_sum += (
        normalized
        * weight
    )


csi = (
    weighted_sum
    * 100
)


severity = classify_severity(
    csi
)


# ============================================================
# SAVE CSV
# ============================================================

csv_output = (
    f"{OUTPUT_DIR}/"
    f"{base_name}_result.csv"
)


result = {

    "image":
        os.path.basename(
            IMAGE_PATH
        ),

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
            2
        ),

    "severity":
        severity
}


with open(
    csv_output,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            result.keys()
        )
    )

    writer.writeheader()

    writer.writerow(
        result
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("======================================================")
print("FINAL RESULT")
print("======================================================")


print()
print(
    f"Crack area: "
    f"{crack_area} pixels"
)


print(
    f"Crack coverage: "
    f"{coverage:.2f}%"
)


print(
    f"Average width: "
    f"{width_features['average_width_pixels']:.2f}px"
)


print(
    f"Maximum width: "
    f"{width_features['maximum_width_pixels']:.2f}px"
)


print(
    f"Skeleton length: "
    f"{skeleton_length:.2f}px"
)


print(
    f"Branch points: "
    f"{len(branch_points)}"
)


print(
    f"Connected components: "
    f"{connected_components}"
)


print()
print(
    f"CSI: {csi:.2f} / 100"
)


print(
    f"Severity: {severity}"
)


print()
print("------------------------------------------------------")
print("OUTPUT FILES")
print("------------------------------------------------------")


print(
    "Probability:",
    probability_output
)


print(
    "Crack mask:",
    mask_output
)


print(
    "Skeleton:",
    skeleton_output
)


print(
    "CSV:",
    csv_output
)


print()
print("======================================================")
print("REAL-WORLD ANALYSIS COMPLETED")
print("======================================================")