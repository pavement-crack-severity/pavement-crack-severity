import csv
import os
import cv2

from geometric_features import (
    load_crack_mask,
    calculate_width_features,
    calculate_crack_coverage,
    calculate_crack_area,
    calculate_crack_orientation
)

from feature_extraction import (
    load_skeleton,
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)

import networkx as nx


# ============================================================
# INPUTS
# ============================================================

MASK_PATH = "outputs/ml_v2/crack2_rf_mask_v2.png"

SKELETON_PATH = "outputs/ml_v2/crack2_rf_skeleton.png"

OUTPUT_DIR = "outputs/features"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_combined_features.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MASK
# ============================================================

print("Loading RF mask...")

mask = load_crack_mask(
    MASK_PATH
)

print(
    "Mask shape:",
    mask.shape
)


# ============================================================
# GEOMETRIC FEATURES
# ============================================================

print()
print("Calculating geometric features...")

width_features = calculate_width_features(
    mask
)

coverage = calculate_crack_coverage(
    mask
)

area = calculate_crack_area(
    mask
)

orientation = calculate_crack_orientation(
    mask
)


# ============================================================
# LOAD SKELETON
# ============================================================

print()
print("Loading RF skeleton...")

skeleton = load_skeleton(
    SKELETON_PATH
)

print(
    "Skeleton pixels:",
    skeleton.sum()
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


# ============================================================
# COMBINE ALL FEATURES
# ============================================================

row = {

    "image":
        "crack2",

    # -----------------------------
    # Geometric
    # -----------------------------

    "crack_area_pixels":
        area,

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

    # -----------------------------
    # Graph
    # -----------------------------

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
        )
}


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = list(
    row.keys()
)

with open(
    OUTPUT_PATH,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerow(
        row
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("======================================================")
print("RF COMBINED FEATURE EXTRACTION")
print("======================================================")

for key, value in row.items():

    print(
        f"{key}: {value}"
    )

print()
print("======================================================")
print("COMBINED FEATURES SAVED SUCCESSFULLY")
print("======================================================")

print(
    "Output:",
    OUTPUT_PATH
)