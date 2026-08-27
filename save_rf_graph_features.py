import csv
import os
import networkx as nx

from feature_extraction import (
    load_skeleton,
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)


# ============================================================
# RF SKELETON INPUTS
# ============================================================

SKELETONS = [
    "crack2_rf_skeleton.png"
]

INPUT_DIR = "outputs/ml_v2"

OUTPUT_DIR = "outputs/features"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "rf_graph_features.csv"
)


# Create output directory if needed
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


rows = []


# ============================================================
# PROCESS EACH SKELETON
# ============================================================

for filename in SKELETONS:

    path = os.path.join(
        INPUT_DIR,
        filename
    )

    print()
    print("Processing:", filename)

    if not os.path.exists(path):
        print("WARNING: File not found:", path)
        continue

    # --------------------------------------------------------
    # Load skeleton
    # --------------------------------------------------------

    skeleton = load_skeleton(path)

    print(
        "Skeleton pixels:",
        skeleton.sum()
    )


    # --------------------------------------------------------
    # Detect endpoints
    # --------------------------------------------------------

    endpoints = get_clustered_endpoints(
        skeleton,
        radius=3
    )


    # --------------------------------------------------------
    # Detect branch points
    # --------------------------------------------------------

    branch_points = get_clustered_branch_points(
        skeleton,
        radius=3
    )


    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    graph = build_skeleton_graph(
        skeleton
    )


    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    connected_components = (
        nx.number_connected_components(
            graph
        )
    )


    # --------------------------------------------------------
    # Skeleton length
    # --------------------------------------------------------

    skeleton_length = (
        calculate_skeleton_length(
            graph
        )
    )


    # --------------------------------------------------------
    # Densities
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    image_name = filename.replace(
        "_rf_skeleton.png",
        ""
    )


    rows.append({

        "image":
            image_name,

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
                2
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
    })


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [

    "image",

    "endpoint_count",

    "branch_point_count",

    "connected_components",

    "graph_nodes",

    "graph_edges",

    "skeleton_length_pixels",

    "branch_density",

    "endpoint_density"
]


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

    writer.writerows(rows)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("======================================================")
print("RF GRAPH FEATURE EXTRACTION")
print("======================================================")

for row in rows:

    print()
    print("Image:", row["image"])

    print(
        "Endpoint count:",
        row["endpoint_count"]
    )

    print(
        "Branch point count:",
        row["branch_point_count"]
    )

    print(
        "Connected components:",
        row["connected_components"]
    )

    print(
        "Graph nodes:",
        row["graph_nodes"]
    )

    print(
        "Graph edges:",
        row["graph_edges"]
    )

    print(
        "Skeleton length:",
        row["skeleton_length_pixels"]
    )

    print(
        "Branch density:",
        row["branch_density"]
    )

    print(
        "Endpoint density:",
        row["endpoint_density"]
    )


print()
print("======================================================")
print("RF GRAPH FEATURES SAVED SUCCESSFULLY")
print("======================================================")

print(
    "Output:",
    OUTPUT_PATH
)