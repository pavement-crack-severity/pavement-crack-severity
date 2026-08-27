import csv
import os
import numpy as np
from collections import Counter


# ============================================================
# FILES
# ============================================================

INPUT_PATH = "outputs/features/rf_csi_dataset.csv"

OUTPUT_PATH = "outputs/features/rf_final_csi_dataset.csv"


# ============================================================
# FEATURES USED IN FINAL CSI
# ============================================================
#
# Endpoint density is intentionally excluded because its
# correlation with the preliminary CSI was negative.
#
# Total weight = 1.00
#

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
# PERCENTILE LIMITS
# ============================================================
#
# We use the 5th and 95th percentiles rather than minimum and
# maximum so that a few extreme images do not dominate the
# normalization.
#
# Values below P5 -> 0
# Values above P95 -> 1
#

LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95


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
# LOAD DATA
# ============================================================

print()
print("======================================================")
print("FINAL RF-BASED CRACK SEVERITY INDEX")
print("======================================================")

print()
print("Loading:")
print(INPUT_PATH)


with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


if len(rows) == 0:

    raise RuntimeError(
        "No rows found in input CSV."
    )


print(
    f"Images loaded: {len(rows)}"
)


# ============================================================
# EXTRACT FEATURE ARRAYS
# ============================================================

feature_values = {}


for feature in WEIGHTS:

    values = []

    for row in rows:

        try:

            value = float(
                row[feature]
            )

            if np.isfinite(value):

                values.append(value)

        except:

            pass


    if len(values) != len(rows):

        raise RuntimeError(
            f"Invalid/missing values in feature: {feature}"
        )


    feature_values[feature] = np.array(
        values,
        dtype=np.float64
    )


# ============================================================
# CALCULATE PERCENTILE LIMITS
# ============================================================

print()
print("------------------------------------------------------")
print("ROBUST NORMALIZATION LIMITS")
print("------------------------------------------------------")


limits = {}


for feature in WEIGHTS:

    values = feature_values[
        feature
    ]


    low = np.percentile(
        values,
        LOW_PERCENTILE
    )


    high = np.percentile(
        values,
        HIGH_PERCENTILE
    )


    limits[feature] = (
        low,
        high
    )


    print()
    print(feature)

    print(
        f"  P{LOW_PERCENTILE}:  {low:.4f}"
    )

    print(
        f"  P{HIGH_PERCENTILE}: {high:.4f}"
    )


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(
    value,
    low,
    high
):

    if high <= low:

        return 0.0


    normalized = (
        value - low
    ) / (
        high - low
    )


    # Clamp to [0, 1]

    normalized = max(
        0.0,
        min(
            1.0,
            normalized
        )
    )


    return normalized


# ============================================================
# CALCULATE FINAL CSI
# ============================================================

print()
print("------------------------------------------------------")
print("CALCULATING FINAL CSI")
print("------------------------------------------------------")


results = []


for index, row in enumerate(
    rows,
    start=1
):

    image_name = row[
        "image"
    ]


    weighted_sum = 0.0


    # --------------------------------------------------------
    # Normalize + weighted sum
    # --------------------------------------------------------

    for feature, weight in WEIGHTS.items():

        value = float(
            row[feature]
        )


        low, high = limits[
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


    # --------------------------------------------------------
    # Convert to 0–100
    # --------------------------------------------------------

    csi = (
        weighted_sum
        * 100.0
    )


    severity = classify_severity(
        csi
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    result = {

        "image":
            image_name,

        "crack_area_pixels":
            row[
                "crack_area_pixels"
            ],

        "crack_coverage_percent":
            row[
                "crack_coverage_percent"
            ],

        "average_width_pixels":
            row[
                "average_width_pixels"
            ],

        "maximum_width_pixels":
            row[
                "maximum_width_pixels"
            ],

        "dominant_orientation_degrees":
            row[
                "dominant_orientation_degrees"
            ],

        "endpoint_count":
            row[
                "endpoint_count"
            ],

        "branch_point_count":
            row[
                "branch_point_count"
            ],

        "connected_components":
            row[
                "connected_components"
            ],

        "graph_nodes":
            row[
                "graph_nodes"
            ],

        "graph_edges":
            row[
                "graph_edges"
            ],

        "skeleton_length_pixels":
            row[
                "skeleton_length_pixels"
            ],

        "branch_density":
            row[
                "branch_density"
            ],

        "endpoint_density":
            row[
                "endpoint_density"
            ],

        "final_CSI":
            round(
                csi,
                4
            ),

        "severity":
            severity
    }


    results.append(
        result
    )


    # Progress

    if (
        index % 100 == 0
        or index == len(rows)
    ):

        print(
            f"Processed: "
            f"{index}/{len(rows)}"
        )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs/features",
    exist_ok=True
)


fieldnames = list(
    results[0].keys()
)


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# CSI STATISTICS
# ============================================================

csi_values = np.array(
    [
        float(
            row["final_CSI"]
        )
        for row in results
    ]
)


print()
print("======================================================")
print("FINAL CSI STATISTICS")
print("======================================================")


print(
    f"Minimum CSI: "
    f"{np.min(csi_values):.2f}"
)


print(
    f"Maximum CSI: "
    f"{np.max(csi_values):.2f}"
)


print(
    f"Mean CSI: "
    f"{np.mean(csi_values):.2f}"
)


print(
    f"Median CSI: "
    f"{np.median(csi_values):.2f}"
)


print(
    f"Std Dev: "
    f"{np.std(csi_values):.2f}"
)


# ============================================================
# PERCENTILES
# ============================================================

print()
print("------------------------------------------------------")
print("FINAL CSI PERCENTILES")
print("------------------------------------------------------")


for p in [
    10,
    25,
    50,
    75,
    90,
    95
]:

    value = np.percentile(
        csi_values,
        p
    )


    print(
        f"{p}th percentile: "
        f"{value:.2f}"
    )


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

print()
print("------------------------------------------------------")
print("FINAL SEVERITY DISTRIBUTION")
print("------------------------------------------------------")


severity_counts = Counter(
    row["severity"]
    for row in results
)


severity_order = [
    "Low",
    "Moderate",
    "High",
    "Very High",
    "Severe"
]


for category in severity_order:

    count = severity_counts.get(
        category,
        0
    )


    percentage = (
        count
        / len(results)
        * 100
    )


    print(
        f"{category:<12}"
        f"{count:>5} images "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# HIGHEST CSI
# ============================================================

print()
print("------------------------------------------------------")
print("TOP 10 HIGHEST CSI")
print("------------------------------------------------------")


sorted_results = sorted(
    results,
    key=lambda row:
        float(row["final_CSI"]),
    reverse=True
)


for row in sorted_results[:10]:

    print(
        f"{row['image']:<40}"
        f"CSI={float(row['final_CSI']):6.2f} "
        f"Coverage="
        f"{float(row['crack_coverage_percent']):6.2f}%"
    )


# ============================================================
# LOWEST CSI
# ============================================================

print()
print("------------------------------------------------------")
print("10 LOWEST CSI")
print("------------------------------------------------------")


for row in sorted_results[-10:][::-1]:

    print(
        f"{row['image']:<40}"
        f"CSI={float(row['final_CSI']):6.2f} "
        f"Coverage="
        f"{float(row['crack_coverage_percent']):6.2f}%"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("======================================================")
print("FINAL RF CSI CALCULATION COMPLETED")
print("======================================================")

print()
print("Output:")
print(OUTPUT_PATH)

print()
print("CSI formula uses:")
print("  Coverage           = 30%")
print("  Skeleton length    = 25%")
print("  Average width      = 15%")
print("  Maximum width      = 10%")
print("  Branch points      = 10%")
print("  Branch density     = 5%")
print("  Components         = 5%")
print()
print("Endpoint density is excluded.")
print()