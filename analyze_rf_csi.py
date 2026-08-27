import csv
import numpy as np
from collections import Counter


INPUT_PATH = "outputs/features/rf_csi_dataset.csv"


print()
print("======================================================")
print("RF CSI DATASET ANALYSIS V2")
print("======================================================")


# ============================================================
# LOAD
# ============================================================

with open(INPUT_PATH, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)
    rows = list(reader)


print()
print("Images analyzed:", len(rows))


if not rows:
    raise RuntimeError("CSV contains no data.")


# ============================================================
# HELPER
# ============================================================

def get_values(column):

    values = []

    for row in rows:

        try:
            value = float(row[column])

            if np.isfinite(value):
                values.append(value)

        except:
            pass

    return np.array(values, dtype=float)


def print_stats(name, values):

    if len(values) == 0:

        print(
            f"{name}: NO VALID DATA"
        )

        return

    print()
    print(name)

    print(
        f"  Count:  {len(values)}"
    )

    print(
        f"  Min:    {np.min(values):.4f}"
    )

    print(
        f"  Max:    {np.max(values):.4f}"
    )

    print(
        f"  Mean:   {np.mean(values):.4f}"
    )

    print(
        f"  Median: {np.median(values):.4f}"
    )


# ============================================================
# CSI
# ============================================================

csi = get_values("CSI")


print()
print("------------------------------------------------------")
print("CSI STATISTICS")
print("------------------------------------------------------")

print_stats(
    "CSI",
    csi
)


# ============================================================
# PERCENTILES
# ============================================================

print()
print("------------------------------------------------------")
print("CSI PERCENTILES")
print("------------------------------------------------------")

if len(csi) > 0:

    for p in [10, 25, 50, 75, 90, 95]:

        print(
            f"{p}th percentile: "
            f"{np.percentile(csi, p):.2f}"
        )


# ============================================================
# SEVERITY
# ============================================================

print()
print("------------------------------------------------------")
print("SEVERITY DISTRIBUTION")
print("------------------------------------------------------")

severity_values = [
    row.get("severity", "")
    for row in rows
]

counts = Counter(
    severity_values
)

for category in [
    "Low",
    "Moderate",
    "High",
    "Very High",
    "Severe"
]:

    count = counts.get(
        category,
        0
    )

    percentage = (
        count / len(rows)
    ) * 100

    print(
        f"{category:<12}"
        f"{count:>4} images "
        f"({percentage:.1f}%)"
    )


# ============================================================
# FEATURES
# ============================================================

print()
print("------------------------------------------------------")
print("FEATURE DISTRIBUTIONS")
print("------------------------------------------------------")


features = {

    "Coverage (%)":
        "crack_coverage_percent",

    "Average width":
        "average_width_pixels",

    "Maximum width":
        "maximum_width_pixels",

    "Skeleton length":
        "skeleton_length_pixels",

    "Branch density":
        "branch_density",

    "Endpoint density":
        "endpoint_density",

    "Branch points":
        "branch_point_count",

    "Connected components":
        "connected_components"
}


feature_arrays = {}


for name, column in features.items():

    values = get_values(column)

    feature_arrays[name] = values

    print_stats(
        name,
        values
    )


# ============================================================
# CORRELATION
# ============================================================

print()
print("------------------------------------------------------")
print("CORRELATION WITH CSI")
print("------------------------------------------------------")


for name, values in feature_arrays.items():

    if len(values) != len(csi):

        print(
            f"{name:<22} insufficient data"
        )

        continue

    if (
        len(values) < 2
        or np.std(values) == 0
        or np.std(csi) == 0
    ):

        print(
            f"{name:<22} unavailable"
        )

        continue

    correlation = np.corrcoef(
        values,
        csi
    )[0, 1]

    print(
        f"{name:<22}"
        f"{correlation:+.4f}"
    )


# ============================================================
# CSI RANGES
# ============================================================

print()
print("------------------------------------------------------")
print("CSI RANGE COUNTS")
print("------------------------------------------------------")


ranges = [

    ("0 - 10", 0, 10),

    ("10 - 20", 10, 20),

    ("20 - 30", 20, 30),

    ("30 - 40", 30, 40),

    ("40 - 50", 40, 50),

    ("50 - 60", 50, 60),

    ("60 - 70", 60, 70),

    ("70 - 80", 70, 80),

    ("80 - 100", 80, 101)
]


for name, low, high in ranges:

    count = np.sum(
        (csi >= low)
        &
        (csi < high)
    )

    print(
        f"{name:<10}"
        f"{count:>4} images"
    )


# ============================================================
# TOP 10
# ============================================================

print()
print("------------------------------------------------------")
print("TOP 10 HIGHEST CSI")
print("------------------------------------------------------")


sorted_rows = sorted(
    rows,
    key=lambda row: float(row["CSI"]),
    reverse=True
)


for row in sorted_rows[:10]:

    print(
        f"{row['image']:<40}"
        f"CSI={float(row['CSI']):6.2f} "
        f"Coverage="
        f"{float(row['crack_coverage_percent']):6.2f}%"
    )


# ============================================================
# LOWEST 10
# ============================================================

print()
print("------------------------------------------------------")
print("10 LOWEST CSI")
print("------------------------------------------------------")


for row in sorted_rows[-10:][::-1]:

    print(
        f"{row['image']:<40}"
        f"CSI={float(row['CSI']):6.2f} "
        f"Coverage="
        f"{float(row['crack_coverage_percent']):6.2f}%"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("======================================================")
print("CSI ANALYSIS V2 COMPLETED")
print("======================================================")