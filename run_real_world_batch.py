import os
import sys
import csv
import subprocess
import glob


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/real_world"
OUTPUT_DIR = "outputs/real_world"

PREDICT_SCRIPT = "predict_real_image.py"


# ============================================================
# FIND IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

images = []

for ext in extensions:
    images.extend(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                ext
            )
        )
    )

images = sorted(
    set(images)
)


if len(images) == 0:

    print("No images found in:")
    print(INPUT_DIR)

    sys.exit(1)


# ============================================================
# START
# ============================================================

print()
print("======================================================")
print("REAL-WORLD BATCH CRACK ANALYSIS")
print("======================================================")

print()
print(
    f"Images found: {len(images)}"
)

print(
    f"Input folder: {INPUT_DIR}"
)

print()


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

successful = 0
failed = 0

for index, image_path in enumerate(
    images,
    start=1
):

    print()
    print("------------------------------------------------------")

    print(
        f"[{index}/{len(images)}] "
        f"{os.path.basename(image_path)}"
    )

    print("------------------------------------------------------")

    try:

        result = subprocess.run(
            [
                sys.executable,
                PREDICT_SCRIPT,
                image_path
            ],
            check=True
        )

        successful += 1

    except subprocess.CalledProcessError:

        print()
        print(
            "ERROR processing:",
            image_path
        )

        failed += 1


# ============================================================
# COMBINE RESULT CSV FILES
# ============================================================

print()
print("======================================================")
print("COMBINING RESULTS")
print("======================================================")


result_files = glob.glob(
    os.path.join(
        OUTPUT_DIR,
        "*_result.csv"
    )
)


combined_rows = []


for csv_file in result_files:

    try:

        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                combined_rows.append(row)

    except Exception as e:

        print(
            "Could not read:",
            csv_file,
            "|",
            e
        )


# ============================================================
# SAVE COMBINED CSV
# ============================================================

combined_output = os.path.join(
    OUTPUT_DIR,
    "real_world_combined_results.csv"
)


if combined_rows:

    fieldnames = list(
        combined_rows[0].keys()
    )

    with open(
        combined_output,
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
            combined_rows
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("======================================================")
print("BATCH PROCESSING COMPLETED")
print("======================================================")

print()
print(
    f"Total images:      {len(images)}"
)

print(
    f"Successfully done: {successful}"
)

print(
    f"Failed:            {failed}"
)

print(
    f"Results collected: {len(combined_rows)}"
)

print()
print(
    "Combined CSV:"
)

print(
    combined_output
)

print()
print("======================================================")