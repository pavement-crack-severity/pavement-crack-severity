import cv2
import numpy as np


# ============================================================
# 1. OTSU THRESHOLDING
# ============================================================

def apply_otsu_threshold(gabor_image):
    """
    Jassi — Otsu thresholding.

    Provides a global thresholding baseline for the
    enhanced crack-response image.
    """

    blurred = cv2.GaussianBlur(
        gabor_image,
        (5, 5),
        0
    )

    _, binary_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_mask


# ============================================================
# 2. STRONG RESPONSE THRESHOLD
# ============================================================

def apply_strong_response_threshold(
    gabor_image,
    percentile=85
):
    """
    Jassi — Strong-response percentile thresholding.

    Keeps pixels whose response is within the strongest
    percentage of the enhanced response image.

    This function is retained for compatibility with the
    existing project pipeline.
    """

    blurred = cv2.GaussianBlur(
        gabor_image,
        (5, 5),
        0
    )

    threshold_value = np.percentile(
        blurred,
        percentile
    )

    binary_mask = np.where(
        blurred >= threshold_value,
        255,
        0
    ).astype(np.uint8)

    return binary_mask


# ============================================================
# 3. ADAPTIVE RESPONSE THRESHOLD
# ============================================================

def apply_adaptive_response_threshold(
    gabor_image,
    block_size=31,
    C=-2,
    percentile=None
):
    """
    Jassi — Adaptive local thresholding.

    Instead of using one threshold for the complete image,
    adaptive thresholding calculates a local threshold for
    each neighborhood.

    This is useful for pavement images because illumination,
    pavement texture and crack contrast can vary across
    different regions.

    percentile is accepted for compatibility with earlier
    versions but is not required by this method.
    """

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd number greater than 1."
        )

    blurred = cv2.GaussianBlur(
        gabor_image,
        (5, 5),
        0
    )

    binary_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return binary_mask


# ============================================================
# 4. MORPHOLOGICAL CLEANING
# ============================================================

def clean_binary_mask(binary_mask):
    """
    Jassi — Morphological cleaning.

    Opening:
        Removes small isolated noise.

    Closing:
        Fills small gaps and helps reconnect nearby crack
        regions.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    # Opening removes small isolated regions.
    opened = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Closing reconnects small gaps.
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return closed


# ============================================================
# 5. REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    binary_mask,
    min_area=200
):
    """
    Remove disconnected components smaller than min_area.

    This helps suppress small pavement-texture regions
    and isolated noise.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(
        binary_mask
    )

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            filtered_mask[
                labels == label
            ] = 255

    return filtered_mask


# ============================================================
# 6. CRACK-LIKE COMPONENT FILTERING
# ============================================================

def filter_crack_like_components(
    binary_mask,
    min_area=200,
    min_elongation=2.0
):
    """
    Jassi — Shape-based crack filtering.

    Retains elongated/irregular structures and suppresses
    compact regions that are more likely to be pavement
    texture or noise.

    min_elongation is retained for compatibility with the
    existing project API.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8
        )
    )

    filtered_mask = np.zeros_like(
        binary_mask
    )

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        component = (
            np.uint8(labels == label) * 255
        )

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        # Compactness:
        #
        # A compact circular/blobby region tends to have
        # higher compactness.
        #
        # An elongated crack-like region tends to have
        # lower compactness.

        compactness = (
            4.0 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        if compactness <= 0.65:

            filtered_mask[
                labels == label
            ] = 255

    return filtered_mask


# ============================================================
# 7. MULTI-SCALE BLACK-HAT RESPONSE
# ============================================================

def create_multiscale_blackhat_response(
    gray_image,
    scales=(7, 13, 19, 25)
):
    """
    Jassi — Multi-scale morphological crack enhancement.

    Applies black-hat morphology at several spatial scales.

    Smaller kernels:
        Better for relatively thin crack structures.

    Larger kernels:
        Better for broader crack structures.

    The strongest response at every pixel is retained.
    """

    responses = []

    for scale in scales:

        if scale <= 1 or scale % 2 == 0:
            raise ValueError(
                "Every scale must be an odd number greater than 1."
            )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (scale, scale)
        )

        blackhat = cv2.morphologyEx(
            gray_image,
            cv2.MORPH_BLACKHAT,
            kernel
        )

        normalized = cv2.normalize(
            blackhat,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        responses.append(
            normalized.astype(np.float32)
        )

    stacked_responses = np.stack(
        responses,
        axis=0
    )

    # Keep strongest response at each pixel.
    fused_response = np.max(
        stacked_responses,
        axis=0
    )

    fused_response = cv2.normalize(
        fused_response,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return fused_response.astype(np.uint8)