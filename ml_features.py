import cv2
import numpy as np
from skimage.filters import frangi


def normalize_feature(image):
    """
    Normalize a feature to the range 0-1.
    """
    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return (
        (image - minimum)
        / (maximum - minimum)
    ).astype(np.float32)


def extract_ml_features(gray_image):
    """
    Extract classical image-processing features for
    Random Forest crack classification.

    No deep learning is used.

    Features:
        1. Grayscale intensity
        2. CLAHE intensity
        3. Gabor response
        4. Black-hat response
        5. Frangi ridge response
        6. Gradient magnitude
        7. Local mean
        8. Local standard deviation
        9. Local contrast

    Returns:
        feature_image -> H x W x 9
    """

    gray = gray_image.astype(np.uint8)

    # ========================================================
    # 1. Grayscale intensity
    # ========================================================

    intensity = normalize_feature(gray)


    # ========================================================
    # 2. CLAHE
    # ========================================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(gray)

    clahe_feature = normalize_feature(
        clahe_image
    )


    # ========================================================
    # 3. Gabor response
    # ========================================================

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
            gray.astype(np.float32),
            cv2.CV_32F,
            kernel
        )

        # Dark structures
        response = np.maximum(
            -response,
            0
        )

        gabor_responses.append(
            response
        )

    gabor_max = np.max(
        np.array(gabor_responses),
        axis=0
    )

    gabor_feature = normalize_feature(
        gabor_max
    )


    # ========================================================
    # 4. Black-hat response
    # ========================================================

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel
    )

    blackhat_feature = normalize_feature(
        blackhat
    )


    # ========================================================
    # 5. Frangi / Hessian ridge response
    # ========================================================

    frangi_response = frangi(
        gray.astype(np.float32) / 255.0,
        sigmas=(1, 2, 3, 4),
        alpha=0.5,
        beta=0.5,
        black_ridges=True
    )
    frangi_response = np.nan_to_num(frangi_response)
    frangi_feature = normalize_feature(
        frangi_response
    )


    # ========================================================
    # 6. Gradient magnitude
    # ========================================================

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    gradient_feature = normalize_feature(
        gradient
    )


    # ========================================================
    # 7. Local mean
    # ========================================================

    local_mean = cv2.GaussianBlur(
        gray.astype(np.float32),
        (9, 9),
        0
    )

    local_mean_feature = normalize_feature(
        local_mean
    )


    # ========================================================
    # 8. Local standard deviation
    # ========================================================

    gray_float = gray.astype(
        np.float32
    )

    mean_square = cv2.GaussianBlur(
        gray_float ** 2,
        (9, 9),
        0
    )

    square_mean = (
        local_mean ** 2
    )

    local_variance = np.maximum(
        mean_square - square_mean,
        0
    )

    local_std = np.sqrt(
        local_variance
    )

    local_std_feature = normalize_feature(
        local_std
    )


    # ========================================================
    # 9. Local contrast
    # ========================================================
    """"
    local_contrast = (
        np.abs(
            gray_float - local_mean
        )
    )

    local_contrast_feature = normalize_feature(
        local_contrast
    )
    """
    
    
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
        - local_min.astype(np.float32)
    )

    local_contrast_feature = normalize_feature(
        local_contrast_raw
    )
    
    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    feature_image = np.stack(
        [
            intensity,
            clahe_feature,
            gabor_feature,
            blackhat_feature,
            frangi_feature,
            gradient_feature,
            local_mean_feature,
            local_std_feature,
            local_contrast_feature
        ],
        axis=-1
    )

    return feature_image.astype(
        np.float32
    )


def feature_names():
    """
    Return feature names in the same
    order as extract_ml_features().
    """

    return [
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