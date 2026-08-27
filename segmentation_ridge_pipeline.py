"""
segmentation_ridge_pipeline.py
--------------------------------
NEW segmentation stage for the pavement crack severity project.

Purpose
=======
Replaces / supplements the Gabor + black-hat fusion (which produced ~42-44%
predicted crack coverage against a ~5.56% ground truth, i.e. massive false
positives from pavement texture) with a ridge/vesselness-based detector that
is selective for thin, elongated, curvilinear structures -- which is what
cracks actually are, and what blob-like aggregate texture is NOT.

Pipeline implemented here (Approach 1 + 3 + shape filtering + optional
Approach 5 from the analysis):

    grayscale + CLAHE (existing)
        -> bilateral denoise                          [NEW]
        -> multiscale Hessian vesselness (Frangi)      [NEW]  (classical DIP)
        -> local robust adaptive threshold + hysteresis[NEW]  (classical DIP)
        -> opening-by-reconstruction                   [NEW]  (classical DIP)
        -> shape / continuity filtering                (reworked)
        -> OPTIONAL Random Forest false-positive filter[NEW]  (traditional ML)
        -> final binary crack mask

Every stage is classical DIP or traditional ML. No deep learning, no CNNs,
no pretrained neural models are used anywhere in this file.

Drop these functions into your existing segmentation.py (they are written
to be independent of your current file so you can add them without
disturbing apply_otsu_threshold / apply_adaptive_response_threshold / etc.,
which you can keep around for the baseline comparison in your report).
"""

import numpy as np
import cv2

from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.morphology import (
    reconstruction, remove_small_objects, skeletonize, disk
)
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# STAGE 1: multiscale Hessian vesselness (Frangi filter)
# ---------------------------------------------------------------------------
def compute_ridge_response(gray_image, scales=(1, 2, 3, 4),
                            beta=0.5, denoise=True):
    """
    Multiscale Hessian-eigenvalue-based ridge (vesselness) response.

    Cracks are dark, thin, curvilinear structures on a textured background.
    At a ridge pixel, the local Hessian has one large-magnitude eigenvalue
    (curvature across the ridge) and one near-zero eigenvalue (curvature
    along the ridge). At a blob/texture pixel, both eigenvalues are large.
    This is the classical Frangi (1998) vesselness measure, reused here for
    crack ridge enhancement instead of blood vessels -- pure math, no
    learning, no training data required.

    Parameters
    ----------
    gray_image : uint8 grayscale image (after CLAHE)
    scales     : tuple of Gaussian sigmas (in pixels) spanning your expected
                 crack width range. Measure a handful of CRACK500 ground
                 truth masks (average skeleton-normal width) to set this;
                 (1,2,3,4) is a reasonable starting range for CRACK500's
                 typically 2-8 px wide cracks.
    beta       : Frangi shape parameter (blob vs. line discrimination).
                 0.5 is the standard default; rarely needs tuning.
    denoise    : apply an edge-preserving bilateral filter before the
                 ridge filter, to suppress salt-and-pepper texture noise
                 without blurring the crack ridges themselves.

    Returns
    -------
    response : float32 array in [0, 1], higher = more ridge-like (crack-like)
    """
    img = gray_image.astype(np.float32)

    if denoise:
        img = cv2.bilateralFilter(img.astype(np.uint8), d=5,
                                   sigmaColor=25, sigmaSpace=5).astype(np.float32)

    # Cracks are DARKER than surrounding pavement -> invert so ridges become
    # bright peaks, since skimage's frangi() looks for bright ridges by
    # default with black_ridges=True meaning it looks for dark ridges on the
    # ORIGINAL image; we pass black_ridges=True directly instead of inverting
    # manually, which is numerically cleaner.
    img_norm = img / 255.0

    response = frangi(
        img_norm,
        sigmas=scales,
        alpha=0.5,
        beta=beta,
        black_ridges=True,   # cracks are dark ridges on a lighter background
    )

    # Normalize to [0, 1] for consistent downstream thresholding.
    if response.max() > 0:
        response = response / response.max()

    return response.astype(np.float32)


# ---------------------------------------------------------------------------
# STAGE 2: local robust adaptive threshold + hysteresis
# ---------------------------------------------------------------------------
def adaptive_hysteresis_threshold(response, tile_size=32, k_high=3.0, k_low=1.5):
    """
    Local, robust (median/MAD-based) hysteresis thresholding of the ridge
    response map. This fixes the core precision problem: a single global
    (or naively adaptive) cut on the Gabor/black-hat response cannot
    separate crack ridges from texture, because pavement brightness and
    texture strength vary tile-to-tile. Using the MEDIAN and MAD (median
    absolute deviation) instead of mean/std makes the per-tile statistic
    robust to the very texture outliers we are trying to reject.

    Hysteresis (strong seeds + weak-but-connected growth) additionally
    enforces continuity: an isolated pixel that only clears the LOW
    threshold, with no connection to any pixel above the HIGH threshold,
    is discarded -- this is what naive single-threshold methods can't do,
    and is a major source of your current false positives.

    Parameters
    ----------
    response  : float32 ridge response in [0,1] from compute_ridge_response
    tile_size : size of local tiles (pixels) for robust statistics
    k_high    : strong-seed threshold = median + k_high * 1.4826 * MAD
    k_low     : weak-growth threshold = median + k_low  * 1.4826 * MAD

    Returns
    -------
    binary mask (bool array), same shape as response
    """
    h, w = response.shape
    high_map = np.zeros_like(response)
    low_map = np.zeros_like(response)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = response[y:y + tile_size, x:x + tile_size]
            if tile.size == 0:
                continue
            med = np.median(tile)
            mad = np.median(np.abs(tile - med)) + 1e-8
            robust_std = 1.4826 * mad
            high_map[y:y + tile_size, x:x + tile_size] = med + k_high * robust_std
            low_map[y:y + tile_size, x:x + tile_size] = med + k_low * robust_std

    mask = apply_hysteresis_threshold(response, low_map, high_map)
    return mask


# ---------------------------------------------------------------------------
# STAGE 3: opening-by-reconstruction (NOT plain opening/closing)
# ---------------------------------------------------------------------------
def morphological_reconstruction_clean(binary_mask, erosion_radius=1):
    """
    Removes isolated specks while fully preserving any thin structure that
    survives a small erosion -- unlike plain morphological opening/closing,
    which can arbitrarily thin, break, or merge crack segments. This is the
    standard "opening by reconstruction" trick: erode to get a seed, then
    reconstruct (grow the seed back out) constrained to the original mask.

    Parameters
    ----------
    binary_mask    : bool array
    erosion_radius : structuring element radius for the seed erosion.
                     Keep this SMALL (1 px) -- large values will erase
                     genuinely thin cracks entirely, since a thin crack may
                     not survive even a single erosion step at radius > 1.

    Returns
    -------
    cleaned bool mask
    """
    mask_u8 = binary_mask.astype(np.uint8)
    selem = disk(erosion_radius)
    seed = cv2.erode(mask_u8, selem.astype(np.uint8))
    reconstructed = reconstruction(seed, mask_u8, method='dilation')
    return reconstructed.astype(bool)


# ---------------------------------------------------------------------------
# STAGE 4: shape / continuity filtering (crack-like vs texture-like components)
# ---------------------------------------------------------------------------
def continuity_shape_filter(binary_mask, min_size=25,
                             min_elongation=3.0, max_solidity=0.6):
    """
    Keeps only connected components that are elongated and non-blobby --
    i.e. crack-like -- and removes components that are compact / blob-like,
    which is the signature of residual texture false positives.

    Elongation is measured via the skeleton-length-to-area ratio rather
    than bounding-box aspect ratio, because cracks can be curved/branching
    (a curved crack has a low bounding-box aspect ratio but a high
    skeleton-length/area ratio, so this is the more reliable geometric
    signature for THIS problem).

    Parameters
    ----------
    binary_mask     : bool array (output of morphological_reconstruction_clean)
    min_size        : minimum component area in pixels (removes speckle)
    min_elongation  : minimum (skeleton length / sqrt(area)) ratio to keep
                       a component. A compact blob has skeleton length
                       roughly proportional to sqrt(area); a thin crack
                       segment has skeleton length that grows much faster
                       than sqrt(area). Tune this on a validation subset.
    max_solidity    : maximum (area / convex_hull_area). Cracks are
                       non-convex/wiggly, so solidity is low; texture
                       blobs tend to be near-convex, so solidity is high.

    Returns
    -------
    filtered bool mask, and a list of per-component diagnostics (for
    tuning / report figures)
    """
    labeled = label(remove_small_objects(binary_mask, min_size=min_size))
    output = np.zeros_like(binary_mask, dtype=bool)
    diagnostics = []

    for region in regionprops(labeled):
        comp_mask = (labeled == region.label)
        skel = skeletonize(comp_mask)
        skel_len = skel.sum()
        area = region.area
        solidity = region.solidity if region.solidity > 0 else 1.0
        elongation = skel_len / (np.sqrt(area) + 1e-6)

        keep = (elongation >= min_elongation) and (solidity <= max_solidity)

        diagnostics.append({
            "label": region.label, "area": area, "skel_len": int(skel_len),
            "elongation": round(float(elongation), 2),
            "solidity": round(float(solidity), 3), "kept": keep,
        })

        if keep:
            output |= comp_mask

    return output, diagnostics


# ---------------------------------------------------------------------------
# STAGE 5 (OPTIONAL): traditional-ML false-positive suppressor (Random Forest)
# ---------------------------------------------------------------------------
def extract_pixel_features(gray_image, ridge_response=None):
    """
    Feature stack for the optional Random Forest stage. Each feature is a
    classical, hand-engineered descriptor (no learned representations),
    which is what keeps this "traditional ML" rather than deep learning.

    Returns a (H, W, F) float32 feature volume.
    """
    gray = gray_image.astype(np.float32)

    if ridge_response is None:
        ridge_response = compute_ridge_response(gray_image)

    # Multi-orientation Gabor bank (reuse team's existing function if you
    # prefer -- shown standalone here for a self-contained file).
    gabor_feats = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        kernel = cv2.getGaborKernel((31, 31), sigma=5.0, theta=theta,
                                     lambd=20.0, gamma=0.5, psi=np.pi / 2,
                                     ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_feats.append(np.maximum(-resp, 0))
    gabor_max = np.max(np.array(gabor_feats), axis=0)

    blackhat = cv2.morphologyEx(
        gray.astype(np.uint8), cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ).astype(np.float32)

    local_var = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    local_var = cv2.GaussianBlur(local_var ** 2, (9, 9), 0)

    def norm01(x):
        x = x.astype(np.float32)
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    feats = np.stack([
        norm01(ridge_response),
        norm01(gabor_max),
        norm01(blackhat),
        norm01(local_var),
        norm01(gray),
    ], axis=-1)

    return feats.astype(np.float32)


def train_rf_false_positive_filter(image_mask_pairs, samples_per_image=2000,
                                    n_estimators=300, max_depth=14,
                                    random_state=42):
    """
    Trains a RandomForestClassifier to distinguish real crack pixels from
    texture false-positives, using CRACK500 ground-truth masks as labels.
    This is the traditional-ML component (CrackForest-style, simplified).

    Parameters
    ----------
    image_mask_pairs : list of (gray_image, gt_binary_mask) tuples, TRAIN
                        split only (see evaluate_batch.py for the split).
    samples_per_image : class-balanced pixel subsample per image, to keep
                        training tractable on ~1,900 images on a CPU.

    Returns
    -------
    trained sklearn RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    X_list, y_list = [], []

    for gray, gt_mask in image_mask_pairs:
        feats = extract_pixel_features(gray)
        h, w, f = feats.shape
        flat_feats = feats.reshape(-1, f)
        flat_labels = gt_mask.reshape(-1).astype(np.uint8)

        pos_idx = np.where(flat_labels == 1)[0]
        neg_idx = np.where(flat_labels == 0)[0]

        n_pos = min(len(pos_idx), samples_per_image // 2)
        n_neg = min(len(neg_idx), samples_per_image // 2)
        if n_pos == 0:
            continue

        rng = np.random.default_rng(random_state)
        pos_sample = rng.choice(pos_idx, n_pos, replace=False)
        neg_sample = rng.choice(neg_idx, n_neg, replace=False)

        idx = np.concatenate([pos_sample, neg_sample])
        X_list.append(flat_feats[idx])
        y_list.append(flat_labels[idx])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", n_jobs=-1, random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def apply_rf_false_positive_filter(gray_image, candidate_mask, clf,
                                    prob_threshold=0.5):
    """
    Applies a trained RF classifier ONLY to pixels already flagged by the
    classical pipeline (candidate_mask), to suppress residual false
    positives. Running the classifier only on candidate pixels (not the
    whole image) keeps inference fast.

    At inference time no ground-truth mask is required -- only the trained
    clf object (persisted with e.g. joblib.dump / joblib.load).
    """
    feats = extract_pixel_features(gray_image)
    h, w, f = feats.shape

    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        return candidate_mask.copy()

    X = feats[ys, xs, :]
    probs = clf.predict_proba(X)[:, 1]
    keep = probs >= prob_threshold

    refined = np.zeros_like(candidate_mask)
    refined[ys[keep], xs[keep]] = True
    return refined


# ---------------------------------------------------------------------------
# END-TO-END PIPELINE
# ---------------------------------------------------------------------------
def segment_crack_ridge_pipeline(gray_image_clahe, rf_classifier=None,
                                  ridge_scales=(1, 2, 3, 4),
                                  tile_size=32, k_high=3.0, k_low=1.5,
                                  min_size=25, min_elongation=3.0,
                                  max_solidity=0.6, rf_prob_threshold=0.5):
    """
    Full pipeline: ridge response -> robust hysteresis threshold ->
    reconstruction cleanup -> shape/continuity filter -> optional RF stage.

    Parameters
    ----------
    gray_image_clahe : grayscale image AFTER your existing apply_clahe()
    rf_classifier     : optional trained classifier from
                        train_rf_false_positive_filter(); if None, stage 5
                        is skipped and the pipeline is pure classical DIP.

    Returns
    -------
    final_mask  : bool array, the crack mask to hand to skeletonization.py
    debug       : dict of intermediate results, useful for report figures
                  (ridge_response, raw_mask, cleaned_mask, diagnostics)
    """
    ridge_response = compute_ridge_response(gray_image_clahe, scales=ridge_scales)
    raw_mask = adaptive_hysteresis_threshold(
        ridge_response, tile_size=tile_size, k_high=k_high, k_low=k_low
    )
    cleaned_mask = morphological_reconstruction_clean(raw_mask, erosion_radius=1)
    shape_filtered, diagnostics = continuity_shape_filter(
        cleaned_mask, min_size=min_size,
        min_elongation=min_elongation, max_solidity=max_solidity
    )

    final_mask = shape_filtered
    if rf_classifier is not None:
        final_mask = apply_rf_false_positive_filter(
            gray_image_clahe, shape_filtered, rf_classifier,
            prob_threshold=rf_prob_threshold
        )

    debug = {
        "ridge_response": ridge_response,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "shape_filtered": shape_filtered,
        "diagnostics": diagnostics,
    }
    return final_mask, debug