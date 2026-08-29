import streamlit as st
import os
import sys
import subprocess
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pavement Crack Severity Assessment",
    page_icon="🛣️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   PAGE
   ============================================================ */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}


/* ============================================================
   HEADER
   ============================================================ */

.main-title {
    font-size: 40px;
    font-weight: 750;
    color: #1e293b;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    line-height: 1.5;
    margin-bottom: 20px;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #1e293b;
    margin-top: 28px;
    margin-bottom: 15px;
}


/* ============================================================
   RESULT / METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dfe5ec;
    border-radius: 14px;
    padding: 20px 22px;
    min-height: 105px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetric"]:hover {
    box-shadow: 0 5px 16px rgba(15, 23, 42, 0.10);
}

[data-testid="stMetricLabel"] {
    font-size: 15px !important;
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 750 !important;
    color: #1e293b !important;
}


/* CSI VALUE */

.csi-card [data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-size: 40px !important;
}


/* ============================================================
   INFORMATION BOX
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 12px;
}


/* ============================================================
   IMAGES
   ============================================================ */

[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton > button {
    border-radius: 9px;
    font-weight: 650;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    border-radius: 10px;
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding-top: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛣️ Pavement Crack Severity Assessment System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Automated pavement surface assessment using Random Forest-based
    crack detection, geometric feature extraction, skeletonization,
    graph analysis, and a composite Crack Severity Index (CSI).
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📷 Pavement Image Inspection</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a pavement surface image for automated crack detection "
    "and severity assessment."
)

uploaded_file = st.file_uploader(
    "Select pavement image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # ORIGINAL IMAGE
    # ONLY THIS IMAGE IS REDUCED
    # ========================================================

    st.image(
        uploaded_file,
        caption="Input Pavement Image",
        width=450
    )

    st.success(
        "Pavement image loaded successfully."
    )

    analyze_button = st.button(
        "🔍 Run Pavement Assessment",
        type="primary",
        use_container_width=False
    )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyze_button:

        # ----------------------------------------------------
        # DIRECTORIES
        # ----------------------------------------------------

        input_dir = os.path.join(
            "data",
            "real_world"
        )

        output_dir = os.path.join(
            "outputs",
            "real_world"
        )

        os.makedirs(
            input_dir,
            exist_ok=True
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # SAVE UPLOADED IMAGE
        # ----------------------------------------------------

        image_name = uploaded_file.name

        input_path = os.path.join(
            input_dir,
            image_name
        )

        with open(
            input_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        base_name = os.path.splitext(
            image_name
        )[0]


        # ====================================================
        # PAVEMENT ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">⚙️ Pavement Analysis</div>',
            unsafe_allow_html=True
        )

        progress = st.status(
            "Processing pavement surface...",
            expanded=True
        )


        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "predict_real_image.py",
                    input_path
                ],
                capture_output=True,
                text=True
            )


            # ------------------------------------------------
            # CHECK PIPELINE ERROR
            # ------------------------------------------------

            if result.returncode != 0:

                progress.update(
                    label="Analysis failed",
                    state="error"
                )

                st.error(
                    "The pavement analysis could not be completed."
                )

                with st.expander(
                    "Technical Error Details"
                ):

                    st.code(
                        result.stderr
                    )

                st.stop()


            progress.update(
                label="Pavement assessment completed",
                state="complete"
            )


        except Exception as error:

            progress.update(
                label="Analysis failed",
                state="error"
            )

            st.error(
                f"Unable to run the analysis: {error}"
            )

            st.stop()


        # ====================================================
        # RESULT FILE PATHS
        # ====================================================

        csv_path = os.path.join(
            output_dir,
            f"{base_name}_result.csv"
        )

        probability_path = os.path.join(
            output_dir,
            f"{base_name}_probability.png"
        )

        mask_path = os.path.join(
            output_dir,
            f"{base_name}_mask.png"
        )

        skeleton_path = os.path.join(
            output_dir,
            f"{base_name}_skeleton.png"
        )


        # ====================================================
        # CHECK CSV
        # ====================================================

        if not os.path.exists(csv_path):

            st.error(
                "Analysis completed, but the assessment result file "
                "was not found."
            )

            st.stop()


        # ====================================================
        # READ RESULT CSV
        # ====================================================

        df = pd.read_csv(
            csv_path
        )

        row = df.iloc[0]


        coverage = float(
            row["crack_coverage_percent"]
        )

        average_width = float(
            row["average_width_pixels"]
        )

        maximum_width = float(
            row["maximum_width_pixels"]
        )

        skeleton_length = float(
            row["skeleton_length_pixels"]
        )

        branch_points = int(
            row["branch_point_count"]
        )

        connected_components = int(
            row["connected_components"]
        )

        crack_area = int(
            row["crack_area_pixels"]
        )

        csi = float(
            row["CSI"]
        )

        severity = str(
            row["severity"]
        )


        # ====================================================
        # PAVEMENT CONDITION ASSESSMENT
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Pavement Condition Assessment</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # CSI + SEVERITY
        # USING STREAMLIT METRICS
        # SO NO HTML CODE WILL APPEAR
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2,
            gap="large"
        )


        with col1:

            st.markdown(
                '<div class="csi-card">',
                unsafe_allow_html=True
            )

            st.metric(
                "Crack Severity Index",
                f"{csi:.2f} / 100"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with col2:

            st.metric(
                "Visual Severity Classification",
                severity
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.info(
            "The CSI represents the severity of visible crack "
            "characteristics detected from the pavement image. "
            "It is an image-based assessment indicator and should "
            "not be interpreted as a direct structural safety rating "
            "of the pavement."
        )


        # ====================================================
        # DETECTED CRACK CHARACTERISTICS
        # ====================================================

        st.markdown(
            '<div class="section-title">📐 Detected Crack Characteristics</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # FIRST ROW
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Crack Coverage",
                f"{coverage:.2f}%"
            )


        with col2:

            st.metric(
                "Average Crack Width",
                f"{average_width:.2f} px"
            )


        with col3:

            st.metric(
                "Maximum Crack Width",
                f"{maximum_width:.2f} px"
            )


        # ----------------------------------------------------
        # SECOND ROW
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Skeleton Length",
                f"{skeleton_length:.2f} px"
            )


        with col2:

            st.metric(
                "Branch Points",
                branch_points
            )


        with col3:

            st.metric(
                "Connected Crack Regions",
                connected_components
            )


        # ----------------------------------------------------
        # CRACK AREA
        # ----------------------------------------------------

        st.metric(
            "Detected Crack Area",
            f"{crack_area:,} pixels"
        )


        # ====================================================
        # VISUAL ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">🧬 Crack Detection & Structural Representation</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PROBABILITY + MASK
        # KEEP SAME SIZE
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2,
            gap="large"
        )


        with col1:

            st.subheader(
                "Random Forest Probability Map"
            )

            if os.path.exists(
                probability_path
            ):

                st.image(
                    probability_path,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Probability map unavailable."
                )


        with col2:

            st.subheader(
                "Detected Crack Mask"
            )

            if os.path.exists(
                mask_path
            ):

                st.image(
                    mask_path,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Crack mask unavailable."
                )


        # ====================================================
        # SKELETON
        # KEEP SAME SIZE
        # ====================================================

        st.subheader(
            "Skeleton / Crack Network Representation"
        )


        if os.path.exists(
            skeleton_path
        ):

            st.image(
                skeleton_path,
                use_container_width=True
            )

        else:

            st.warning(
                "Skeleton representation unavailable."
            )


        # ====================================================
        # ASSESSMENT REPORT
        # ====================================================

        st.markdown(
            '<div class="section-title">📥 Assessment Report</div>',
            unsafe_allow_html=True
        )


        csv_data = df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )


        st.download_button(
            label="⬇️ Download Detailed Assessment Report",
            data=csv_data,
            file_name=f"{base_name}_pavement_assessment.csv",
            mime="text/csv"
        )


        # ====================================================
        # METHODOLOGY
        # ====================================================

        st.markdown(
            '<div class="section-title">🔬 Assessment Methodology</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            """
            **1. Image Processing**  
            The uploaded pavement image is converted and processed
            to extract classical image features.

            **2. Random Forest Crack Detection**  
            A trained Random Forest model estimates the probability
            of each image pixel belonging to a pavement crack.

            **3. Crack Segmentation**  
            A probability threshold is applied to obtain the detected
            crack mask followed by morphological cleaning.

            **4. Geometric Analysis**  
            Crack coverage, crack area, average width and maximum width
            are calculated from the detected crack regions.

            **5. Skeleton and Graph Analysis**  
            The crack mask is skeletonized to analyze crack length,
            branch points and connected crack regions.

            **6. Crack Severity Index**  
            The extracted features are normalized and combined using
            weighted scoring to obtain the final CSI on a 0–100 scale.
            """
        )


        # ====================================================
        # FOOTER
        # ====================================================

        st.divider()

        st.markdown(
            """
            <div class="footer">
            Pavement Crack Severity Assessment System |
            Random Forest + Image Processing + Skeleton & Graph Analysis
            </div>
            """,
            unsafe_allow_html=True
        )