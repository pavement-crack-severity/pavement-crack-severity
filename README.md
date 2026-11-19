# 🛣️ Pavement Crack Severity Assessment

An image-based pavement crack detection and severity assessment system using **Random Forest, Image Processing, Skeletonization, and Graph-Based Analysis**.

Streamlit Link - https://pavement-crack-severity-yskqe4la3tiwpkek7jkzfq.streamlit.app/

##  About the Project

Pavement cracks are early indicators of road deterioration. Identifying and assessing them at an early stage can help support timely road maintenance.

This project presents an automated system that analyzes pavement surface images, detects crack regions, extracts geometric and structural features, and calculates a **Crack Severity Index (CSI)** to estimate the severity of visible pavement cracks.

The application is built using Python and Streamlit, allowing users to upload pavement images and view the crack detection and severity assessment results.

##  Key Features

- **Image Feature Extraction:** Extracts nine image features, including grayscale intensity, CLAHE, Gabor, Black-hat, Frangi, gradient magnitude, local mean, local standard deviation, and local contrast.
- **Crack Detection:** Uses a trained Random Forest classifier to identify crack pixels and generate a probability map.
- **Crack Segmentation:** Applies thresholding and morphological processing to obtain a refined crack mask.
- **Skeletonization:** Converts detected crack regions into thin skeleton structures for further analysis.
- **Graph-Based Analysis:** Examines crack connectivity, branch points, endpoints, and connected components.
- **Geometric Measurements:** Calculates crack coverage, average width, maximum width, and skeleton length.
- **Severity Assessment:** Calculates a weighted Crack Severity Index (CSI) ranging from 0 to 100.
- **Interactive Interface:** Provides a Streamlit application for uploading images and viewing assessment results.

##  Working Methodology

The system follows these main steps:

1. Upload a pavement surface image.
2. Extract nine image features using image processing techniques.
3. Apply the trained Random Forest classifier for pixel-level crack detection.
4. Generate a probability map and obtain a binary crack mask.
5. Refine the detected crack regions using morphological processing.
6. Perform skeletonization to represent crack structures.
7. Apply graph-based analysis to identify crack connectivity and branching patterns.
8. Calculate geometric and topological features.
9. Normalize the extracted features and calculate the Crack Severity Index.
10. Display the final measurements and severity category through the Streamlit application.

##  Crack Severity Classification

The system classifies pavement conditions into five severity levels based on the calculated CSI.

| CSI Score     | Severity Level |
| ------------- | -------------- |
| 0 – Below 20  | Low            |
| 20 – Below 40 | Moderate       |
| 40 – Below 60 | High           |
| 60 – Below 80 | Very High      |
| 80 – 100      | Severe         |

##  Technologies Used

| Technology   | Purpose                                       |
| ------------ | --------------------------------------------- |
| Python       | Core programming language                     |
| OpenCV       | Image processing and morphological operations |
| NumPy        | Numerical operations and feature handling     |
| scikit-image | Image processing and skeletonization          |
| scikit-learn | Random Forest classification                  |
| NetworkX     | Graph construction and connectivity analysis  |
| Joblib       | Saving and loading the trained model          |
| Pandas       | Handling assessment results                   |
| Streamlit    | Web application interface                     |

## 📂 Project Structure

```text
pavement-crack-severity/
│
├── app.py
├── predict_real_image.py
├── ml_features.py
├── geometric_features.py
├── feature_extraction.py
├── skeletonization.py
├── preprocessing.py
├── segmentation.py
│
├── models/
│   ├── crack_random_forest.joblib
│   └── crack_random_forest_v2.joblib
│
├── data/
│   └── samples/
│
├── outputs/
│
├── requirements.txt
└── README.md
```

##  Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pavement-crack-severity/pavement-crack-severity.git
```

### 2. Navigate to the project directory

```bash
cd pavement-crack-severity
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit address.

**Note:**Make sure that the trained model files and all supporting Python scripts are available in their respective directories before running the application.

##  Expected Output

The application provides:

- Uploaded pavement image
- Crack probability map
- Detected crack mask
- Skeleton representation
- Geometric and graph-based measurements
- Crack Severity Index (CSI)
- Final pavement severity classification

##  Future Improvements

- Improve crack detection using advanced machine learning and deep learning techniques.
- Convert pixel-based measurements into real-world units using camera calibration.
- Test the system with larger and more diverse pavement datasets.
- Integrate GPS-based location information for road damage mapping.
- Develop a mobile application for pavement inspection in the field.

##  Project Information

**Project Title:** Pavement Crack Severity Assessment Using Random Forest and Graph-Based Crack Analysis

**Domain:** Machine Learning, Computer Vision, and Image Processing

**Dataset:** CRACK500

**Application:** Streamlit

---

 If you find this project interesting, feel free to explore the repository and try the live application.
