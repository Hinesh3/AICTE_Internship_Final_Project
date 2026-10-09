# AquaSafe — Water Quality Prediction Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wmw8UdVCqjs9MaZr0GTKJ0ym1b_pIbpf?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, data-driven water potability prediction and advisory system. AquaSafe analyzes physicochemical water sample parameters using machine learning to provide instant safety assessments, health risk evaluations, and actionable remediation guidance.

---

## Google Colab Notebook

You can explore the full exploratory data analysis (EDA), preprocessing, and model training workflow directly in your browser without local setup:

- **Interactive Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1wmw8UdVCqjs9MaZr0GTKJ0ym1b_pIbpf?usp=sharing)
- **Local Notebook File:** `Colab File/Water_Quality_Prediction_Agent.ipynb`

### What is covered in the Colab notebook:
- Data exploration and missing value distributions
- Correlation analysis across physicochemical properties
- Median imputation and standard scaling pipelines
- Stratified 5-fold cross-validation benchmarking Logistic Regression, Random Forest, and XGBoost
- Evaluation curves (ROC-AUC, Precision-Recall) and feature importance analysis

---

## Project Overview

Access to clean drinking water is vital for public health. In many communities and remote areas, access to certified analytical laboratories is delayed or cost-prohibitive. AquaSafe provides an accessible first-pass screening tool that assists users in identifying potential contamination risks before official testing.

The system evaluates **9 core water quality parameters** against guidelines established by the **World Health Organization (WHO)** and the **Bureau of Indian Standards (BIS)**, and runs inference through a machine learning classifier trained on empirical water potability records.

---

## Key Features

- **Real-Time Potability Prediction:** Instant classification (Safe vs. Unsafe) with class probability confidence scores.
- **Rule-Based Remediation Advisor:** Parameter-by-parameter diagnostic checks against WHO guideline limits with practical, plain-language corrective measures (e.g., reverse osmosis, activated carbon filtration, neutralization).
- **AI-Powered Explanations:** Optional integration with Google Gemini API to generate concise, friendly explanations for non-technical users.
- **Safe Defaults & Presets:** Pre-loaded with safe baseline parameters, alongside quick-test presets (Clean Well Water, Polluted Runoff, Borderline Sample).
- **Historical Tracking Dashboard:** Local SQLite persistence logging every test with interactive Plotly trendlines, distribution charts, and consecutive unsafe sample alerts.
- **Model Transparency:** Dedicated diagnostic page displaying test-set performance metrics, confusion matrix, and feature importances.

---

## Water Quality Parameters

The model evaluates nine physicochemical parameters:

| Parameter | Unit | WHO / BIS Guideline | Description & Significance |
| :--- | :---: | :---: | :--- |
| **pH** | — | 6.5 – 8.5 | Indicates acidity (<6.5) or alkalinity (>8.5). Extreme values accelerate corrosion or scale formation. |
| **Hardness** | mg/L | ≤ 500.0 | Concentration of dissolved calcium and magnesium salts. Causes scale buildup in pipes and appliances. |
| **Solids (TDS)** | ppm | ≤ 500.0 | Total dissolved organic and inorganic substances. High levels affect palatability and may indicate mineral contamination. |
| **Chloramines** | ppm | ≤ 4.0 | Chlorine-ammonia disinfectant used in water treatment. Elevated levels cause odor and taste degradation. |
| **Sulfate** | mg/L | ≤ 250.0 | Naturally occurring mineral. High concentrations can cause gastrointestinal distress and a laxative effect. |
| **Conductivity** | μS/cm | ≤ 400.0 | Electrical conductivity reflecting dissolved ionic species and total mineral content. |
| **Organic Carbon** | ppm | ≤ 2.0 | Total organic carbon (TOC) from decaying vegetation or runoff. Reacts with disinfectants to create harmful byproducts. |
| **Trihalomethanes** | μg/L | ≤ 80.0 | Chemical byproducts formed when chlorine reacts with organic matter. Known long-term health risk. |
| **Turbidity** | NTU | ≤ 1.0 | Water clarity measurement. Suspended particles can shelter pathogens from disinfection. |

---

## Machine Learning Architecture

### Preprocessing & Modeling Pipeline
1. **Median Imputation (`SimpleImputer`):** Missing parameter values are imputed using median values to maintain robustness against outliers.
2. **Feature Standardization (`StandardScaler`):** Features are standardized to zero mean and unit variance.
3. **Model Selection:** Three classifiers were evaluated via 5-fold Stratified Cross-Validation:
   - Logistic Regression
   - XGBoost Classifier
   - Random Forest Classifier (Selected best model based on cross-validated ROC-AUC)
4. **Class Imbalance Strategy:** Handled via balanced sample weighting. The pipeline specifically tracks **Recall for the Unsafe class**, minimizing false negatives (calling contaminated water safe).

### Test-Set Performance

Evaluated on a held-out test split (20% of data):

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Accuracy** | 0.6387 | Overall classification accuracy |
| **Precision** | 0.5463 | Precision for Potable class |
| **Recall** | 0.4375 | Recall for Potable class |
| **F1-Score** | 0.4859 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.6622 | Area under the ROC curve |
| **Unsafe Class Recall** | **0.7675** | High sensitivity for detecting unsafe samples |

Artifacts generated during training are saved to the `outputs/` directory:
- `outputs/metrics.csv` — Quantitative evaluation metrics.
- `outputs/confusion_matrix.png` — Heatmap of true vs. predicted classes.
- `outputs/feature_importance.png` — Ranked feature importances.

---

## Project Structure

```
Water Quality Prediction Agent/
├── Colab File/
│   └── Water_Quality_Prediction_Agent.ipynb  # Interactive Google Colab notebook
├── data/
│   ├── history.db                            # SQLite database of test records
│   └── water_potability.csv                  # Kaggle water quality dataset
├── models/
│   └── model.pkl                             # Trained scikit-learn pipeline
├── outputs/
│   ├── confusion_matrix.png                  # Evaluation confusion matrix plot
│   ├── feature_importance.png                # Feature importance visualization
│   └── metrics.csv                           # Test metrics table
├── advisor.py                                # Rule-based & Gemini AI advisor
├── app.py                                    # Streamlit web application
├── config.py                                 # WHO guideline limits & parameters
├── requirements.txt                          # Python dependencies
├── train.py                                  # Training & cross-validation script
└── README.md                                 # Documentation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository & Install Dependencies
```bash
git clone <repository-url>
cd "Water Quality Prediction Agent"
pip install -r requirements.txt
```

### 3. Verify Dataset
The dataset file `data/water_potability.csv` is included. If re-downloading is required, obtain it from [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) and place it in the `data/` folder.

---

## Running the Application

### Launch Web Interface
```bash
streamlit run app.py
```
Open your browser to:
```
http://localhost:8501
```

### Application Pages:
1. **Test Water:** Enter physicochemical parameters manually or select a preset to obtain an instant safety assessment and remediation advice.
2. **Dashboard:** Review test records, distribution pie charts, parameter historical trends, and export records to CSV.
3. **Model Info:** Inspect test metrics, confusion matrix, feature importances, and pipeline architecture.

---

## Retraining the Model

To execute the cross-validation benchmarking, fit the best pipeline, and refresh the saved model and visualizations:

```bash
python train.py
```

The script will:
- Run 5-fold cross-validation on Logistic Regression, Random Forest, and XGBoost.
- Identify the highest ROC-AUC model.
- Save the trained pipeline to `models/model.pkl`.
- Export metrics and plots to `outputs/`.

---

## Optional: Gemini AI Advice Setup

To enable natural-language explanations powered by Google Gemini:

1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the environment variable:

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
streamlit run app.py
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
streamlit run app.py
```

**macOS / Linux:**
```bash
export GEMINI_API_KEY="your-api-key-here"
streamlit run app.py
```

*Note: If no API key is configured, the application seamlessly uses the built-in rule-based expert advisor.*

---

## Disclaimer

This software is designed solely for **educational, exploratory, and general guidance purposes**. It does not constitute certified laboratory analysis or an official safety clearance. For regulatory compliance and potable drinking water certification, always submit physical samples to an accredited water testing laboratory.
