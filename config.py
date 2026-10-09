"""
config.py
=========
Central configuration for Water Quality Prediction Agent.
Stores WHO / BIS approximate drinking-water guideline limits and
parameter metadata used by the Streamlit UI and the advisor.

NOTE: These are *approximate* guideline values for general educational
reference only.  Always verify against the latest official standards
and certified laboratory test results.
"""

# ---------------------------------------------------------------------------
# Feature column order (must match training data)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# ---------------------------------------------------------------------------
# Parameter display metadata
# ---------------------------------------------------------------------------
# Each entry:  { display_name, unit, min_val, max_val, default,
#                who_min, who_max, description }
# who_min / who_max = approximate WHO / BIS guideline limits.
# None means "no lower / upper limit specified in guidelines".
# ---------------------------------------------------------------------------
PARAM_CONFIG = {
    "ph": {
        "display_name": "pH",
        "unit": "—",
        "min_val": 0.0,
        "max_val": 14.0,
        "default": 7.2,
        "step": 0.1,
        "who_min": 6.5,
        "who_max": 8.5,
        "description": "Measure of acidity or alkalinity of water.",
    },
    "Hardness": {
        "display_name": "Hardness",
        "unit": "mg/L",
        "min_val": 0.0,
        "max_val": 1000.0,
        "default": 180.0,
        "step": 1.0,
        "who_min": None,
        "who_max": 500.0,
        "description": "Calcium & magnesium salts dissolved in water.",
    },
    "Solids": {
        "display_name": "Total Dissolved Solids (TDS)",
        "unit": "ppm",
        "min_val": 0.0,
        "max_val": 50000.0,
        "default": 320.0,
        "step": 10.0,
        "who_min": None,
        "who_max": 500.0,
        "description": "Total amount of dissolved substances.",
    },
    "Chloramines": {
        "display_name": "Chloramines",
        "unit": "ppm",
        "min_val": 0.0,
        "max_val": 15.0,
        "default": 2.5,
        "step": 0.1,
        "who_min": None,
        "who_max": 4.0,
        "description": "Disinfectant used in water treatment.",
    },
    "Sulfate": {
        "display_name": "Sulfate",
        "unit": "mg/L",
        "min_val": 0.0,
        "max_val": 1000.0,
        "default": 140.0,
        "step": 1.0,
        "who_min": None,
        "who_max": 250.0,
        "description": "Naturally occurring mineral in water.",
    },
    "Conductivity": {
        "display_name": "Conductivity",
        "unit": "μS/cm",
        "min_val": 0.0,
        "max_val": 2000.0,
        "default": 280.0,
        "step": 1.0,
        "who_min": None,
        "who_max": 400.0,
        "description": "Ability of water to conduct electric current.",
    },
    "Organic_carbon": {
        "display_name": "Organic Carbon",
        "unit": "ppm",
        "min_val": 0.0,
        "max_val": 30.0,
        "default": 1.2,
        "step": 0.1,
        "who_min": None,
        "who_max": 2.0,
        "description": "Total organic carbon from organic matter.",
    },
    "Trihalomethanes": {
        "display_name": "Trihalomethanes",
        "unit": "μg/L",
        "min_val": 0.0,
        "max_val": 130.0,
        "default": 25.0,
        "step": 0.5,
        "who_min": None,
        "who_max": 80.0,
        "description": "By-products of water disinfection with chlorine.",
    },
    "Turbidity": {
        "display_name": "Turbidity",
        "unit": "NTU",
        "min_val": 0.0,
        "max_val": 10.0,
        "default": 0.5,
        "step": 0.1,
        "who_min": None,
        "who_max": 1.0,
        "description": "Cloudiness / haziness of water.",
    },
}

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_PATH          = "data/water_potability.csv"
MODEL_PATH         = "models/model.pkl"
CM_IMAGE_PATH      = "outputs/confusion_matrix.png"
FI_IMAGE_PATH      = "outputs/feature_importance.png"
METRICS_PATH       = "outputs/metrics.csv"
HISTORY_DB_PATH    = "data/history.db"

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "This is general guidance and does not replace certified "
    "laboratory testing. Consult your local water authority for official results."
)
