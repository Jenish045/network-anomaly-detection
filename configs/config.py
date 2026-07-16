"""
Global configuration for the Network Anomaly Detection System.

This file keeps all configurable values in one place so that experiments,
training parameters, and file locations can be changed without touching
the rest of the codebase.
"""

from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project folders
ASSETS_DIR = PROJECT_ROOT / "assets"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "exports"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
METRICS_DIR = MODELS_DIR / "metrics"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Raw NSL-KDD dataset files
TRAIN_DATA_PATH = RAW_DATA_DIR / "KDDTrain+.txt"
TEST_DATA_PATH = RAW_DATA_DIR / "KDDTest+.txt"

# These are the only categorical columns in the dataset.
# Everything else is already numerical.
CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag",
]

TARGET_COLUMN = "label"
NORMAL_LABEL = "normal"

# We don't use the difficulty level anywhere in the project.
DROP_COLUMNS = ["difficulty"]

# Keeping the random seed fixed makes our experiments reproducible.
RANDOM_STATE = 42

# Isolation Forest settings
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.50

# Default DBSCAN parameters.
# These can still be overridden from the Streamlit dashboard.
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 10

# These values gave stable results during experimentation.
# Having them here makes hyperparameter tuning much easier later.
AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_ENCODING_DIM = 8
AE_THRESHOLD_PERCENTILE = 90

# Plotly settings used across notebooks and the dashboard.
PLOT_TEMPLATE = "plotly_white"
FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 700

APP_TITLE = "Network Anomaly Detection System"
APP_ICON = "🛡️"
APP_LAYOUT = "wide"

# Most evaluation functions work with integer labels instead of strings.
NORMAL_CLASS = 0
ANOMALY_CLASS = 1

# Preprocessing strategy configurations for future scalability
# Supported options: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), 'robust' (RobustScaler)
SCALER_TYPE = "standard"
# Supported options: 'label' (LabelEncoder)
ENCODER_TYPE = "label"

# Autoencoder architectural hyperparameters
AE_HIDDEN_DIMS = [64, 32, 16]
AE_DROPOUT_RATE = 0.20

# Centralized file paths for models and data artifacts
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.joblib"
ENCODER_PATH = PROCESSED_DATA_DIR / "label_encoders.joblib"

AUTOENCODER_MODEL_PATH = TRAINED_MODELS_DIR / "autoencoder.keras"
AUTOENCODER_METRICS_PATH = METRICS_DIR / "autoencoder_metrics.json"

DBSCAN_METRICS_PATH = METRICS_DIR / "dbscan_metrics.json"

ISOLATION_FOREST_MODEL_PATH = TRAINED_MODELS_DIR / "isolation_forest.joblib"
ISOLATION_FOREST_METRICS_PATH = METRICS_DIR / "isolation_forest_metrics.json"