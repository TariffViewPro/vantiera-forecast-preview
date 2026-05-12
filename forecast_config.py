# forecast_config.py

PROFILE_THRESHOLDS = {
    "flat_cv": 0.15,          # CV < 15% = Flat (relaxed from 0.10)
    "decline_threshold": -0.15,  # Linear trend slope < -0.15 = Declining
    "intermittent_zero_ratio": 0.5,  # >50% zero months = Intermittent
    "seasonality_period": 12
}

OUTLIER_IQR_FACTOR = 1.5  # Standard IQR multiplier (from 1.0)
OUTLIER_STD_DEV_THRESHOLD = 2.0  # Standard std-dev threshold (from 1.0)
OUTLIER_WINDOW = 3  # Wider window for local std-dev (from 1)

# --- General Settings ---
MIN_HISTORY_REQUIRED = 12        # Reduced to 12 periods (from 18)
VALIDATION_PERIODS = 4          # Reduced to 4 periods (from 6)
DEFAULT_FORECAST_LENGTH = 12    # Months to forecast
MAX_HISTORY_COLUMNS = 36        # Cap on number of columns to process

# --- Recency Weighting for Model Selection ---
RECENCY_WEIGHTING_ENABLED = False      # Toggle to True if recent errors should be weighted more
RECENT_WEIGHT_RATIO = 0.7             # Weight for recent 4 months in MAPE calculation
# Note: RECENT_WEIGHT_RATIO applies to VALIDATION_PERIODS (now 4)

# --- Re-Evaluation Logic ---
RE_EVALUATION_THRESHOLD = 0.3   # If last 3-month avg deviates from forecast by 30%, flag for review
# Note: Currently unused; implement in forecast_app.py if needed

# ==== Summary badges (table) ==== Added 10/12/25 for visualization table
STABLE_DELTA_PCT = 0.07   # was 0.05  → widen “Stable” band to ±7%
STABLE_CV        = 0.35   # was 0.30  → slightly more tolerant variance for “Stable”
VOLATILE_CV      = 0.60   # was 0.40  → fewer items flagged “Volatile” on CV alone
ZERO_RATIO_VOL   = 0.50   # was 0.33  → require ≥50% zeros to call “Volatile”
SMALL_BASELINE   = 24     # was 12    → treat very low-volume series as low-volume

# --- Holt-Winters Config ---
HOLT_WINTERS_PARAMS = {
    "seasonal": "add",           # or "mul"
    "seasonal_periods": 12
}