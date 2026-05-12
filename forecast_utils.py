import pandas as pd
import numpy as np
import math
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None
from forecast_config import (
    OUTLIER_IQR_FACTOR, OUTLIER_STD_DEV_THRESHOLD, OUTLIER_WINDOW, PROFILE_THRESHOLDS,
    HOLT_WINTERS_PARAMS, VALIDATION_PERIODS, DEFAULT_FORECAST_LENGTH, MIN_HISTORY_REQUIRED
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Optional, Dict, Tuple

import logging
logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
except Exception:
    ETSModel = None  # fallback handled in AutoETS

import os
LS_UP   = float(os.getenv("LS_UP",   "1.50"))  # level shift up threshold (baseline defaults)
LS_DOWN = float(os.getenv("LS_DOWN", "0.67"))  # level shift down threshold (baseline defaults)

ZD_MIN_ZERO = float(os.getenv("ZD_MIN_ZERO", "0.50"))  # ZeroDecay zero-ratio guard (U19)
ZD_ACC_LO   = float(os.getenv("ZD_ACC_LO",   "0.90"))  # totals lower bound for accept (U19)
ZD_ACC_HI   = float(os.getenv("ZD_ACC_HI",   "1.10"))  # totals upper bound for accept (U19)

def _too_low(forecast, recent_mean, floor=0.60):
    f = np.asarray(forecast, dtype=float)
    if f.size == 0:
        return False
    # Veto if *all* future points are implausibly low vs recent level
    return (recent_mean > 0) and np.all(f <= floor * recent_mean + 1e-8)

def _reject_flat_constant(forecast, recent_mean, band=0.25):
    """
    Veto if the forecast is (a) essentially constant AND
    (b) that constant level is far from the recent mean by > band.
    """

    f = np.asarray(forecast, dtype=float)
    if f.size == 0 or recent_mean <= 0:
        return False

    # treat tiny wiggle as constant
    if np.nanmax(f) - np.nanmin(f) <= 1e-9:
        c = float(f[0])
        lo = (1 - band) * recent_mean
        hi = (1 + band) * recent_mean
        return not (lo <= c <= hi)

    return False

def _mad_zscores(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad == 0:
        return np.zeros_like(a, dtype=float)
    return 0.6745 * (a - med) / mad

def clean_outliers_with_local_median_mad(y: np.ndarray, z_thresh: float = 3.5, window: int = 5):
    """
    Flag extreme points by MAD-z and replace them with a local median
    of +/- `window` neighbors (leaves other points untouched).
    Returns (y_clean, mask_flagged).
    """
    y = np.asarray(y, dtype=float)
    z = _mad_zscores(y)
    mask = np.abs(z) > z_thresh
    if not np.any(mask):
        return y.copy(), mask
    y_clean = y.copy()
    idxs = np.where(mask)[0]
    for i in idxs:
        lo = max(0, i - window)
        hi = min(len(y), i + window + 1)
        neigh = np.r_[y[lo:i], y[i+1:hi]]
        neigh = neigh[np.isfinite(neigh)]
        neigh = neigh[neigh > 0] #Added 10/9/25
        if neigh.size == 0:
            continue
        y_clean[i] = float(np.median(neigh))
    return y_clean, mask

def _winsorize_positive(a, p=90):
    a = np.asarray(a, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0)
    pos = a[a > 0]
    if pos.size == 0:
        return a
    cap = np.percentile(pos, p)
    a = a.copy()
    a[a > cap] = cap
    return a

def _winsorized_mean_total(y: np.ndarray, p: float = 90.0, m: int = 12) -> float:
    """
    Compute the winsorized-mean × horizon total used by Smoothed Profile INT.
    Caps positives at p-th percentile, averages, multiplies by m.
    """
    a = np.asarray(y, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0)
    if a.size == 0:
        return 0.0
    pos = a[a > 0]
    if pos.size == 0:
        return 0.0
    cap = float(np.percentile(pos, p))
    a = a.copy()
    a[a > cap] = cap
    return float(np.mean(a) * m)

def _moving_average_1d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Simple centered moving average with edge padding.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    pad = int(k // 2)
    x = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(x, np.ones(k) / float(k), mode='valid')

# --- Added 10/12/25 for visualization results ---
def _badge_from_metrics(pct_change: float, cv12: float, zero_ratio12: float, hist_last12_sum: float,
                        cfg) -> Tuple[str, str]:
    # Order of precedence: Low-Volume → Volatile → Stable/Rising/Falling
    if hist_last12_sum <= cfg.SMALL_BASELINE:
        return "⚪ Low-Volume", "Low totals in last 12 months"
    if (cv12 > cfg.VOLATILE_CV) or (zero_ratio12 >= cfg.ZERO_RATIO_VOL):
        return "🟠 Volatile", f"CV={cv12:.2f}, zeros={zero_ratio12:.2f}"
    # Treat small deltas as Stable unless flagged Volatile/Low-Volume above
    if abs(pct_change) <= cfg.STABLE_DELTA_PCT:
        return "🔵 Stable", f"Δ={pct_change:+.0%}, CV={cv12:.2f}"
    if pct_change > 0:
        return "🟢 Rising", f"Δ={pct_change:+.0%}"
    else:
        return "🟡 Falling", f"Δ={pct_change:+.0%}"

def summarize_items_for_table(
    items: List[str],
    last12_history: Dict[str, List[float]],
    next12_forecast: Dict[str, List[float]],
    cfg_module,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      item_id | fcst_12_sum | hist_last12_sum | pct_change | cv12 | zero_ratio12 | badge | badge_hint
    """
    rows = []
    for it in items:
        hist = last12_history.get(it, [])
        fcst = next12_forecast.get(it, [])
        # sums
        hist_sum = float(np.nansum(hist)) if len(hist) else 0.0
        fcst_sum = float(np.nansum(fcst)) if len(fcst) else 0.0
        # pct change
        denom = hist_sum if hist_sum != 0 else 1e-9
        pct_change = (fcst_sum - hist_sum) / denom
        # cv and zeros (history-based)
        cv12 = 0.0
        zero_ratio = 0.0
        if len(hist):
            arr = np.array(hist, dtype=float)
            mu = float(np.nanmean(arr)) if arr.size else 0.0
            sd = float(np.nanstd(arr, ddof=0)) if arr.size else 0.0
            cv12 = (sd / mu) if mu > 0 else 0.0
            zero_ratio = float(np.count_nonzero(arr == 0)) / max(len(arr), 1)
        badge, hint = _badge_from_metrics(pct_change, cv12, zero_ratio, hist_sum, cfg_module)
        rows.append({
            "item_id": it,
            "fcst_12_sum": fcst_sum,
            "hist_last12_sum": hist_sum,
            "pct_change": pct_change,
            "cv12": cv12,
            "zero_ratio12": zero_ratio,
            "badge": badge,
            "badge_hint": hint,
        })
    df = pd.DataFrame(rows)
    # Presentation-friendly columns added here to keep app.py clean
    df["Forecast (12)"] = (df["fcst_12_sum"].round(0)).astype(int)
    df["vs. Last 12"] = df["pct_change"].apply(lambda x: f"{x:+.0%}")
    df["Badge"] = df["badge"]
    # final ordering
    return df[["item_id", "Forecast (12)", "vs. Last 12", "Badge",
               "fcst_12_sum", "hist_last12_sum", "pct_change", "cv12", "zero_ratio12", "badge_hint"]]
# --- END Added 10/12/25 for visualization results ---

class SmoothedProfileINT:
    """
    Forecaster wrapper so the method can be scored alongside others.
    Note: forecast() returns integers (sum preserved) by design.
    """
    def __init__(self, series, winsor_p: float = 90.0, smooth_k: int = 3):
        self.series = np.asarray(series, dtype=np.float64)
        self.winsor_p = float(winsor_p)
        self.smooth_k = int(smooth_k)

    def forecast(self, steps: int):
        return _smoothed_profile_int_forecast(self.series, horizon=steps, winsor_p=self.winsor_p, smooth_k=self.smooth_k)

def _is_obsolete(a, tail=12):
    a = np.asarray(a, dtype=np.float64)
    if a.size < tail:
        return False
    return np.all(a == 0) or np.sum(a) < 0.1  # Stricter threshold

def _is_sparse(a, min_nonzero=3, zero_ratio_thr=0.7):
    a = np.asarray(a, dtype=np.float64)
    nonzero_cnt = int(np.sum(a > 0))
    zero_ratio = float(np.sum(a == 0)) / max(1, a.size)
    return (zero_ratio >= zero_ratio_thr) or (nonzero_cnt <= min_nonzero), nonzero_cnt, zero_ratio

def _is_lumpy_cv2(a, cv2_thr=0.49, min_n=24):
    """High-variability (lumpy) detector using CV^2 = (std/mean)^2."""
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return False
    x = x[x >= 0.0]
    if x.size == 0:
        return False
    mu = float(np.mean(x))
    if mu <= 0.0:
        return False
    sigma = float(np.std(x))
    cv2 = (sigma / mu) ** 2
    return cv2 >= cv2_thr

def _has_seasonality(y, period=12, min_len=None, acf_threshold=0.25):
    """
    Seasonality hint: require enough history (≥ 2 seasons) AND decent lag-`period` ACF.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)

    req = max(18, int(period) * 2) if min_len is None else int(min_len)
    if len(y) < max(req, int(period) + 1):
        return False

    z = y - np.mean(y) if np.mean(y) != 0 else y.copy()
    denom = float(np.dot(z, z))
    if denom <= 0:
        return False

    lag = int(period)
    num = float(np.dot(z[lag:], z[:-lag]))
    acf = num / denom
    return acf >= float(acf_threshold)

def detect_profile(y):
    y = np.array(y, dtype=np.float64)
    cv = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0
    zero_ratio = np.sum(y == 0) / len(y)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    if zero_ratio > PROFILE_THRESHOLDS['intermittent_zero_ratio']:
        return 'intermittent'
    elif slope < PROFILE_THRESHOLDS['decline_threshold']:
        return 'declining'
    elif cv < 0.3 and zero_ratio > 0.4:  # Relaxed flat criteria
        return 'flat'
    elif _has_seasonality(y, period=HOLT_WINTERS_PARAMS['seasonal_periods']):
        return 'seasonal'
    elif slope > 0.01 or cv > 0.5:  # Explicit trending criteria
        return 'trending'
    else:
        return 'flat'

# ---- Robust trend features (for Damped Trend gating) ------------------------
def _theil_sen_slope(y: np.ndarray) -> float:
    """
    Robust slope via Theil–Sen on the last up to 24 points.
    O(n^2) on <=24 is ~300 slopes, trivial cost.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0
    # restrict to tail for stability + speed
    tail = y[-min(24, y.size):]
    n = tail.size
    if n < 3:
        return 0.0
    slopes = []
    for i in range(n - 1):
        yi = tail[i]
        for j in range(i + 1, n):
            dj = j - i
            num = tail[j] - yi
            if dj != 0:
                slopes.append(num / dj)
    if not slopes:
        return 0.0
    return float(np.median(slopes))

def _robust_trend_features(y: np.ndarray) -> dict:
    """
    Returns:
      - slope_abs_over_level: |slope| / level (level = median of last 6)
      - r2: variance explained by robust line on tail
      - mu6: mean of last 6 points (recent level)
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)
    if y.size == 0:
        return {"slope_abs_over_level": 0.0, "r2": 0.0, "mu6": 0.0}

    # recent level
    tail6 = y[-min(6, y.size):]
    mu6 = float(np.mean(tail6)) if tail6.size else 0.0
    level = float(np.median(tail6)) if tail6.size else (float(np.median(y)) if y.size else 0.0)
    level = max(level, 1e-8)

    # fit robust line on tail (<=24)
    tail = y[-min(24, y.size):]
    n = tail.size
    x = np.arange(n, dtype=np.float64)
    slope = _theil_sen_slope(tail)
    # intercept anchored at median(x), median(y) for robustness
    x_med = float(np.median(x))
    y_med = float(np.median(tail))
    intercept = y_med - slope * x_med
    y_hat = intercept + slope * x

    # R^2 on the tail
    ss_res = float(np.sum((tail - y_hat) ** 2))
    ss_tot = float(np.sum((tail - np.mean(tail)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)

    return {
        "slope_abs_over_level": float(abs(slope) / level),
        "r2": float(r2),
        "mu6": float(mu6),
    }

def compute_guard_params(y, m: int = 12):
    """
    Derive dynamic guardrail multipliers from the series itself.
    Returns a dict with spike_factor (mean-level guard) and peak_factor (point-level guard).
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {"spike_factor": 3.0, "peak_factor": 3.0, "profile": "flat"}

    profile = detect_profile(y)
    zero_ratio = float(np.mean(y == 0))

    recent = y[-m:] if y.size >= m else y
    recent = recent[np.isfinite(recent)]
    if recent.size == 0:
        return {"spike_factor": 3.0, "peak_factor": 3.0, "profile": profile}

    # robust stats
    recent_pos = recent[recent >= 0]
    recent_mean = float(np.mean(recent_pos)) if recent_pos.size > 0 else float(np.mean(recent))
    recent_std  = float(np.std(recent))
    cv = (recent_std / recent_mean) if recent_mean > 0 else 1.5  # treat ~0 mean as "high variability"
    slope = float(np.polyfit(np.arange(y.size), y, 1)[0]) if y.size >= 2 else 0.0

    # Base factors scale with variability; then we tweak by profile.
    spike = np.clip(2.0 + 1.2 * cv, 1.8, 5.0)   # guard on forecast average vs recent mean
    peak  = np.clip(1.5 + 1.0 * cv, 1.5, 6.0)   # guard on any single-period spike vs recent high

    if profile == 'flat':
        spike = min(spike, 2.5);  peak = min(peak, 2.5)
    elif profile == 'seasonal':
        spike = max(spike, 2.6);  peak = max(peak, 3.0)
    elif profile == 'trending' and slope > 0:
        spike = max(spike, 3.2);  peak = max(peak, 3.8)
    elif profile == 'declining':
        spike = min(spike, 2.2);  peak = min(peak, 2.6)

    # Heavily intermittent series: be stricter so we don’t explode on zeros
    if zero_ratio >= 0.60:
        spike = min(spike, 2.0)
        peak  = min(peak, 2.2)

    return {
        "spike_factor": float(spike),
        "peak_factor": float(peak),
        "profile": profile,
    }

def guardrail_or_fallback(
    y: np.ndarray,
    fcst: List[float],
    period: int = 12,
    spike_factor: Optional[float] = None,
    peak_factor: Optional[float] = None
) -> tuple[List[float], bool]:
    """
    1) If the forecast *average* is an implausible spike vs recent mean, fall back to Seasonal Naive.
    2) Cap any single-period forecast above a dynamic peak limit based on historical percentiles.
    Returns (final_forecast_list, fallback_used_bool).
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(fcst, dtype=float)
    f[~np.isfinite(f)] = 0.0

    recent = y[-period:] if len(y) >= period else y
    recent = recent[np.isfinite(recent)]
    if recent.size == 0:
        return f.tolist(), False

    # Dynamic spike threshold from series variability
    cv = (np.std(recent) / np.mean(recent)) if np.mean(recent) > 0 else 1.0
    if spike_factor is None:
        spike_factor = float(np.clip(2.5 + 2.0 * cv, 2.5, 6.0))

    recent_pos = recent[recent >= 0]
    base_mean = float(np.mean(recent_pos)) if recent_pos.size else float(np.mean(recent))

    # 1) Mean-spike fallback
    if base_mean > 0 and float(np.mean(f)) > spike_factor * base_mean:
        sn = SeasonalNaive(y, period=period)
        return sn.forecast(len(f)), True

    # 2) Peak cap per horizon (keeps shape but trims absurd highs)
    if peak_factor is None:
        peak_factor = 1.6  # conservative; adjust if needed

    hist = y[np.isfinite(y)]
    p98 = float(np.quantile(hist, 0.98)) if hist.size else 0.0
    cap = peak_factor * p98 if p98 > 0 else None
    if cap is not None and np.isfinite(cap) and cap > 0:
        f = np.minimum(f, cap)

    return f.tolist(), False

def _cv_mape(series_arr, factory):
    """
    Cross-validation error using WMAPE (0..1).
    Uses the SAME horizon as your validation/forecast window.
    """
    arr = np.asarray(series_arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)

    horizon = int(VALIDATION_PERIODS)
    folds = 3
    n = int(arr.size)
    if n <= horizon + folds:
        return float("inf")

    starts = [n - (folds - k) * horizon - horizon for k in range(folds)]
    errs = []
    for s in starts:
        train = arr[:s]
        test  = arr[s:s + horizon]
        if test.size == 0:
            continue
        try:
            model = factory(train)
            fc = np.asarray(model.forecast(len(test)), dtype=np.float64)
        except Exception:
            return float("inf")

        # WMAPE fold error
        denom = float(np.sum(np.abs(test)))
        if denom <= 1e-9:
            err = 0.0 if float(np.sum(np.abs(fc - test))) <= 1e-9 else float("inf")
        else:
            err = float(np.sum(np.abs(fc - test)) / denom)
        errs.append(err)

    return float(np.mean(errs)) if errs else float("inf")

def _rolling_origin_error(y, make_model_fn, horizon=4, folds=3, metric="wmape", eps=1e-9):
    """
    Rolling-origin backtest: fit on expanding windows, forecast `horizon`, score last-`horizon` each fold.
    metric: 'wmape', 'smape', 'mae', or 'mase' (with m=12 baseline).
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)
    n = len(y)
    if n <= horizon + folds:
        return float("inf")
    starts = [n - (folds - k) * horizon - horizon for k in range(folds)]
    errs = []
    for s in starts:
        train = y[:s]
        test = y[s:s + horizon]
        try:
            model = make_model_fn(train)
            fc = np.asarray(model.forecast(len(test)), dtype=np.float64)
        except Exception:
            return float("inf")

        if metric == "wmape":
            denom = np.sum(np.abs(test)) + eps
            err = np.sum(np.abs(test - fc)) / denom
        elif metric == "smape":
            denom = (np.abs(test) + np.abs(fc)) + eps
            err = np.mean(2.0 * np.abs(test - fc) / denom)
        elif metric == "mae":
            err = float(np.mean(np.abs(test - fc)))
        elif metric == "mase":
            m = 12
            if n <= m + 1:
                return float("inf")
            denom = np.mean(np.abs(y[m:] - y[:-m])) + eps
            err = float(np.mean(np.abs(test - fc)) / denom)
        else:
            err = float("inf")
        errs.append(float(err))
    return float(np.mean(errs)) if errs else float("inf")

def _robust_caps(y, m=12):
    """
    Robust global caps from recent data: 5th–98th percentile over last ≤3 seasons.
    Keeps seasonal baselines from exploding on outliers.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)
    n = len(y)
    if n == 0:
        return 0.0, 0.0
    tail = y[-min(n, m * 3):]
    lo = float(np.percentile(tail, 5))
    hi = float(np.percentile(tail, 98))
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi

class SNaiveShrink:
    """
    Seasonal-Naïve with shrinkage to SES and robust clipping.
    Safer seasonal baseline than pure SNaive for short/noisy series.
    """
    def __init__(self, series, m=12, shrink=None):
        y = np.asarray(series, dtype=np.float64)
        self.y = np.nan_to_num(y, nan=0.0)
        self.m = int(m)
        n = len(self.y)
        # Fewer seasons -> more shrinkage
        seasons = max(0, n // self.m - 1)
        rel = min(1.0, seasons / 3.0)
        self.w = float(0.35 + 0.15 * rel) if shrink is None else float(shrink)

    def _seasonal_replay(self, steps):
        y, m = self.y, self.m
        n = len(y)
        if n < m:
            avg = float(np.mean(y)) if n > 0 else 0.0
            return [avg] * steps
        last_m = y[n - m:n]
        return [float(last_m[h % m]) for h in range(steps)]

    def forecast(self, steps):
        y, m = self.y, self.m
        n = len(y)
        if n == 0:
            return [0.0] * steps

        # SES component
        ses = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True, use_brute=True)
        ses_fc = np.asarray(ses.forecast(steps), dtype=np.float64)

        # Seasonal-naïve replay
        snaive_fc = np.asarray(self._seasonal_replay(steps), dtype=np.float64)

        # Shrinkage blend + robust caps
        fc = (1.0 - self.w) * ses_fc + self.w * snaive_fc
        lo, hi = _robust_caps(y, m=m)
        fc = np.clip(fc, lo, hi)
        return [float(max(0.0, v)) for v in fc]

class SimpleAverage:
    def __init__(self, series):
        self.series = series

    def forecast(self, steps):
        avg = np.mean(self.series)
        return [avg] * steps

class HoltWinters:
    def __init__(self, series):
        self.series = series
        self.model = ExponentialSmoothing(
            series,
            seasonal=HOLT_WINTERS_PARAMS['seasonal'],
            seasonal_periods=HOLT_WINTERS_PARAMS['seasonal_periods']
        ).fit()

    def forecast(self, steps):
        return self.model.forecast(steps).tolist()



class LinearTrend:
    def __init__(self, series):
        self.series = np.asarray(series, dtype=np.float64)

    def forecast(self, steps):
        y = self.series
        n = len(y)
        if n == 0:
            return [0.0] * steps
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, y, 1)
        forecast_x = np.arange(n, n + steps, dtype=np.float64)
        y_forecast = np.polyval(coeffs, forecast_x)
        return y_forecast.astype(float).tolist()

# === ROBUST FORECASTERS (added 10/5/25) =====================================
import math

class WinsorizedMeanSeasonal:
    """
    Per month-of-year, cap values at p-th percentile (per-position) and forecast
    with the winsorized mean of that position. Robust to spikes.
    """
    def __init__(self, series, m=12, p=90):
        self.m = int(m)
        self.p = float(p)
        self.y = np.asarray(getattr(series, "values", series), dtype=float)

    def forecast(self, steps):
        m = self.m
        y = self.y
        n = len(y)
        if (m <= 1) or (n < m):
            mu = np.nanmean(np.sort(y))
            return np.repeat(float(mu if np.isfinite(mu) else 0.0), steps)
        # month-of-year buckets
        means = []
        for k in range(m):
            vals = y[k:n:m]
            if vals.size == 0:
                means.append(float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0)
                continue
            pos = vals[vals > 0]
            cap = float(np.percentile(pos, self.p)) if pos.size else 0.0
            means.append(float(np.nanmean(np.minimum(vals, cap))))
        reps = int(math.ceil(steps / m))
        return np.tile(np.asarray(means, dtype=float), reps)[:steps]

class FlatLevel:
    """
    Flat monthly level from winsorized last-12 total (p). Good when shape is noisy
    but yearly total is stable.
    """
    def __init__(self, series, m=12, p=99):
        self.m = int(m)
        self.p = float(p)
        self.y = np.asarray(getattr(series, "values", series), dtype=float)

    def forecast(self, steps):
        y = self.y
        m = self.m
        pos = y[y > 0]
        cap = float(np.percentile(pos, self.p)) if pos.size else 0.0
        last12 = y[-m:] if len(y) >= m else y
        lvl = float(np.nansum(np.minimum(last12, cap)))
        monthly = lvl / float(m if m > 0 else 1)
        return np.repeat(monthly, steps).astype(float)

class SeasonalizedFlatLevel:
    """
    Take the same total as FlatLevel(p) but distribute it by a robust seasonal profile
    (winsorized per-MOY means). Looks seasonal, same total, spike-proof.
    """
    def __init__(self, series, m=12, p_profile=90, p_total=99):
        import numpy as np, math
        self.np = np; self.math = math
        self.m = int(m)
        self.p_profile = float(p_profile)   # profile robustness (per-month winsor cap)
        self.p_total = float(p_total)       # same total logic as FlatLevel
        self.y = np.asarray(getattr(series, "values", series), dtype=float)

    def _winsor_cap(self, arr, p):
        x = self.np.asarray(arr, float)
        pos = x[x > 0]
        cap = float(self.np.percentile(pos, p)) if pos.size else 0.0
        return self.np.minimum(x, cap), cap

    def _profile(self, y):
        m = self.m; n = len(y)
        means = []
        for k in range(m):
            vals = y[k:n:m]
            if vals.size == 0:
                means.append(float(self.np.nanmean(y)) if n else 0.0)
                continue
            pos = vals[vals > 0]
            cap = float(self.np.percentile(pos, self.p_profile)) if pos.size else 0.0
            means.append(float(self.np.nanmean(self.np.minimum(vals, cap))))
        prof = self.np.asarray(means, float)
        s = float(prof.sum())
        return (prof / s) if s > 0 else self.np.ones(m, float) / float(m)

    def forecast(self, steps):
        np = self.np; m = self.m
        # total level from winsorized last-12 (like FlatLevel)
        _, cap = self._winsor_cap(self.y, self.p_total)
        last12 = self.y[-m:] if len(self.y) >= m else self.y
        total = float(np.nansum(np.minimum(last12, cap)))
        # seasonal proportions
        p = self._profile(self.y)
        reps = int(self.math.ceil(steps / m))
        shape = np.tile(p, reps)[:steps]
        # scale to the same total over 12 months
        shape_sum = float(shape[:m].sum()) if steps >= m else float(shape.sum())
        scale = (total / shape_sum) if shape_sum > 0 else 0.0
        return (shape * scale).astype(float)

class TrendCapped:
    """
    Linear trend on last K points after capping extremes at p. Floors negatives to 0.
    Good when seasonality is weak but there’s a drift.
    """
    def __init__(self, series, k=18, p=90):
        self.k = int(k)
        self.p = float(p)
        self.y = np.asarray(getattr(series, "values", series), dtype=float)

    def forecast(self, steps):
        y = self.y
        pos = y[y > 0]
        cap = float(np.percentile(pos, self.p)) if pos.size else 0.0
        yc = np.minimum(y, cap)
        n = len(yc)
        s = max(0, n - self.k)
        x = np.arange(s, n, dtype=float)
        ys = yc[s:n]
        if ys.size < 4:
            lvl = float(np.nanmean(ys)) if ys.size else 0.0
            return np.repeat(lvl, steps)
        X = np.vstack([np.ones_like(x), x]).T
        b0, b1 = np.linalg.lstsq(X, ys, rcond=None)[0]
        xf = np.arange(n, n + steps, dtype=float)
        fc = b0 + b1 * xf
        return np.clip(fc, 0.0, None)
# ====================================================================

class DampedTrend:
    def __init__(self, series, phi=0.9):
        self.series = np.array(series, dtype=np.float64)
        self.model = ExponentialSmoothing(
            self.series,
            trend="add",
            damped_trend=True,
            seasonal=None
        ).fit(optimized=True, use_brute=True)

    def forecast(self, steps):
        return self.model.forecast(steps).tolist()

def _smoothed_profile_int_forecast(series: np.ndarray,
                                   horizon: int = 12,
                                   winsor_p: float = 90.0,
                                   smooth_k: int = 3,
                                   eps: float = 1.0) -> list:
    """
    Smoothed Profile INT:
      1) Winsorize full history at p-th percentile (on positives only).
      2) Target total = mean(winsorized history) * horizon.
      3) Take last-`horizon` of the winsorized history and smooth with a k-window MA.
      4) Convert to weights (add eps), scale to target.
      5) Round to integers; adjust last month so the integer sum equals the target.
    Returns a list[int] of length `horizon`.
    """
    y = np.asarray(series, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0)
    if horizon <= 0:
        return []

    # Winsorize on positives to control spikes
    pos = y[y > 0]
    cap = float(np.percentile(pos, winsor_p)) if pos.size else 0.0
    yw = np.minimum(y, cap) if cap > 0 else y.copy()

    # Annual target from robust level
    target_total = float(np.mean(yw) * horizon)

    # Profile weights from last `horizon` months
    last = yw[-horizon:] if y.size >= horizon else yw
    if last.size == 0:
        return [0] * horizon

    # Simple centered moving average with edge padding
    pad = int(smooth_k // 2)
    x = np.pad(last, (pad, pad), mode='edge')
    sm = np.convolve(x, np.ones(smooth_k) / float(smooth_k), mode='valid')

    w = sm + float(eps)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        each = int(round(target_total / float(horizon))) if horizon > 0 else 0
        out = [each] * horizon
        diff = int(round(target_total)) - int(sum(out))
        out[-1] += diff
        return out

    w = w / w_sum
    fc = target_total * w

    # Integer rounding with sum preservation
    r = np.floor(fc + 0.5).astype(int)
    diff = int(round(target_total)) - int(np.sum(r))
    r[-1] += diff
    return [int(v) for v in r]

class SmoothedProfileINT:
    """
    Wrapper so SPI can be scored like other models.
    forecast() returns integers that sum to the winsorized target.
    """
    def __init__(self, series, winsor_p: float = 90.0, smooth_k: int = 3):
        self.series = np.asarray(series, dtype=np.float64)
        self.winsor_p = float(winsor_p)
        self.smooth_k = int(smooth_k)

    def forecast(self, steps: int):
        return _smoothed_profile_int_forecast(self.series,
                                              horizon=steps,
                                              winsor_p=self.winsor_p,
                                              smooth_k=self.smooth_k)

class SES:
    def __init__(self, series):
        self.series = np.array(series, dtype=np.float64)
        self.model = SimpleExpSmoothing(
            self.series, initialization_method="estimated"
        ).fit(optimized=True, use_brute=True)

    def forecast(self, steps):
        return self.model.forecast(steps).tolist()

class Theta:
    """
    Simple Theta method (bias-corrected). Good general-purpose trend baseline.
    """
    def __init__(self, series):
        y = np.asarray(series, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0)
        self.y = y

    def forecast(self, steps: int):
        y = self.y
        n = len(y)
        if n == 0:
            return [0.0] * steps
        x = np.arange(n, dtype=np.float64)

        # Level via SES (robust init)
        ses = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True, use_brute=True)
        level_fc = np.asarray(ses.forecast(steps), dtype=np.float64)

        # Linear trend on original series, **damped**
        slope, intercept = np.polyfit(x, y, 1)
        damp = 0.8 if n >= 36 else 0.6  # shorter series -> stronger damping
        slope *= damp
        trend_fc = np.asarray([intercept + slope * (n + h) for h in range(1, steps + 1)], dtype=np.float64)

        fc = 0.5 * (level_fc + trend_fc)

        # Robust floor/cap from recent data
        m = 12
        lo, hi = _robust_caps(y, m=m)
        floor = max(0.0, np.percentile(y[-min(n, m * 2):], 5))
        fc = np.clip(fc, floor * 0.2, hi)  # avoid hard drop-to-zero cliffs
        return [float(max(0.0, v)) for v in fc]

# === New candidate: ZeroDecay (obsolescence) === Added 9/6/25
class ZeroDecay:
    """
    Exponential decay toward zero with a configurable half-life.
    Intended ONLY for likely obsolescence / wind-down cases.
    - Level is the recent mean over the last up to 6 periods (non-negative).
    - Forecast_t = level * (0.5) ** (t / half_life)
    - Robustly clipped to recent upper bound via _robust_caps.
    """
    def __init__(self, series, half_life=4, winsor_p=95):
        y = np.asarray(series, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0)
        self.y = y
        self.half_life = float(max(1.0, half_life))
        # compute recent non-negative level
        tail = y[-6:] if y.size >= 6 else y
        self.level = float(max(0.0, np.nanmean(tail))) if tail.size else 0.0
        self.winsor_p = float(winsor_p)

    def forecast(self, steps: int):
        steps = int(steps)
        if steps <= 0:
            return []
        lvl = float(self.level)
        if not np.isfinite(lvl) or lvl < 1e-12:
            return [0.0] * steps
        # exponential decay
        t = np.arange(1, steps + 1, dtype=np.float64)
        fc = lvl * np.power(0.5, t / self.half_life)
        # robust cap based on recent distribution
        lo, hi = _robust_caps(self.y, m=12)
        fc = np.clip(fc, 0.0, hi)
        return [float(x) for x in fc]

class _CrostonSBAFixed:
    """
    Internal helper used only for CV: Croston-SBA with fixed alpha/beta.
    """
    def __init__(self, series, alpha, beta, winsor_p=90):
        y = np.asarray(series, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0)
        self.y = _winsorize_positive(y, p=winsor_p)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forecast(self, steps):
        y = self.y
        n = y.size
        if n == 0 or np.all(y == 0):
            return [0.0] * steps

        nonzero_idx = [i for i, v in enumerate(y) if v > 0]
        if not nonzero_idx:
            return [0.0] * steps

        z_hat = float(y[nonzero_idx[0]])
        p_hat = float(nonzero_idx[0] + 1)
        last = nonzero_idx[0]

        for idx in nonzero_idx[1:]:
            q_t = float(idx - last)
            z_hat = z_hat + self.alpha * (y[idx] - z_hat)
            p_hat = p_hat + self.beta * (q_t - p_hat)
            last = idx

        sba_factor = self.alpha / (2.0 - self.alpha)
        rate = sba_factor * (z_hat / max(p_hat, 1e-9))

        if _is_obsolete(y, tail=12):
            rate *= 0.5

        return [float(max(0.0, rate))] * steps

class AutoETS:
    """
    Minimal wrapper around statsmodels ETSModel with a small, sane grid
    (error ∈ {add, mul}, trend ∈ {None, add}, damped ∈ {False, True},
     seasonal ∈ {None, add}; seasonal_periods=m). Picks lowest AICc/AIC.
    Uses Box-Cox by default to stabilize variance. Non-negative forecasts.
    """
    def __init__(self, series, m=12, use_boxcox=True):
        self.y = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
        self.m = int(m)
        self.use_boxcox = bool(use_boxcox)
        self.res_ = None
        self.spec_ = None
        self._fit_best()

    def _fit_best(self):
        if ETSModel is None or self.y.size == 0:
            return
        combos = []
        for error in ("add", "mul"):
            for trend in (None, "add"):
                for damped_trend in (False, True):
                    for seasonal in (None, "add"):
                        combos.append((error, trend, damped_trend, seasonal))

        best_val = np.inf
        best_res = None
        best_spec = None
        for (error, trend, damped_trend, seasonal) in combos:
            try:
                sp = self.m if seasonal else None
                mod = ETSModel(
                    self.y,
                    error=error,
                    trend=trend,
                    damped_trend=damped_trend,
                    seasonal=seasonal,
                    seasonal_periods=sp,
                    initialization_method="estimated",
                    use_boxcox=self.use_boxcox,
                )
                res = mod.fit(maxiter=200, disp=False)
                aicc = getattr(res, "aicc", np.inf)
                val = aicc if np.isfinite(aicc) else getattr(res, "aic", np.inf)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_res = res
                    best_spec = (error, trend, damped_trend, seasonal)
            except Exception:
                continue
        self.res_ = best_res
        self.spec_ = best_spec

    def forecast(self, steps):
        steps = int(steps)
        if steps <= 0:
            return np.array([], dtype=float)
        if self.res_ is None:
            # very defensive fallback: recent average
            k = int(max(3, self.m))
            base = float(np.nanmean(self.y[-k:])) if self.y.size else 0.0
            return np.repeat(max(base, 0.0), steps)
        fc = np.asarray(self.res_.forecast(steps), dtype=float)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(fc, 0.0)

    def __str__(self):
        if self.spec_ is None:
            return "Auto-ETS"
        e, t, d, s = self.spec_
        return f"Auto-ETS(e={e}, t={t}, d={d}, s={s}, m={self.m})"
    
class CrostonSBA:
    """
    Public class used by the app: selects alpha/beta via CV on the *training* series,
    then uses those parameters to forecast on the full series.
    """
    def __init__(self, series, winsor_p=90):
        y = np.asarray(series, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0)
        self.y = _winsorize_positive(y, p=winsor_p)
        self.winsor_p = int(winsor_p)

        grid_a = [0.05, 0.1, 0.15, 0.2, 0.25]
        grid_b = [0.05, 0.1, 0.15, 0.2, 0.25]
        best_alpha, best_beta, best_err = 0.15, 0.15, float("inf")

        for a in grid_a:
            for b in grid_b:
                try:
                    err = _cv_mape(self.y, lambda tr: _CrostonSBAFixed(tr, a, b, self.winsor_p))
                except Exception:
                    err = float("inf")
                if err < best_err:
                    best_alpha, best_beta, best_err = a, b, err

        self.alpha = float(best_alpha)
        self.beta = float(best_beta)

    def forecast(self, steps):
        return _CrostonSBAFixed(self.y, self.alpha, self.beta, self.winsor_p).forecast(steps)

class ADIDA_SBA:
    """
    ADIDA (Aggregate-Disaggregate) with k=3:
      1) Sum series in non-overlapping 3-month blocks
      2) Forecast the aggregated series with Croston-SBA
      3) Disaggregate each block across its 3 months using SNaive month-of-year proportions
    """
    def __init__(self, series, k=3, m=12):
        self.series = np.asarray(series, dtype=float)
        self.k = int(k)
        self.m = int(m)

    def _aggregate_k(self, x):
        if self.k <= 1:
            return x
        n = x.size
        if n == 0:
            return x
        L = int(np.ceil(n / self.k))
        agg = np.zeros(L, dtype=float)
        for i in range(L):
            lo = i * self.k
            hi = min(n, (i + 1) * self.k)
            agg[i] = float(np.nansum(x[lo:hi]))
        return agg

    def _snaive_shape(self, H):
        m = self.m
        x = self.series
        if x.size >= m:
            base = x[-m:].copy()
        else:
            last = float(x[-1]) if x.size else 0.0
            base = np.full(m, last, dtype=float)
        base = np.nan_to_num(base, nan=0.0)
        shape = np.array([base[t % m] for t in range(H)], dtype=float)
        return np.maximum(shape, 0.0)

    def forecast(self, H):
        H = int(H)
        x = np.nan_to_num(self.series, nan=0.0)
        x[x < 0.0] = 0.0
        k = self.k

        # 1) aggregate by k
        xk = self._aggregate_k(x)

        # 2) forecast aggregated series with Croston-SBA
        agg_model = CrostonSBA(xk)
        Hk = int(np.ceil(H / k))
        fk = np.asarray(agg_model.forecast(Hk), dtype=float)
        fk = np.maximum(fk, 0.0)

        # 3) disaggregate by SNaive month-of-year proportions
        shape = self._snaive_shape(H)
        yhat = np.zeros(H, dtype=float)
        for i in range(Hk):
            block_lo = i * k
            block_hi = min(H, (i + 1) * k)
            block = shape[block_lo:block_hi]
            s = float(block.sum())
            if s <= 0.0:
                yhat[block_lo:block_hi] = fk[i] / (block_hi - block_lo)
            else:
                yhat[block_lo:block_hi] = fk[i] * (block / s)
        return yhat

class SeasonalNaive:
    """Seasonal Naive: y_{t+h} = y_{t+h-m}, default monthly m=12."""
    def __init__(self, series, period: int = 12):
        self.y = pd.Series(series).astype(float).values
        self.m = int(period)

    def forecast(self, steps: int):
        if steps <= 0:
            return []
        if len(self.y) == 0:
            return [0.0] * steps
        if self.m <= 1:
            last = self.y[-1] if len(self.y) else 0.0
            return [float(last)] * steps
        # Use last full season if available; otherwise repeat the tail
        if len(self.y) >= self.m:
            tail = self.y[-self.m:]
        else:
            tail = self.y[-min(len(self.y), steps):]
        rep = int(np.ceil(steps / len(tail)))
        out = np.tile(tail, rep)[:steps].astype(float)
        out[~np.isfinite(out)] = 0.0
        return out.tolist()

def detect_outliers(y):
    """
    High-spike detector (positives only).
    Step 1 (Global): candidate spikes must clear a very high global bar:
        y > max(q3 + 3.5*IQR, 3*median_pos, p95)
    Step 2 (Local): candidate must also beat a local median+MAD bar AND be
        at least +100% above local level AND exceed a minimum absolute lift.
    This dramatically cuts small/annoying flags on zero-heavy items.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 6 or np.all(~np.isfinite(y)):
        return []

    finite = np.isfinite(y)
    pos_mask = (y > 0) & finite
    x = y[pos_mask]  # positives only

    if x.size < 4:
        return []  # too little positive signal to call spikes

    # Robust global stats on positives
    med = float(np.median(x))
    q1, q3 = np.percentile(x, [25, 75])
    iqr = float(q3 - q1) if q3 > q1 else 1.0
    p95 = float(np.percentile(x, 95))

    # ---- TUNING KNOBS (raise these to be stricter) -----------------------
    MED_MULT = 3.0              # require ≥ 3× series median
    IQR_MULT = 3.5              # high-side fence strength
    MAD_Z    = max(3.0, float(OUTLIER_STD_DEV_THRESHOLD))  # local MAD "z"
    MIN_REL  = 1.00             # ≥ +100% above local median
    MIN_ABS_FLOOR = 50.0        # never flag lifts smaller than this
    MIN_ABS_FRACTION_OF_P95 = 0.25  # also require ≥ 25% of p95 lift
    WIN = max(5, int(OUTLIER_WINDOW))
    # ---------------------------------------------------------------------

    global_upper = max(q3 + IQR_MULT * iqr, MED_MULT * med, p95)

    # Step 1: global candidates (positives that are truly large for this series)
    cand_idx = [i for i in np.where(pos_mask)[0] if y[i] > global_upper]
    if not cand_idx:
        return []

    # Step 2: local confirmation (median/MAD + relative + absolute lift)
    out = set()
    min_abs_lift = max(MIN_ABS_FLOOR, MIN_ABS_FRACTION_OF_P95 * p95, 2.0 * med)

    for i in cand_idx:
        lo = max(0, i - WIN)
        hi = min(n, i + WIN + 1)
        neigh = np.r_[y[lo:i], y[i+1:hi]]
        neigh = neigh[np.isfinite(neigh)]
        neigh = neigh[neigh > 0]  # compare against positive local level
        if neigh.size < 5:
            continue
        m = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - m)))
        if mad == 0:
            mad = 1.0
        local_bar = m + MAD_Z * 1.4826 * mad

        rel_ok = (y[i] >= m * (1.0 + MIN_REL))
        abs_ok = ((y[i] - m) >= min_abs_lift)

        if y[i] > max(local_bar, m) and rel_ok and abs_ok:
            out.add(i)

    return sorted(out)

def clean_outliers_with_local_median(y, indices, window=OUTLIER_WINDOW):
    """
    Replace each flagged point with the median of its nearby neighbors.
    This is our 'zero-weight' proxy so models aren't driven by those spikes.
    """
    arr = np.asarray(y, dtype=float).copy()
    if not indices:
        return arr
    n = len(arr)
    finite_all = np.isfinite(arr)
    series_med = float(np.median(arr[finite_all])) if finite_all.any() else 0.0

    for i in indices:
        left = max(0, i - window)
        right = min(n, i + window + 1)
        neigh = np.concatenate([arr[left:i], arr[i+1:right]])
        neigh = neigh[np.isfinite(neigh)]
        repl = float(np.median(neigh)) if neigh.size >= 2 else series_med
        arr[i] = repl
    return arr

class SeasonalARIMA:
    """
    Very small SARIMA grid for seasonal hints.
    Grid: (p,d,q) in {(0,1,1),(1,1,0)}, (P,D,Q) in {(0,1,1),(1,0,1),(0,0,0)} at seasonal period m.
    """
    def __init__(self, series, m=12):
        y = np.asarray(series, dtype=np.float64)
        self.y = np.nan_to_num(y, nan=0.0)
        self.m = int(m)
        self.model = None

    def _fit_best(self):
        orders = [(0,1,1), (1,1,0)]
        sorders = [(0,1,1), (1,0,1), (0,0,0)]
        best = None
        best_aic = float("inf")
        for (p,d,q) in orders:
            for (P,D,Q) in sorders:
                try:
                    mod = SARIMAX(self.y, order=(p,d,q),
                                  seasonal_order=(P,D,Q,self.m),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    res = mod.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best = res
                except Exception:
                    continue
        self.model = best

    def forecast(self, steps: int):
        if self.model is None:
            self._fit_best()
        if self.model is None:
            last = float(self.y[-1]) if len(self.y) else 0.0
            return [last] * steps
        fc = np.asarray(self.model.forecast(steps), dtype=np.float64)
        lo, hi = _robust_caps(self.y, m=self.m)
        fc = np.clip(fc, 0.0, hi)  # keep nonnegative, cap outliers
        return [float(v) for v in fc]

# Repeat for other patches, e.g., def apply_cap_moy_patch(...)
def handle_sparse_obsolete(series, forecast_length, VALIDATION_PERIODS):
    y_all = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
    y_all = np.nan_to_num(y_all, nan=0.0)
    near_zero = np.isclose(y_all, 0.0, atol=1e-8)
    zero_ratio = float(np.mean(near_zero))
    is_sparse, nonzero_cnt, zr = _is_sparse(y_all, min_nonzero=3, zero_ratio_thr=0.7)
    arr_lumpy = y_all.copy()
    pos = arr_lumpy[arr_lumpy > 0.0]
    cv2 = float(np.var(pos) / (np.mean(pos) ** 2)) if (pos.size > 0 and np.mean(pos) > 0.0) else 0.0
    tail = arr_lumpy[-24:] if arr_lumpy.size >= 24 else arr_lumpy
    pos_tail = tail[tail > 0.0]
    q1 = float(np.percentile(pos_tail, 25)) if pos_tail.size else 0.0
    q3 = float(np.percentile(pos_tail, 75)) if pos_tail.size else 0.0
    iqr = q3 - q1
    spike_thr = q3 + 1.5 * iqr
    spikes = int(np.sum(tail > spike_thr))
    ratio_spikes = float(spikes) / float(min(24, arr_lumpy.size))

    # Define lumpy flag locally (avoid NameError)
    # Lumpy = high variance among nonzeros OR frequent spikes in recent tail
    is_lumpy = False
    try:
        is_lumpy = (pos.size >= 6 and cv2 >= 0.60) or (ratio_spikes >= 0.10)
    except Exception:
        is_lumpy = False    

    candidate_model = None
    candidate_forecast = None
    candidate_mape = float('inf')

    if np.all(near_zero) or _is_obsolete(y_all):
        candidate_model = 'Sparse/Obsolete (zeros)'
        candidate_forecast = [0.0] * forecast_length
        candidate_mape = 0.0
    elif zero_ratio >= 0.90 and (np.count_nonzero(~near_zero) < 3):
        candidate_model = 'Sparse/Obsolete (near-zeros)'
        candidate_forecast = [0.0] * forecast_length
        candidate_mape = 0.0
    elif (is_sparse or is_lumpy) and (nonzero_cnt >= 6):
        try:
            train = series[:-VALIDATION_PERIODS] if len(series) >= VALIDATION_PERIODS else series
            test = series[-VALIDATION_PERIODS:] if len(series) >= VALIDATION_PERIODS else []
            sba_val = CrostonSBA(train).forecast(len(test)) if len(test) > 0 else []
            sba_mape = compute_mape(test, sba_val) if len(test) > 0 else float('inf')
            sba_mape = min(sba_mape, _cv_mape_cached('CrostonSBA', lambda tr: CrostonSBA(tr)))
            sa_val = SimpleAverage(train).forecast(len(test)) if len(test) > 0 else []
            sa_mape = compute_mape(test, sa_val) if len(test) > 0 else float('inf')
            sa_mape = min(sa_mape, _cv_mape_cached('SimpleAverage', lambda tr: SimpleAverage(tr)))
            if sba_mape < sa_mape:
                candidate_mape = sba_mape
                candidate_model = 'Croston-SBA'
                candidate_forecast = CrostonSBA(series).forecast(forecast_length)
            else:
                candidate_mape = sa_mape
                candidate_model = 'SimpleAverage'
                candidate_forecast = SimpleAverage(series).forecast(forecast_length)
        except Exception as e:
            logger.warning(f"Sparse model failed: {e}")

    return candidate_model, candidate_forecast, candidate_mape

def run_full_forecast(df, period_type="Monthly", start_date=None, outlier_detection=True, zero_weight_outliers_all=False, progress_callback=None, **kwargs):
    if 'detect_outliers' in kwargs and kwargs['detect_outliers'] is not None:
        outlier_detection = bool(kwargs['detect_outliers'])
    results = []
    outlier_dict = {}
    period_columns = [col for col in df.columns if col.startswith("H")]

    if len(period_columns) < MIN_HISTORY_REQUIRED:
        return {"status": "error", "data": [], "outliers": {}, "message": f"Insufficient history: {len(period_columns)} periods provided, {MIN_HISTORY_REQUIRED} required."}

    forecast_length = DEFAULT_FORECAST_LENGTH
    if start_date:
        if period_type == "Monthly":
            forecast_dates = pd.date_range(start=start_date, periods=len(period_columns) + forecast_length, freq="MS")[-forecast_length:]
        elif period_type == "Weekly":
            forecast_dates = pd.date_range(start=start_date, periods=len(period_columns) + forecast_length, freq="W-MON")[-forecast_length:]
        elif period_type == "Daily":
            forecast_dates = pd.date_range(start=start_date, periods=len(period_columns) + forecast_length, freq="D")[-forecast_length:]
        else:
            forecast_dates = [f"Period {i+1}" for i in range(len(period_columns), len(period_columns) + forecast_length)]
        forecast_labels = [d.strftime("%b %Y") if not isinstance(d, str) else d for d in forecast_dates]
    else:
        forecast_labels = [f"Period {i+1}" for i in range(1, forecast_length + 1)]

    # --- metrics first (these are needed by PATCH M right away) ---
    def compute_smape(y_true, y_pred):
        """
        Returns sMAPE as a percentage (0..200). Kept for reporting/diagnostics.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(valid):
            return float('inf')
        yt = y_true[valid]
        yp = y_pred[valid]
        denom = (np.abs(yt) + np.abs(yp))
        denom[denom == 0] = 1.0
        return 100.0 * np.mean(2.0 * np.abs(yp - yt) / denom)

    def compute_wmape(y_true, y_pred, eps: float = 1e-9):
        """
        Unifies all selection scoring to WMAPE on [0..1]:
            sum(|yhat - y|) / sum(|y|)
        Degenerate case handling:
          - If sum(|y|) ~ 0 and predictions also ~ 0 -> 0.0 (perfect)
          - If sum(|y|) ~ 0 but predictions not ~ 0 -> inf (penalize)
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(valid):
            return float('inf')
        yt = y_true[valid]
        yp = y_pred[valid]
        abs_err = float(np.sum(np.abs(yp - yt)))
        scale   = float(np.sum(np.abs(yt)))
        if scale <= eps:
            return float('inf')
        return abs_err / scale

    # Back-compat alias:
    # From now on, treat compute_mape() as WMAPE (0..1) so all selection logic
    # uses the same metric. "MAPE (Validation)" in your UI remains % by *100.
    def compute_mape(y_true, y_pred):
        return compute_wmape(y_true, y_pred)

    def _metrics_valid_for_series(y_full_arr, last_k=12, min_total=50.0, max_zero_ratio=0.60):
        """
        Returns False when WMAPE/holdout metrics are not meaningful.
        """
        y = np.asarray(y_full_arr, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0)
        tail = y[-last_k:] if y.size >= last_k else y
        total = float(np.sum(np.abs(tail)))
        if total < float(min_total):
            return False
        zr = float(np.mean(tail == 0.0)) if tail.size else 1.0
        if zr > float(max_zero_ratio):
            return False
        return True

    # --- ARIMA wrapper must be defined before we reference it below ---
    class ARIMAForecaster:
        def __init__(self, series, order=(1, 1, 1)):
            self.model = ARIMA(series, order=order).fit()
        def forecast(self, steps):
            return self.model.forecast(steps).tolist()

    total_items = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        item = row["Item Number"]
        series = row[period_columns].values.astype(float)
        raw_last12_sum = float(np.nansum(np.abs(series[-12:]))) if len(series) >= 12 else float(np.nansum(np.abs(series))) #Added 10/9/25

        # === Outlier detection & (optional) global zero-weight cleaning ===
        outlier_indices = []
        if outlier_detection:
            outlier_indices = detect_outliers(series)

        # Keep UX list (H1.. labels) for the sidebar/report
        if outlier_indices:
            outlier_dict[item] = [f"H{i+1}" for i in outlier_indices]

        # Apply the ALL switch: replace flagged points with local median
        if zero_weight_outliers_all and outlier_indices:
            series = clean_outliers_with_local_median(series, outlier_indices)

        # Convert to pandas Series with your existing index
        series = pd.Series(series, index=period_columns)
        # === Per-item CV cache (speed only; no behavior changes) ===
        _cv_cache = {}

        # Reusable array form of this item's full series
        y_full = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
        y_full = np.nan_to_num(y_full, nan=0.0)
        metrics_ok = _metrics_valid_for_series(y_full, last_k=12, min_total=50.0, max_zero_ratio=0.60)

        # Cache common stats early
        mu6 = float(np.mean(y_full[-min(6, y_full.size):])) if y_full.size > 0 else 0.0
        zero_ratio = float(np.mean(y_full == 0)) if y_full.size > 0 else 0.0

        def _cv_mape_cached(tag, factory, arr=None):
            """
            Cache for _cv_mape(...). 'tag' should identify the model & key params.
            """
            key = ("cv", str(tag))
            if key in _cv_cache:
                return _cv_cache[key]
            a = y_full if arr is None else np.asarray(arr, dtype=float)
            err = _cv_mape(a, factory)
            _cv_cache[key] = err
            return err

        def _cv_ro_cached(tag, factory):
            """
            Cache for rolling-origin CV using your standard horizon/folds/metric.
            """
            key = ("ro", str(tag), int(VALIDATION_PERIODS), 3, "wmape")
            if key in _cv_cache:
                return _cv_cache[key]
            err = _rolling_origin_error(y_full, factory, horizon=VALIDATION_PERIODS, folds=3, metric="wmape")
            _cv_cache[key] = err
            return err        
        
        profile = detect_profile(series)

        train = series[:-VALIDATION_PERIODS] if len(series) >= VALIDATION_PERIODS else series
        test = series[-VALIDATION_PERIODS:] if len(series) >= VALIDATION_PERIODS else []

        best_mape = float('inf')
        best_model = None
        best_forecast = None
        lock_model = False
        guard_locked = False  # Add this to init early, avoiding undefined in guards
        level_shift_flag = False  # initialize per item
        baseline_err = float('inf')  # Add this to init early, used in patches/guards
        baseline_name = None  # Add this to init early, used in patches/guards

        # we need these flags/counts early (PATCH M and later use them)
        is_sparse, nonzero_cnt, zr = _is_sparse(series.values, min_nonzero=3, zero_ratio_thr=0.7)

        # ============================================================
        # ZERO-RULE MUST ONLY APPLY WHEN ALL HISTORY IS ZERO (ship rule)
        # ============================================================
        # If ANY nonzero exists, we must NOT use Zero-Rule.
        try:
            _y_all = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
        except Exception:
            _y_all = np.asarray(series, dtype=float)
        _y_all = np.nan_to_num(_y_all, nan=0.0)

        if np.all(_y_all == 0.0):
            best_model = "Zero-Rule"
            best_forecast = [0.0] * forecast_length
            best_mape = 0.0
            lock_model = True

        # ============================================================
        # EXTREMELY SPARSE (not all-zero): use a simple intermittent baseline
        # Example: 34 zeros + 2 ones -> small constant forecast, not trend/ARIMA
        # ============================================================
        if not lock_model:
            try:
                n_hist = int(_y_all.size)
                nz = _y_all[_y_all > 0.0]
                nz_cnt = int(nz.size)
                zero_ratio_local = float(np.mean(_y_all == 0.0)) if n_hist > 0 else 1.0

                # Trigger band: very sparse, but not all-zero
                # - lots of zeros
                # - only a few nonzero observations
                if (n_hist >= 18) and (zero_ratio_local >= 0.75) and (1 <= nz_cnt <= 4):
                    p = float(nz_cnt) / float(max(n_hist, 1))
                    size = float(np.mean(nz)) if nz_cnt > 0 else 0.0
                    baseline_level = p * size

                    # Score it on validation window (same metric)
                    horizon_val = len(test) if len(test) > 0 else VALIDATION_PERIODS
                    base_val = [baseline_level] * int(horizon_val)
                    base_err = compute_mape(test, base_val) if len(test) > 0 else float("inf")

                    # Only adopt if it beats whatever is currently best (or if best is empty)
                    if (best_forecast is None) or (base_err < best_mape):
                        best_model = "IntermittentBaseline(p×size)"
                        best_forecast = [baseline_level] * int(forecast_length)
                        best_mape = base_err
            except Exception:
                pass

        # Lumpy (high-variance) detection even when zeros are low
        arr_lumpy = np.asarray(series.values, dtype=np.float64)
        arr_lumpy = np.nan_to_num(arr_lumpy, nan=0.0)
        pos = arr_lumpy[arr_lumpy > 0.0]

        cv2 = float(np.var(pos) / (np.mean(pos) ** 2)) if (pos.size > 0 and np.mean(pos) > 0.0) else 0.0

        # spike ratio over last 24 observations (any values)
        tail = arr_lumpy[-24:] if arr_lumpy.size >= 24 else arr_lumpy
        if tail.size > 0:
            pos_tail = tail[tail > 0.0]
            if pos_tail.size > 0:
                q1 = float(np.percentile(pos_tail, 25))
                q3 = float(np.percentile(pos_tail, 75))
                iqr = q3 - q1
                spike_thr = q3 + 1.5 * iqr
                spikes = int(np.sum(tail > spike_thr))
                ratio_spikes = float(spikes) / float(min(24, arr_lumpy.size))
            else:
                ratio_spikes = 0.0
        else:
            ratio_spikes = 0.0

        # ---- PATCH M (now computes candidate but doesn't lock early) ---------------------------
        sparse_model, sparse_forecast, sparse_mape = handle_sparse_obsolete(series, forecast_length, VALIDATION_PERIODS)
        if metrics_ok and sparse_model is not None and sparse_mape < best_mape:
            best_mape = sparse_mape
            best_model = sparse_model
            best_forecast = sparse_forecast

        if not lock_model:
            # SES for flat series with moderate sparsity
            try:
                if profile == 'flat' and 0.3 < zero_ratio < 0.5:
                    ses_model = SES(train)
                    ses_forecast = ses_model.forecast(VALIDATION_PERIODS)
                    ses_mape = compute_mape(test, ses_forecast) if len(test) > 0 else float('inf')
                    ses_mape = min(ses_mape, _cv_mape_cached('SES', lambda tr: SES(tr)))
                    if ses_mape < best_mape:
                        best_mape = ses_mape
                        best_model = 'SES'
                        best_forecast = SES(series).forecast(forecast_length)
            except Exception as e:
                logger.warning(f"Model SES failed: {e}")

            # Holt-Winters if seasonal hint
            if not lock_model:
                try:
                    if _has_seasonality(series.values.astype(float), period=HOLT_WINTERS_PARAMS['seasonal_periods']):
                        hw_model = HoltWinters(train)
                        hw_forecast = hw_model.forecast(VALIDATION_PERIODS)
                        hw_mape = compute_mape(test, hw_forecast) if len(test) > 0 else float('inf')
                        hw_mape = min(hw_mape, _cv_mape_cached('Holt-Winters', lambda tr: HoltWinters(tr)))
                        if hw_mape < best_mape:
                            best_mape = hw_mape
                            best_model = 'Holt-Winters'
                            best_forecast = HoltWinters(series).forecast(forecast_length)
                except Exception as e:
                    logger.warning(f"Model Holt-Winters failed for item {item}: {str(e)}")
            
            # ---- Damped Trend candidate (gated to avoid low-level collapse) ----
            if not lock_model:
                try:
                    # ensure y_full exists locally
                    y_full = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
                    y_full = np.nan_to_num(y_full, nan=0.0)

                    if profile in ('trending', 'declining') and zero_ratio < 0.50:
                        feats = _robust_trend_features(y_full)
                        strength = feats["slope_abs_over_level"]   # |slope| / level
                        r2 = feats["r2"]
                        mu6 = feats["mu6"]

                        # validation horizon forecast
                        horizon = len(test) if len(test) else VALIDATION_PERIODS
                        dt_model = DampedTrend(train)
                        dt_val = dt_model.forecast(horizon)
                        dt_mape_val = compute_mape(test, dt_val) if len(test) else float('inf')

                        # same CV you already use
                        dt_cv = _cv_mape_cached('DampedTrend', lambda tr: DampedTrend(tr), arr=y_full)
                        dt_err = min(dt_mape_val, dt_cv)

                        # recent-level guard: don't collapse below 60% of recent mean
                        dt_last = float(dt_val[-1]) if len(dt_val) else 0.0
                        recent_guard_ok = (mu6 <= 0.0) or (dt_last >= 0.60 * mu6)

                        # require real trend OR a clear win
                        real_trend = (strength >= 0.04) and (r2 >= 0.25)
                        clear_win = (dt_err + 0.05) <= best_mape  # beats current best by ≥5% absolute

                        if recent_guard_ok and (real_trend or clear_win) and (dt_err < best_mape):
                            best_mape = dt_err
                            best_model = 'Damped Trend'
                            best_forecast = DampedTrend(series).forecast(forecast_length)
                except Exception:
                    pass

            # ---- PATCH P-3: SARIMA only if seasonality is present -------------------
            if not lock_model:
                try:
                    m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
                    y_full = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
                    y_full = np.nan_to_num(y_full, nan=0.0)

                    # reuse the module-level _has_seasonality to avoid shadowing
                    if _has_seasonality(y_full, period=m) and len(series) >= (2 * m):
                        sarima_val = SeasonalARIMA(train, m=m).forecast(len(test) if len(test) else VALIDATION_PERIODS)
                        sarima_mape = compute_mape(test, sarima_val) if len(test) else float('inf')

                        if sarima_mape < best_mape:
                            best_mape = sarima_mape
                            best_model = f'SARIMA(m={m})'
                            best_forecast = SeasonalARIMA(series, m=m).forecast(forecast_length)
                except Exception:
                    pass

            # ARIMA (single, lightweight sweep; unified WMAPE scoring)
            try:
                if metrics_ok and profile in ['trending', 'declining'] and zero_ratio < 0.03:
                    # Smaller grid based on what performed best in your set
                    orders = [(0,1,1), (1,1,0), (1,1,1)]
                    if len(series) >= 18:
                        orders += [(2,1,1), (2,1,2)]
                    horizon = len(test) if len(test) else VALIDATION_PERIODS

                    best_ar_err = float('inf')
                    best_ar_order = None
                    best_ar_forecast = None

                    for order in orders:
                        try:
                            # Validation window (same horizon you use elsewhere)
                            ar_model = ARIMA(train, order=order).fit()
                            ar_val = ar_model.forecast(horizon)
                            val_err = compute_mape(test, ar_val) if len(test) else float('inf')

                            # Cross-validation (uses WMAPE via _cv_mape from Patch 1)
                            cv_err = _cv_mape(y_full, lambda tr: ARIMAForecaster(tr, order=order))

                            # Conservative score = min(validation, CV)
                            ar_err = min(val_err, cv_err)

                            if ar_err < best_ar_err:
                                best_ar_err = ar_err
                                best_ar_order = order
                                
                        except Exception:
                            continue

                    if best_ar_order is not None and best_ar_err < best_mape:
                        best_mape = best_ar_err
                        best_model = f'ARIMA({best_ar_order[0]},{best_ar_order[1]},{best_ar_order[2]})'
                        # Ensure list[float]
                        # Fit on FULL series for the production forecast (not train)
                        ar_full = ARIMA(series, order=best_ar_order).fit()
                        full_fc = ar_full.forecast(forecast_length)
                        best_forecast = list(np.asarray(full_fc, dtype=float))
            except Exception:
                pass

            # Simple Average contender
            try:
                sa_model = SimpleAverage(train)
                sa_forecast = sa_model.forecast(VALIDATION_PERIODS)
                sa_mape = compute_mape(test, sa_forecast) if len(test) > 0 else float('inf')
                sa_mape = min(sa_mape, _cv_mape_cached('SimpleAverage', lambda tr: SimpleAverage(tr)))
                if sa_mape < best_mape:
                    best_mape = sa_mape
                    best_model = 'Simple Average'
                    best_forecast = SimpleAverage(series).forecast(forecast_length)
            except:
                pass

            # ---- Smoothed Profile INT candidate (winsorized total + shaped profile) ----
            try:
                # Gate to where it helps most:
                # - not heavy intermittent (avoid if zeros dominate),
                # - not strongly seasonal (you already have seasonal candidates),
                # - lumpy/erratic recent behavior where flat lines underperform.
                y_arr = np.asarray(series.values, dtype=float)
                y_arr = np.nan_to_num(y_arr, nan=0.0)

                last12 = y_arr[-12:] if y_arr.size >= 12 else y_arr
                # lumpy-ness proxy: p95 / median among positives in last12
                pos = last12[last12 > 0.0]
                if pos.size >= 3:
                    p50 = float(np.percentile(pos, 50))
                    p95 = float(np.percentile(pos, 95))
                    lumpy_ratio = (p95 / max(p50, 1e-9)) if p50 > 0 else 0.0
                else:
                    lumpy_ratio = 0.0

                season_hint = _has_seasonality(y_arr, period=HOLT_WINTERS_PARAMS['seasonal_periods'])
                allow_spi = (zero_ratio < 0.60) and (not season_hint) and (lumpy_ratio >= 2.0)

                if allow_spi:
                    # Validation forecast for scoring uses the same integer logic
                    spi_model = SmoothedProfileINT(train, winsor_p=90.0, smooth_k=3)
                    spi_val = spi_model.forecast(len(test)) if len(test) > 0 else []
                    spi_mape = compute_mape(test, spi_val) if len(test) > 0 else float('inf')

                    # Cross-validation as a safety net (same helper you use elsewhere)
                    spi_cv = _cv_mape_cached('SmoothedProfileINT', lambda tr: SmoothedProfileINT(tr, winsor_p=90.0, smooth_k=3))

                    spi_err = min(spi_mape, spi_cv)

                    # Require a clear improvement over Simple Average (if SA currently leads),
                    # otherwise allow SPI to become best if it beats the current best_mape.
                    if spi_err < best_mape - 0.01:
                        best_mape = spi_err
                        best_model = 'Smoothed Profile INT'
                        best_forecast = SmoothedProfileINT(series, winsor_p=90.0, smooth_k=3).forecast(forecast_length)
            except Exception:
                pass

            # ---- Auto-ETS (small, safe grid; selected by the same CV WMAPE) ----
            try:
                if not lock_model:
                    # local copy of series as float
                    try:
                        arr = np.asarray(series.values, dtype=float)
                    except Exception:
                        arr = np.asarray(series, dtype=float)
                    arr = np.nan_to_num(arr, nan=0.0)

                    n = arr.size
                    if n >= 18:  # need a bit of history
                        # skip if highly intermittent (ETS is poor there)
                        zero_share_local = float(np.sum(arr <= 0.0)) / max(1, n)
                        if zero_share_local <= 0.60:
                            m = int(HOLT_WINTERS_PARAMS.get("seasonal_periods", 12))

                            class _ETSModel:
                                def __init__(self, y, trend, damped, seasonal, sp):
                                    _y = np.asarray(y, dtype=float)
                                    self._fit = ExponentialSmoothing(
                                        _y,
                                        trend=trend,
                                        damped_trend=(damped if trend else False),
                                        seasonal=seasonal,
                                        seasonal_periods=sp,
                                        initialization_method="estimated",
                                    ).fit(optimized=True, use_brute=False)

                                def forecast(self, steps: int):
                                    fc = self._fit.forecast(int(steps))
                                    return np.clip(np.asarray(fc, dtype=float), 0.0, np.inf)

                            # small, safe grid (additive only; handles zeros)
                            configs = [
                                ("add", False, "add"),  # ETS(A,A,A)
                                ("add", True,  "add"),  # ETS(A,Ad,A)
                                (None, False,  "add"),  # ETS(A,N,A) ~ level+seasonal
                            ]

                            best_ets_mape = float("inf")
                            best_cfg = None
                            for trnd, damp, seas in configs:
                                tag = f"ETS(trend={trnd or 'none'}{'-damped' if (damp and trnd) else ''}, seasonal={seas})"
                                mape = _cv_mape_cached(
                                    tag,
                                    lambda y, t=trnd, d=damp, s=seas, mm=m: _ETSModel(y, t, d, s, mm),
                                )
                                if mape < best_ets_mape:
                                    best_ets_mape = mape
                                    best_cfg = (trnd, damp, seas)

                            if (best_cfg is not None) and (best_ets_mape + 1e-12 < best_mape):
                                mdl = _ETSModel(arr, best_cfg[0], best_cfg[1], best_cfg[2], m)
                                fc = mdl.forecast(forecast_length)
                                best_model = f"ETS(trend={best_cfg[0] or 'none'}{'-damped' if (best_cfg[1] and best_cfg[0]) else ''}, seasonal={best_cfg[2]})"
                                best_forecast = list(np.asarray(fc, dtype=float))
                                best_mape = best_ets_mape
                                lock_model = True
            except Exception:
                pass

            # Intermittent again (guarded)
            try:
                # --- lumpy detection (lets ADIDA+SBA enter the race even when values are tiny, not literal zeros)
                try:
                    arr = np.asarray(series.values, dtype=float)
                except Exception:
                    arr = np.asarray(series, dtype=float)
                arr = np.nan_to_num(arr, nan=0.0)
                nz = arr[arr > 0.0]
                nz_mean = float(np.mean(nz)) if nz.size else 0.0
                nz_std  = float(np.std(nz)) if nz.size else 0.0
                cov_nonzero = (nz_std / nz_mean) if nz_mean > 0.0 else 0.0
                med_nz = float(np.median(nz)) if nz.size else 0.0
                low_thr = 0.2 * med_nz if med_nz > 0.0 else 0.0
                low_share = float(np.mean(arr <= low_thr)) if arr.size else 0.0
                
                if (nonzero_cnt >= 6):
                    sba_model = CrostonSBA(train)
                    sba_val = sba_model.forecast(VALIDATION_PERIODS) if len(test) > 0 else []
                    sba_mape = compute_mape(test, sba_val) if len(test) > 0 else float('inf')
                    sba_mape = min(sba_mape, _cv_mape_cached('CrostonSBA', lambda tr: CrostonSBA(tr)))

                    sa_val = SimpleAverage(train).forecast(VALIDATION_PERIODS) if len(test) > 0 else []
                    sa_mape = compute_mape(test, sa_val) if len(test) > 0 else float('inf')
                    sa_mape = min(sa_mape, _cv_mape_cached('SimpleAverage', lambda tr: SimpleAverage(tr)))

                    if sba_mape < sa_mape and sba_mape < best_mape:
                        best_mape = sba_mape
                        best_model = 'Croston-SBA'
                        best_forecast = CrostonSBA(series).forecast(forecast_length)
            except Exception:
                pass

            # Intermittent again (guarded) — ADIDA-3 + Croston-SBA candidate
            try:
                if (nonzero_cnt >= 6):

                    class _ADIDA3_SBA:
                        """Aggregate by K=3 (non-overlapping), fit Croston-SBA on the 3-month blocks,
                        then de-aggregate by equal split. This is a standard ADIDA variant."""
                        def __init__(self, y):
                            self.y = np.asarray(y, dtype=float)

                        def forecast(self, h: int):
                            y = np.nan_to_num(self.y, nan=0.0)
                            K = 3
                            n = y.size
                            # number of complete 3-month blocks
                            blocks = n // K
                            if blocks <= 1:
                                # not enough blocks → fall back to plain Croston-SBA
                                return CrostonSBA(y).forecast(h)

                            # aggregate by summing each 3-month block
                            starts = np.arange(0, blocks * K, K, dtype=int)
                            agg = np.add.reduceat(y, starts)

                            # fit Croston-SBA on aggregated series
                            model = CrostonSBA(agg)

                            # forecast the number of 3-month blocks needed, ceil(h/3) without math.ceil
                            steps = int((h + K - 1) // K)
                            agg_fc = np.asarray(model.forecast(steps), dtype=float)

                            # de-aggregate by equal split across the 3 months
                            monthly = []
                            for val in agg_fc:
                                v = float(val) / K
                                monthly.extend([v, v, v])
                            return monthly[:h]

                    # Holdout (test) WAPE
                    adida_val = _ADIDA3_SBA(train).forecast(VALIDATION_PERIODS) if len(test) > 0 else []
                    adida_mape = compute_mape(test, adida_val) if len(test) > 0 else float('inf')

                    # Cross-validated MAPE via your existing CV helper
                    adida_mape = min(adida_mape, _cv_mape_cached('ADIDA3+SBA', lambda tr: _ADIDA3_SBA(tr)))

                    # Compete fairly: must beat SimpleAverage and current best
                    if adida_mape < sa_mape and adida_mape < best_mape:
                        best_mape   = adida_mape
                        best_model  = "ADIDA-3 + Croston-SBA"
                        best_forecast = _ADIDA3_SBA(series).forecast(forecast_length)

            except Exception:
                pass

        if best_forecast is None:
            best_model = 'Simple Average'
            best_forecast = SimpleAverage(series).forecast(forecast_length)
            best_mape = compute_mape(test, best_forecast[:len(test)]) if len(test) > 0 else 0.0

        # Baseline enforcement: chosen model must beat Theta and SNaiveShrink.
        try:
            horizon = VALIDATION_PERIODS
            m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
            theta_err = _cv_ro_cached('Theta', lambda tr: Theta(tr))
            sss_err   = _cv_ro_cached(f"SNaive-S(m={m})", lambda tr: SNaiveShrink(tr, m=m))

            if theta_err <= sss_err:
                if theta_err < best_mape:
                    best_mape = theta_err
                    best_model = 'Theta'
                    best_forecast = Theta(series).forecast(forecast_length)

            else:
                if sss_err < best_mape:
                    # prefer SNaive-S only if seasonality is actually present
                    try:
                        m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
                    except Exception:
                        m = 12
                    season_ok = False
                    try:
                        # crude seasonality check: Spearman correlation on last 24 vs 12-month lag
                        x = np.asarray(series.values[-(2*m):], dtype=float)
                        if x.size >= 2*m and np.isfinite(x).all() and spearmanr is not None:
                            r, _ = spearmanr(x[:-m], x[m:])
                            season_ok = (r is not None) and np.isfinite(r) and (r >= 0.30)
                    except Exception:
                        # fallback: allow if CV by month-of-year clearly > noise
                        try:
                            x = np.asarray(series.values[-(2*m):], dtype=float)
                            if x.size >= 2*m:
                                moy = [np.nanmean(x[i::m]) for i in range(m)]
                                mean = np.nanmean(moy)
                                cv_moy = (np.nanstd(moy) / mean) if mean else 0.0
                                season_ok = cv_moy >= 0.20
                        except Exception:
                            season_ok = False

                    if season_ok:
                        best_mape = sss_err
                        best_model = 'SNaive-S'
                        best_forecast = SNaiveShrink(series, m=m).forecast(forecast_length)
                    else:
                        best_mape = theta_err
                        best_model = 'Theta'
                        best_forecast = Theta(series).forecast(forecast_length)
        except Exception:
            pass

        # Profile-aware tie-breaker (WMAPE): conservative bias toward strong defaults
        # Goal: avoid switching away from baseline on flat/intermittent unless a clear gain;
        #       prefer SARIMA on seasonal unless challenger beats it clearly.
        try:
            # Strongest baseline on this item
            baseline_err = min(theta_err, sss_err)
            baseline_name = 'Theta' if theta_err <= sss_err else 'SNaive-S'

            # Flat / Intermittent: require ≥2.5 pts absolute WMAPE improvement to leave baseline
            if profile in ['flat', 'intermittent']:
                if not (best_mape + 0.050 <= baseline_err):
                    best_mape = baseline_err
                    best_model = baseline_name
                    best_forecast = (Theta(series).forecast(forecast_length)
                                     if baseline_name == 'Theta'
                                     else SNaiveShrink(series, m=m).forecast(forecast_length))

            # Seasonal: prefer SARIMA unless challenger beats it by ≥2.0 pts absolute WMAPE
            elif False and profile == 'seasonal':
                try:
                    sarima_cv = _cv_mape_cached(f"SARIMA(m={m})", lambda tr: SeasonalARIMA(tr, m=m), arr=y_full)
                except Exception:
                    sarima_cv = float('inf')
                if sarima_cv < float('inf'):
                    if not (best_mape + 0.020 <= sarima_cv):
                        best_mape = sarima_cv
                        best_model = 'SARIMA(m=12)'
                        best_forecast = SeasonalARIMA(series, m=m).forecast(forecast_length)

            # Other: require at least 1.5 pts WMAPE win over baseline to switch
            elif profile == 'other':
                if not (best_mape + 0.015 <= baseline_err):
                    best_mape = baseline_err
                    best_model = baseline_name
                    best_forecast = (Theta(series).forecast(forecast_length)
                                     if baseline_name == 'Theta'
                                     else SNaiveShrink(series, m=m).forecast(forecast_length))

            # Other profiles: leave as-is (winner-takes-all via unified WMAPE)

        except Exception:
            pass

        # PATCH: Veto zig-zaggy ARIMA on short history (n<=14) and switch to a steadier model Added 9/7/25
        # Rationale: with 12 months, ARIMA can overfit & alternate up/down with large range → poor generalization.
        try:
            model_name = str(best_model)
        except Exception:
            model_name = ""

        if model_name.startswith("ARIMA"):
            # Metrics on the chosen forecast (no re-fit)
            try:
                fc = np.asarray(best_forecast, dtype=float)
            except Exception:
                fc = np.array([], dtype=float)

            # Only act when the training history is short
            try:
                arr = np.asarray(series.values, dtype=float)
            except Exception:
                arr = np.asarray(series, dtype=float)
            n_obs = int(arr.size)

            if n_obs <= 14 and fc.size >= 3:
                diffs = np.diff(fc)
                # Oscillation count: sign changes in successive differences
                sign_changes = int(np.sum(np.sign(diffs[1:]) * np.sign(diffs[:-1]) < 0))
                fc_cv = float(np.std(fc) / max(np.mean(fc), 1e-9))
                fc_range = float(np.max(fc) - np.min(fc))

                # Veto only the true sawtooth failure mode (short-history ARIMA oscillation)
                VETO = (sign_changes >= 5)

                if VETO:
                    # Choose the steadier alternative by cached CV, falling back safely if cache missing
                    def _cv_mape_safe(tag, factory):
                        try:
                            return _cv_mape_cached(tag, factory)
                        except Exception:
                            return _cv_mape(arr, factory)

                    def _cv_ro_safe(tag, factory):
                        try:
                            return _cv_ro_cached(tag, factory)
                        except Exception:
                            return _rolling_origin_error(arr, factory, horizon=VALIDATION_PERIODS, folds=3, metric="wmape")

                    dt_cv = _cv_mape_safe('DampedTrend',  lambda tr: DampedTrend(tr))
                    th_cv = _cv_ro_safe('Theta',           lambda tr: Theta(tr))

                    if np.isfinite(dt_cv) and (dt_cv <= th_cv or not np.isfinite(th_cv)):
                        best_model = "ARIMA → Damped Trend (veto zig-zag on short history)"
                        best_forecast = list(DampedTrend(series).forecast(forecast_length))
                        if 'best_mape' in locals() and np.isfinite(dt_cv):
                            best_mape = min(best_mape, dt_cv)
                    else:
                        best_model = "ARIMA → Theta (veto zig-zag on short history)"
                        best_forecast = list(Theta(series).forecast(forecast_length))
                        if 'best_mape' in locals() and np.isfinite(th_cv):
                            best_mape = min(best_mape, th_cv)
     
        # --- Catastrophic guards (micro) --- Added 9/6/25
        # A) Avoid forecasting into likely zero-demand years
        # B) Avoid all-zero forecasts when recent demand exists
        # C) Cap extreme positive bias on flat/intermittent by gentle rescaling
        guard_locked = False
        try:          
            arr = np.asarray(series.values, dtype=float)
            last6 = arr[-6:] if arr.size >= 6 else arr
            last12 = arr[-12:] if arr.size >= 12 else arr
            mu6 = float(np.nanmean(last6)) if last6.size else 0.0
            zero_ratio_12 = float(np.mean(last12 == 0)) if last12.size else 1.0
            eps = 1e-9

            # 7A: Zero-demand guard — ZERO-RULE ONLY if ALL history is zero (ship rule)
            if (mu6 <= eps) and (zero_ratio_12 >= 0.90) and np.all(_y_all == 0.0):
                best_model = 'Zero-Rule'
                best_forecast = [0.0] * int(forecast_length)
                guard_locked = True

            # 7A.2: Obsolescence guard — evaluate ZeroDecay as a candidate (no change to 7A/7B/7C)
            # Only run when 7A did NOT already force all-zero (i.e., when recent demand isn't ~0 with ≥90% zeros)
            if not ((mu6 <= eps) and (zero_ratio_12 >= 0.90)):

                tot_last12 = float(np.nansum(last12))
                tot_prev12 = float(np.nansum(arr[-24:-12])) if arr.size >= 24 else float('nan')
                zd_hl = 4  # months

                # Hard evidence obsolescence trigger: last-12 total collapsed vs prior-12, with heavy recent zeros
                if (
                    np.isfinite(tot_last12) and np.isfinite(tot_prev12) and tot_prev12 > 0
                    and (tot_last12 <= 0.30 * tot_prev12) and (zero_ratio_12 >= 0.50)
                ):
                    cand = list(ZeroDecay(arr, half_life=zd_hl).forecast(forecast_length))

                    # accept only if it improves last-12 WAPE vs current forecast
                    act12 = np.asarray(series.values[-12:], dtype=float)
                    denom = float(np.nansum(np.abs(act12)))
                    if denom > 0:
                        cur_w = float(np.nansum(np.abs(np.asarray(best_forecast, dtype=float)[:12] - act12)) / denom)
                        cand_w = float(np.nansum(np.abs(np.asarray(cand, dtype=float)[:12] - act12)) / denom)
                        cand_acc = float(np.nansum(np.asarray(cand, dtype=float)[:12])) / max(float(np.nansum(np.abs(act12))), 1e-9)

                        if (cand_w + 0.02) <= cur_w and (ZD_ACC_LO <= cand_acc <= ZD_ACC_HI):
                            best_forecast = cand
                            best_model = f'ZeroDecay(hl={zd_hl})'
                            guard_locked = True

            # 7B: No all-zero forecast when recent demand exists — fall back to best baseline
            if (mu6 > eps) and (np.sum(np.asarray(best_forecast, dtype=float)) == 0.0):
                # Ensure baseline_err/baseline_name in scope; recompute if needed
                try:
                    baseline_err  # noqa
                    baseline_name # noqa
                except NameError:
                    horizon = VALIDATION_PERIODS
                    m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
                    theta_err = _rolling_origin_error(arr,
                                                      lambda tr: Theta(tr),
                                                      horizon=horizon, folds=3, metric="wmape")
                    sss_err   = _rolling_origin_error(arr,
                                                      lambda tr: SNaiveShrink(tr, m=m),
                                                      horizon=horizon, folds=3, metric="wmape")
                    baseline_err = min(theta_err, sss_err)
                    baseline_name = 'Theta' if theta_err <= sss_err else 'SNaive-S'

                if baseline_name == 'Theta':
                    best_forecast = list(Theta(series).forecast(forecast_length))
                    best_model = 'Theta'
                else:
                    best_forecast = list(SNaiveShrink(series, m=m).forecast(forecast_length))
                    best_model = 'SNaive-S'

            if not guard_locked:
                # 7B.3: Level-shift override (fast). If last-6 mean diverges strongly from prior-6,
                # try a smoother that anchors to the new level. No refits beyond SES/DampedTrend.
                try:
                    arr = np.asarray(series.values, dtype=float)
                    arr = np.nan_to_num(arr, nan=0.0)

                    if arr.size >= 12:

                        # Do NOT apply level-shift override to seasonal series
                        m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
                        season_hint = _has_seasonality(arr, period=m)

                        if not season_hint:

                            last6 = arr[-6:]
                            prev6 = arr[-12:-6]

                            mu_last = float(np.nanmean(last6))
                            mu_prev = float(np.nanmean(prev6))

                            if np.isfinite(mu_last) and np.isfinite(mu_prev) and (mu_prev > 0):

                                ratio_6 = mu_last / mu_prev

                                last3 = arr[-3:]
                                prev3 = arr[-6:-3]

                                mu_last3 = float(np.nanmean(last3))
                                mu_prev3 = float(np.nanmean(prev3))

                                ratio_3 = (
                                    mu_last3 / mu_prev3
                                    if (np.isfinite(mu_last3) and np.isfinite(mu_prev3) and mu_prev3 > 0)
                                    else 1.0
                                )

                                level_shift = (
                                    ((ratio_6 >= LS_UP) and (ratio_3 >= LS_UP)) or
                                    ((ratio_6 <= LS_DOWN) and (ratio_3 <= LS_DOWN))
                                )

                                if level_shift:
                                    level_shift_flag = True

                                    act12 = np.asarray(series.values[-12:], dtype=float)

                                    def _wape(a, f):
                                        a = np.asarray(a, dtype=float)[:12]
                                        f = np.asarray(f, dtype=float)[:12]
                                        denom = float(np.nansum(np.abs(a)))
                                        if denom <= 0:
                                            return np.inf
                                        return float(np.nansum(np.abs(f - a)) / denom)

                                    selected = False

                                    try:
                                        fc_dt = DampedTrend(series).forecast(forecast_length)
                                        curr_fc = np.asarray(best_forecast, dtype=float)
                                        curr_w = _wape(act12, curr_fc)
                                        cand_w = _wape(act12, fc_dt)
                                        cand_acc = (
                                            float(np.nansum(np.asarray(fc_dt, dtype=float)[:12])) /
                                            max(float(np.nansum(np.abs(act12))), 1e-9)
                                        )

                                        if (cand_w + 0.02) <= curr_w and (0.90 <= cand_acc <= 1.10):
                                            best_forecast = list(fc_dt)
                                            best_model = 'DampedTrend(level-shift)'
                                            selected = True
                                    except Exception:
                                        pass

                                    if not selected:
                                        if ("Trend" not in best_model) and ("HW" not in best_model):
                                            try:
                                                fc_ses = SES(series).forecast(forecast_length)
                                                curr_fc = np.asarray(best_forecast, dtype=float)
                                                curr_w = _wape(act12, curr_fc)
                                                cand_w = _wape(act12, fc_ses)
                                                cand_acc = (
                                                    float(np.nansum(np.asarray(fc_ses, dtype=float)[:12])) /
                                                    max(float(np.nansum(np.abs(act12))), 1e-9)
                                                )

                                                if (cand_w + 0.02) <= curr_w and (0.90 <= cand_acc <= 1.10):
                                                    best_forecast = list(fc_ses)
                                                    best_model = 'SES(level-shift)'
                                            except Exception:
                                                pass

                except Exception:
                    pass

            # 7C: Hard cap extreme positive bias on flat/intermittent by scaling shape down
            if profile in ['flat','intermittent'] and mu6 > eps:
                fc = np.asarray(best_forecast, dtype=float)
                if fc.size:
                    fc_mean = float(np.mean(fc))
                    if fc_mean > 2.5 * mu6:
                        scale = (2.0 * mu6) / max(fc_mean, eps)  # keep shape, cap mean at ~2×mu6
                        best_forecast = list((fc * scale).astype(float))
                    
        except Exception:
            pass
        
        # PATCH #3: adaptive guardrail against implausible spikes (enhanced 9/7/25)
        # Order: keep your existing guardrail first, then apply light, gated dampers.
        if not guard_locked:
            # 3.0) Keep your existing shape guard (unchanged)
            _guard_fcst, _guard_used = guardrail_or_fallback(
                series.values,
                best_forecast,
                period=HOLT_WINTERS_PARAMS['seasonal_periods']  # your existing seasonal period
            )
            if _guard_used:
                best_forecast = _guard_fcst
                best_model = f"{best_model} + Guardrail(SNaive)"

        # FINALIZER: Ensure SNaive-S MOY cap is applied on the final forecast and labeled
        try:
            _mn = str(best_model)
        except Exception:
            _mn = ""

        if ("SNaive-S" in _mn) and ("CapMOY(SNaive)" not in _mn) and ("lumpy fallback" not in _mn):
            try:
                arr = np.asarray(series.values, dtype=float)
            except Exception:
                arr = np.asarray(series, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0)

            m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
            n = int(arr.size)
            L = int(min(n, 2 * m))
            tail = arr[-L:] if L > 0 else arr

            if tail.size:
                bins = {k: [] for k in range(m)}
                start_idx = n - L
                pos_tail = []
                for off, val in enumerate(tail):
                    if np.isfinite(val) and val > 0.0:
                        pos_tail.append(float(val))
                        bins[(start_idx + off) % m].append(float(val))
                if len(pos_tail) == 0:
                    pos_tail = [0.0]

                q90_all = float(np.percentile(pos_tail, 90)) if len(pos_tail) >= 1 else 0.0
                overall = float(np.mean(pos_tail)) if len(pos_tail) >= 1 else 0.0

                # Trend gate: only cap if there is NOT a strong positive trend
                feats = _robust_trend_features(arr)
                trend_strength = float(feats.get("slope_abs_over_level", 0.0))
                trend_r2       = float(feats.get("r2", 0.0))
                strong_pos_trend = (trend_strength >= 0.055) and (trend_r2 >= 0.28)

                # Intermittency guard (ADI/CV^2) Added 9/23/25
                nz = arr[arr > 0.0]
                adi = (arr.size / max(1, nz.size))
                cv2 = (np.var(nz, ddof=1) / (np.mean(nz)**2)) if (nz.size > 1 and np.mean(nz) > 0.0) else float("inf")
                intermittent = (adi >= 1.2) or (cv2 >= 0.35) #Changed from 1.32 and 0.49
                slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) >= 2 else 0.0

                if (not strong_pos_trend) and (not intermittent) and (cv2 < 0.6) and (adi < 1.5) and (abs(slope) < 5.0):
                    cap_per_moy = np.zeros(m, dtype=float)
                    for k in range(m):
                        mx = float(np.max(bins[k])) if len(bins[k]) else 0.0
                        cap_k = min(mx * 1.15, q90_all * 1.30) if (mx > 0.0 or q90_all > 0.0) else overall * 1.50
                        cap_per_moy[k] = max(cap_k, 1e-9)

                    fc = np.asarray(best_forecast, dtype=float)
                    if fc.size:
                        for t in range(fc.size):
                            fc[t] = min(fc[t], cap_per_moy[t % m])
                        best_forecast = list(fc.astype(float))
                        best_model = f"{best_model} + CapMOY(SNaive)"

            # 3.1) Lightweight history diagnostics (local to this block; safe if earlier 'arr' not in scope)
            try:
                arr = np.asarray(series.values, dtype=float)
            except Exception:
                arr = np.asarray(series, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0)

            last12 = arr[-12:] if arr.size >= 12 else arr
            last6  = arr[-6:]  if arr.size >= 6  else arr
            mu6 = float(np.nanmean(last6)) if last6.size else float(np.nanmean(arr)) if arr.size else 0.0
            zero_ratio_12 = float(np.mean(last12 == 0)) if last12.size else 1.0

            pos = last12[last12 > 0]
            p50 = float(np.percentile(pos, 50)) if pos.size else 0.0
            p95 = float(np.percentile(pos, 95)) if pos.size else 0.0
            spikiness = (p95 / max(1e-9, p50)) if p50 > 0 else np.inf

            fc = np.asarray(best_forecast, dtype=float)

            # Lumpy fallback for SNaive-S (conventional per Hyndman)
            arr = np.asarray(series, dtype=float)
            nz = arr[arr > 0.0]
            cv2 = (np.var(nz, ddof=1) / np.mean(nz)**2) if len(nz) > 1 and np.mean(nz) > 0 else float('inf')
            if "SNaive" in str(best_model) and cv2 > 0.6:
                best_forecast = Theta(series).forecast(forecast_length)
                best_model = "SNaive-S → Theta (lumpy fallback)"

        # ---- VETO: kill flat-constant far from level OR cratered (too-low) finals ----
        # recent level from last up to 6 actuals
        y_full = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
        y_full = np.nan_to_num(y_full, nan=0.0)
        mu6 = float(np.mean(y_full[-6:])) if y_full.size >= 6 else float(np.mean(y_full))  # recent level

        if best_forecast is not None and mu6 > 0:
            f = np.asarray(best_forecast, dtype=float)
            # reject if the whole forecast is a single flat value far from level
            # OR if every point is implausibly low vs recent level (e.g., SES -> ~0)
            if _reject_flat_constant(f, mu6, band=0.35) or _too_low(f, mu6, floor=0.50): #Changed band from 0.25 and floor from 0.60 10/11/25
                # fall back to a safer baseline:
                # - lumpy/intermittent → Theta
                # - otherwise → seasonal shrink naive
                y = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
                y = np.nan_to_num(y, nan=0.0)
                nz = y[y > 0.0]
                cv2 = (np.var(nz, ddof=1) / (np.mean(nz) ** 2)) if (nz.size > 1 and np.mean(nz) > 0.0) else float("inf")

                if cv2 > 0.6:
                    best_forecast = Theta(series).forecast(forecast_length)
                    best_model = f"{best_model} → Theta (veto constant/too-low; lumpy)"
                else:
                    m = int(HOLT_WINTERS_PARAMS.get('seasonal_periods', 12))
                    best_forecast = SNaiveShrink(series, m=m).forecast(forecast_length)
                    best_model = f"{best_model} → SNaive-S (veto constant/too-low)"
        
        best_forecast = [max(0.0, float(v)) if np.isfinite(v) else 0.0 for v in best_forecast]

        row_result = {
            "Item Number": item,
            "Method Used": best_model,
            "MAPE (Validation)": round(best_mape * 100, 2) if best_mape != float('inf') else 0.0,
            "Outliers": ", ".join(outlier_dict.get(item, [])) if outlier_detection else "None"
        }
        for i, label in enumerate(forecast_labels):
            row_result[label] = int(round(best_forecast[i])) if best_forecast is not None else 0.0
        results.append(row_result)

        if progress_callback:
            progress_callback(idx)

    return {
        "status": "ready",
        "data": results,
        "outliers": outlier_dict
    }
# Updated 12/31/25