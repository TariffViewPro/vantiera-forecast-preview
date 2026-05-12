import pandas as pd
import numpy as np

from forecast_utils import run_full_forecast
from forecast_config import (
    MIN_HISTORY_REQUIRED,
    MAX_HISTORY_COLUMNS
)


def parse_history_input(raw_text):
    """
    Accepts pasted values separated by:
    - commas
    - spaces
    - line breaks

    Returns:
        list[float]
    """

    if not raw_text or not raw_text.strip():
        raise ValueError("No history provided.")

    # Normalize separators
    cleaned = raw_text.replace(",", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("\t", " ")

    parts = cleaned.split()

    values = []

    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            raise ValueError(f"Invalid value detected: '{part}'")

    if len(values) < MIN_HISTORY_REQUIRED:
        raise ValueError(
            f"At least {MIN_HISTORY_REQUIRED} periods are required."
        )

    if len(values) > MAX_HISTORY_COLUMNS:
        raise ValueError(
            f"Maximum {MAX_HISTORY_COLUMNS} periods allowed."
        )

    return values


def build_forecast_dataframe(history_values):
    """
    Converts history into the structure expected by
    run_full_forecast().
    """

    row = {
        "Item Number": "PreviewItem"
    }

    for i, value in enumerate(history_values, start=1):
        row[f"H{i}"] = value

    return pd.DataFrame([row])


def generate_simple_average(history_values, forecast_length=12):
    """
    Simple average comparison forecast.
    """

    avg = np.mean(history_values)

    return [round(avg) for _ in range(forecast_length)]


def build_summary_note(history_values):
    """
    Lightweight operational summary.
    """

    values = np.array(history_values, dtype=float)

    if len(values) < 6:
        return None

    recent_avg = np.mean(values[-6:])
    earlier_avg = np.mean(values[:6])

    if recent_avg > earlier_avg * 1.10:
        return "Recent demand shows an upward trend."

    if recent_avg < earlier_avg * 0.90:
        return "Recent demand shows a downward trend."

    return "Demand appears relatively stable."


def generate_preview_forecast(raw_text):
    """
    Main public wrapper used by the preview app.
    """

    history_values = parse_history_input(raw_text)

    df = build_forecast_dataframe(history_values)

    result = run_full_forecast(
        df=df,
        outlier_detection=False
    )

    if result.get("status") == "error":
        raise ValueError(result.get("message"))

    forecast_data = result.get("data", [])

    if not forecast_data:
        raise ValueError("Forecast could not be generated.")

    row = forecast_data[0]

    forecast_values = []

    for key, value in row.items():

        if not isinstance(key, str):
            continue

        if not key.startswith("Period "):
            continue

        try:
            numeric_value = float(value)
            forecast_values.append(round(numeric_value))
        except (ValueError, TypeError):
            continue
        
    simple_average = generate_simple_average(history_values)

    summary_note = build_summary_note(history_values)

    return {
        "history": [round(v) for v in history_values],
        "forecast": forecast_values,
        "simple_average": simple_average,
        "summary_note": summary_note
    }