import json
import os
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Expected input schema
REQUIRED_FIELDS = {"location", "size", "rooms"}

# Reasonable value bounds
BOUNDS = {
    "size":  (100, 20_000),   # sq ft
    "rooms": (1, 20),
}


def load_metadata(path: str) -> dict:
    """Load model metadata (known locations, feature names, etc.)."""
    if not os.path.exists(path):
        log.warning("Metadata file not found at '%s'; using defaults.", path)
        return {"known_locations": ["urban", "suburban", "rural", "coastal"]}
    with open(path) as f:
        return json.load(f)


def validate_input(data: dict, known_locations: list) -> tuple[bool, str]:
    """
    Validate incoming JSON payload.
    Returns (is_valid, error_message).
    """
    # Check for missing fields
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return False, f"Missing required fields: {sorted(missing)}"

    # Validate location
    location = data.get("location")
    if not isinstance(location, str) or location.strip() == "":
        return False, "'location' must be a non-empty string."
    if location.lower() not in [loc.lower() for loc in known_locations]:
        return False, (
            f"Unknown location '{location}'. "
            f"Valid options: {known_locations}"
        )

    # Validate size
    size = data.get("size")
    if not isinstance(size, (int, float)) or isinstance(size, bool):
        return False, "'size' must be a numeric value (sq ft)."
    lo, hi = BOUNDS["size"]
    if not (lo <= size <= hi):
        return False, f"'size' must be between {lo} and {hi} sq ft."

    # Validate rooms
    rooms = data.get("rooms")
    if not isinstance(rooms, (int, float)) or isinstance(rooms, bool):
        return False, "'rooms' must be a numeric value."
    lo, hi = BOUNDS["rooms"]
    if not (lo <= rooms <= hi):
        return False, f"'rooms' must be between {lo} and {hi}."

    return True, ""


def build_input_frame(data: dict, cat_features: list, num_features: list) -> pd.DataFrame:
    """
    Convert validated JSON payload into a DataFrame that matches the
    column order expected by the trained pipeline.
    """
    row = {
        "location": str(data["location"]).lower(),
        "size":     float(data["size"]),
        "rooms":    float(data["rooms"]),
    }
    # Respect the exact column order from training
    cols = cat_features + num_features
    return pd.DataFrame([row], columns=cols)


def format_prediction(raw_price: float) -> dict:
    """Round and package the model output for the API response."""
    return {"predicted_price": round(float(raw_price))}