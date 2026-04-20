import os
import sys
import logging
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure the backend directory is on the path when running directly
sys.path.insert(0, os.path.dirname(__file__))
from house_utils import (
    load_metadata,
    validate_input,
    build_input_frame,
    format_prediction,
)

# ── Logging ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HousePriceAPI")

# ── Path Config ──────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.joblib")
META_PATH   = os.path.join(BASE_DIR, "model_meta.json")

# ── App Init ───────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from the frontend

# ── Load Artefacts at Startup ──────────
def load_artefacts():
    if not os.path.exists(MODEL_PATH):
        log.error("model.joblib not found at '%s'. Run house_model.py first.", MODEL_PATH)
        sys.exit(1)

    model    = joblib.load(MODEL_PATH)
    metadata = load_metadata(META_PATH)
    log.info("Model loaded: %s", metadata.get("model_name", "unknown"))
    log.info("Known locations: %s", metadata.get("known_locations", []))
    return model, metadata

MODEL, METADATA = load_artefacts()


# ── Routes ────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Housing Prediction API is running"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": METADATA.get("model_name")}), 200


@app.route("/meta", methods=["GET"])
def meta():
    safe = {k: v for k, v in METADATA.items() if k != "metrics"}
    return jsonify(safe), 200


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Parse JSON body
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body is not valid JSON."}), 400

    log.info("Prediction request: %s", data)

    # 2. Validate inputs
    known_locations = METADATA.get("known_locations", [])
    is_valid, err   = validate_input(data, known_locations)
    if not is_valid:
        log.warning("Validation failed: %s", err)
        return jsonify({"error": err}), 422

    # 3. Build feature DataFrame
    cat_features = METADATA.get("cat_features", ["location"])
    num_features = METADATA.get("num_features", ["size", "rooms"])
    X = build_input_frame(data, cat_features, num_features)

    # 4. Predict
    try:
        raw = MODEL.predict(X)[0]
    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        return jsonify({"error": "Prediction failed. See server logs."}), 500

    result = format_prediction(raw)
    log.info("Prediction result: $%s", f"{result['predicted_price']:,}")
    return jsonify(result), 200


# ── Error Handlers ─────────────
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "HTTP method not allowed."}), 405

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error."}), 500


# ── Entry Point ───────────────
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting House Price API on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)