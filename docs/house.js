"use strict";

// ── Configuration ─────────────────────
const API_BASE = "https://housing-prediction-76lj.onrender.com";
const ENDPOINT = `${API_BASE}/predict`;

// ── DOM References ────────────────────
const $ = id => document.getElementById(id);

const dom = {
    form:         $("predictionForm"),
    locationEl:   $("location"),
    sizeEl:       $("size"),
    roomsEl:      $("rooms"),
    submitBtn:    $("submitBtn"),
    resultPanel:  $("resultPanel"),
    stateLoading: $("stateLoading"),
    stateSuccess: $("stateSuccess"),
    stateError:   $("stateError"),
    resultPrice:  $("resultPrice"),
    errorMessage: $("errorMessage"),
};

// ── UI State Machine ────────────────
const UIState = {
    IDLE:    "idle",
    LOADING: "loading",
    SUCCESS: "success",
    ERROR:   "error",
};

function setState(state, payload = {}) {
    dom.stateLoading.hidden = true;
    dom.stateSuccess.hidden = true;
    dom.stateError.hidden   = true;
    dom.submitBtn.disabled  = false;

    switch (state) {
        case UIState.IDLE:
            dom.resultPanel.style.display = "none";
            break;
        case UIState.LOADING:
            dom.resultPanel.style.display = "block";
            dom.stateLoading.hidden       = false;
            dom.submitBtn.disabled        = true;
            dom.submitBtn.querySelector(".btn-text").textContent = "Analysing…";
            break;
        case UIState.SUCCESS:
            dom.resultPanel.style.display = "block";
            dom.stateSuccess.hidden       = false;
            dom.resultPrice.textContent   = formatCurrency(payload.predicted_price);
            dom.submitBtn.querySelector(".btn-text").textContent = "Estimate Price";
            break;
        case UIState.ERROR:
            dom.resultPanel.style.display = "block";
            dom.stateError.hidden         = false;
            dom.errorMessage.textContent  = payload.message || "An unexpected error occurred.";
            dom.submitBtn.querySelector(".btn-text").textContent = "Estimate Price";
            break;
    }
}

setState(UIState.IDLE);

// ── Validation ─────────────────────
function clearValidation() {
    [dom.locationEl, dom.sizeEl, dom.roomsEl].forEach(el => {
        el.classList.remove("is-invalid");
    });
}

function validate() {
    clearValidation();
    const errors = [];

    const location = dom.locationEl.value;
    if (!location) {
        dom.locationEl.classList.add("is-invalid");
        errors.push("Please select a location.");
    }

    const size = Number(dom.sizeEl.value);
    if (!dom.sizeEl.value || isNaN(size) || size < 100 || size > 20000) {
        dom.sizeEl.classList.add("is-invalid");
        errors.push("Size must be between 100 and 20,000 sq ft.");
    }

    const rooms = Number(dom.roomsEl.value);
    if (!dom.roomsEl.value || isNaN(rooms) || rooms < 1 || rooms > 20 || !Number.isInteger(rooms)) {
        dom.roomsEl.classList.add("is-invalid");
        errors.push("Rooms must be a whole number between 1 and 20.");
    }

    return { isValid: errors.length === 0, errors, values: { location, size, rooms } };
}

// ── API Call ─────────────────────────
async function fetchPrediction(payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30_000);

    try {
        const response = await fetch(ENDPOINT, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload),
            signal:  controller.signal,
        });

        const data = await response.json();

        if (!response.ok) {
            const msg = data?.error || `Server error (${response.status})`;
            throw new Error(msg);
        }

        if (typeof data.predicted_price !== "number") {
            throw new Error("Invalid response format from server.");
        }

        return data;

    } catch (err) {
        if (err.name === "AbortError") {
            throw new Error("Request timed out. Render may be waking up — try again in 1 minute.");
        }
        if (err instanceof TypeError && err.message.includes("fetch")) {
            throw new Error("Cannot reach the server. Check your internet connection.");
        }
        throw err;
    } finally {
        clearTimeout(timeout);
    }
}

// ── Event Handlers ─────────────────
async function handleSubmit() {
    const { isValid, errors, values } = validate();

    if (!isValid) {
        setState(UIState.ERROR, { message: errors[0] });
        return;
    }

    setState(UIState.LOADING);

    try {
        const result = await fetchPrediction(values);
        setState(UIState.SUCCESS, result);
    } catch (err) {
        setState(UIState.ERROR, { message: err.message });
        console.error("[EstimateIQ] Prediction error:", err);
    }
}

function handleReset() {
    clearValidation();
    setState(UIState.IDLE);
    dom.locationEl.value = "";
    dom.sizeEl.value     = "";
    dom.roomsEl.value    = "";
    dom.locationEl.focus();
}

// ── Utilities ────────────────
function formatCurrency(value) {
    return new Intl.NumberFormat("en-US", {
        style:                 "currency",
        currency:              "USD",
        maximumFractionDigits: 0,
    }).format(value);
}

// ── Keyboard shortcut ───────────
document.addEventListener("keydown", e => {
    if (e.key === "Enter" && document.activeElement !== dom.submitBtn) {
        handleSubmit();
    }
});
