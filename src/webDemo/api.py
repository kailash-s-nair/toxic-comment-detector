from typing import Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.webDemo.demo import load_model, predict_scores, LABEL_COLUMNS

app = Flask(__name__)
CORS(app)

MODEL_NAME = "model2"
MODEL: Any = None


def get_model() -> Any:
    """
    Lazy-load the model on first request, then reuse it.
    """
    global MODEL
    if MODEL is None:
        MODEL = load_model(MODEL_NAME)
    return MODEL


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    data = request.get_json(silent=True) or {}
    comment = data.get("comment", "")

    if not isinstance(comment, str) or not comment.strip():
        return (
            jsonify({"error": "Field 'comment' must be a non-empty string."}),
            400,
        )

    model = get_model()

    try:
        raw_scores: Dict[str, float] = predict_scores(model, comment)
    except NotImplementedError as e:
        return (
            jsonify(
                {
                    "error": "Prediction not implemented. "
                    "Please implement load_model() and predict_scores() in demo.py.",
                    "details": str(e),
                }
            ),
            500,
        )

    # Ensure we only return known labels and cast to float
    scores: Dict[str, float] = {
        lbl: float(raw_scores.get(lbl, 0.0)) for lbl in LABEL_COLUMNS
    }

    return jsonify(
        {
            "comment": comment,
            "model": MODEL_NAME,
            "scores": scores,
        }
    )


if __name__ == "__main__":
    # For local development only. Use a proper WSGI server in production.
    app.run(host="0.0.0.0", port=8000, debug=True)