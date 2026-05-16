import os
import re

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from serve_model import (
    BODY_PARTS,
    NUTRITION_BY_TRIAGE,
    RECOVERY_PROTOCOLS,
    SEV_LABELS,
    SPORTS,
    doctor_clf,
    encode_input,
    recovery_reg,
    triage_clf,
)


app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")


def load_text_model():
    paths = {
        "model": os.path.join(MODEL_DIR, "sports_injury_nlp_model.pkl"),
        "body_encoder": os.path.join(MODEL_DIR, "body_encoder.pkl"),
        "triage_encoder": os.path.join(MODEL_DIR, "triage_encoder.pkl"),
        "sport_encoder": os.path.join(MODEL_DIR, "sport_encoder.pkl"),
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return None

    return {
        "model": joblib.load(paths["model"]),
        "body_encoder": joblib.load(paths["body_encoder"]),
        "triage_encoder": joblib.load(paths["triage_encoder"]),
        "sport_encoder": joblib.load(paths["sport_encoder"]),
    }


TEXT_MODEL = load_text_model()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_from_text(text):
    if not TEXT_MODEL or not text.strip():
        return None

    try:
        cleaned = clean_text(text)
        pred = TEXT_MODEL["model"].predict([cleaned])[0]
        probabilities = TEXT_MODEL["model"].predict_proba([cleaned])

        # Handle multi-output model probabilities structure
        # Each element in probabilities is an array for that output's class probabilities
        if isinstance(probabilities, list):
            body_confidence = float(np.max(probabilities[0][0]))
            triage_confidence = float(np.max(probabilities[1][0]))
            doctor_confidence = float(np.max(probabilities[2][0]))
            sport_confidence = float(np.max(probabilities[3][0]))
        else:
            # Fallback for different probability structure
            body_confidence = float(np.max(probabilities[0]))
            triage_confidence = float(np.max(probabilities[1]))
            doctor_confidence = float(np.max(probabilities[2]))
            sport_confidence = float(np.max(probabilities[3]))

        return {
            "body_part": TEXT_MODEL["body_encoder"].inverse_transform([pred[0]])[0],
            "triage_level": TEXT_MODEL["triage_encoder"].inverse_transform([pred[1]])[0],
            "requires_doctor": bool(int(pred[2])),
            "sport": TEXT_MODEL["sport_encoder"].inverse_transform([pred[3]])[0],
            "confidence": {
                "body_part": round(body_confidence, 3),
                "triage_level": round(triage_confidence, 3),
                "requires_doctor": round(doctor_confidence, 3),
                "sport": round(sport_confidence, 3),
            },
        }
    except Exception as e:
        # If text prediction fails, return None and use form values only
        return None

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AthleteEdge AI</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      color: #162033;
    }
    header {
      background: #10233f;
      color: white;
      padding: 28px 18px;
      text-align: center;
    }
    header h1 { margin: 0 0 8px; font-size: 30px; }
    header p { margin: 0; color: #c8d6ea; }
    main {
      width: min(1040px, 100%);
      margin: 0 auto;
      padding: 24px 16px 40px;
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr 1fr;
    }
    section {
      background: white;
      border: 1px solid #d9e1ee;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 8px 24px rgba(20, 40, 70, 0.06);
    }
    h2 { margin: 0 0 16px; font-size: 20px; }
    form {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 14px;
      font-weight: 700;
      color: #33425c;
    }
    input, select {
      width: 100%;
      padding: 10px 11px;
      border: 1px solid #bac7d9;
      border-radius: 6px;
      font-size: 15px;
      background: white;
      color: #162033;
    }
    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      padding: 10px 11px;
      border: 1px solid #bac7d9;
      border-radius: 6px;
      font: inherit;
      background: white;
      color: #162033;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    button {
      margin-top: 8px;
      border: 0;
      border-radius: 6px;
      padding: 12px 16px;
      background: #1b66d1;
      color: white;
      font-weight: 700;
      font-size: 16px;
      cursor: pointer;
    }
    button:hover { background: #1555af; }
    .result {
      min-height: 220px;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .empty {
      color: #6d7b91;
      line-height: 1.5;
    }
    .badge {
      display: inline-block;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e8f0ff;
      color: #174f9e;
      font-weight: 700;
      text-transform: capitalize;
    }
    .doctor {
      padding: 12px;
      border-radius: 8px;
      background: #fff4e5;
      border: 1px solid #ffd796;
      font-weight: 700;
    }
    .doctor.ok {
      background: #edf9f0;
      border-color: #bfe8c8;
    }
    ul { margin: 8px 0 0; padding-left: 20px; }
    li { margin: 6px 0; line-height: 1.4; }
    .muted { color: #5c6b80; }
    .help {
      color: #5c6b80;
      font-size: 13px;
      font-weight: 400;
      line-height: 1.35;
    }
    .error {
      padding: 12px;
      border-radius: 8px;
      background: #ffecec;
      border: 1px solid #ffb7b7;
      color: #9b1c1c;
      font-weight: 700;
    }
    .nlp-box {
      padding: 12px;
      border-radius: 8px;
      background: #f1f8ff;
      border: 1px solid #badcff;
    }
    .nlp-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .nlp-grid div {
      background: white;
      border: 1px solid #d9e8fa;
      border-radius: 6px;
      padding: 8px;
    }
    .used-box {
      padding: 12px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #d9e1ee;
    }
    code {
      font-family: Consolas, monospace;
      font-size: 13px;
    }
    @media (max-width: 760px) {
      main, .grid, .nlp-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>AthleteEdge AI</h1>
    <p>Simple sports injury triage helper</p>
  </header>

  <main>
    <section>
      <h2>Injury Details</h2>
      <form method="post">
        <div class="grid">
          <label>Sport
            <select name="sport">
              {% for item in sports %}<option value="{{ item }}" {% if form.sport == item %}selected{% endif %}>{{ item }}</option>{% endfor %}
            </select>
          </label>
          <label>Body Part
            <select name="body_part">
              {% for item in body_parts %}<option value="{{ item }}" {% if form.body_part == item %}selected{% endif %}>{{ item }}</option>{% endfor %}
            </select>
          </label>
        </div>

        <div class="grid">
          <label>Self Severity
            <select name="self_severity">
              {% for item in severities %}<option value="{{ item }}" {% if form.self_severity == item %}selected{% endif %}>{{ item.title() }}</option>{% endfor %}
            </select>
          </label>
          <label>Age
            <input name="age" type="number" min="5" max="100" value="{{ form.age }}" required>
          </label>
        </div>

        <div class="grid">
          <label>Weight (kg)
            <input name="weight_kg" type="number" min="20" max="200" value="{{ form.weight_kg }}" required>
          </label>
          <label>Fatigue Score (1-10)
            <input name="fatigue_score" type="number" min="1" max="10" value="{{ form.fatigue_score }}" required>
            <span class="help">1 means fresh/normal energy. 10 means exhausted, overtrained, or unusually tired.</span>
          </label>
        </div>

        <div class="grid">
          <label>Training Days / Week
            <input name="training_days_per_week" type="number" min="0" max="7" value="{{ form.training_days_per_week }}" required>
          </label>
          <label>Training Hours / Week
            <input name="training_hours_per_week" type="number" min="0" max="80" step="0.5" value="{{ form.training_hours_per_week }}" required>
          </label>
        </div>

        <div class="grid">
          <label>Previous Injury
            <select name="previous_injury">
              <option value="0" {% if form.previous_injury == "0" %}selected{% endif %}>No</option>
              <option value="1" {% if form.previous_injury == "1" %}selected{% endif %}>Yes</option>
            </select>
          </label>
          <label>Match Within 48h
            <select name="match_within_48h">
              <option value="0" {% if form.match_within_48h == "0" %}selected{% endif %}>No</option>
              <option value="1" {% if form.match_within_48h == "1" %}selected{% endif %}>Yes</option>
            </select>
          </label>
        </div>

        <label>Describe Your Problem
          <textarea name="problem_description" placeholder="Example: I felt sharp pain in my shoulder while bowling and it hurts when I lift my arm.">{{ form.problem_description }}</textarea>
        </label>

        <button type="submit">Get Prediction</button>
      </form>
    </section>

    <section class="result">
      <h2>Result</h2>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% elif result %}
        <span class="badge">{{ result.triage_level }} triage</span>
        <div class="doctor {% if not result.requires_doctor %}ok{% endif %}">
          {% if result.requires_doctor %}Doctor recommended{% else %}Doctor not immediately required{% endif %}
          <div class="muted">Confidence: {{ "%.0f"|format(result.doctor_confidence * 100) }}%</div>
        </div>
        <p><strong>Estimated recovery:</strong> {{ result.recovery_days }} days</p>
        <p class="muted">Probabilities: mild {{ result.triage_probs.mild }}, moderate {{ result.triage_probs.moderate }}, severe {{ result.triage_probs.severe }}</p>
        <div class="used-box">
          <strong>Final Inputs Used</strong>
          <p class="muted">Sport: {{ result.inputs_used.sport }} · Body Part: {{ result.inputs_used.body_part }} · Severity: {{ result.inputs_used.self_severity.title() }}</p>
        </div>
        {% if result.problem_description %}
          <p><strong>Player description:</strong> {{ result.problem_description }}</p>
        {% endif %}

        {% if result.text_prediction %}
          <div class="nlp-box">
            <strong>Description Analysis</strong>
            <div class="nlp-grid">
              <div><span class="muted">Body Part</span><br>{{ result.text_prediction.body_part }} <span class="muted">({{ "%.0f"|format(result.text_prediction.confidence.body_part * 100) }}%)</span></div>
              <div><span class="muted">Triage</span><br>{{ result.text_prediction.triage_level.title() }} <span class="muted">({{ "%.0f"|format(result.text_prediction.confidence.triage_level * 100) }}%)</span></div>
              <div><span class="muted">Doctor</span><br>{% if result.text_prediction.requires_doctor %}Recommended{% else %}Not immediate{% endif %} <span class="muted">({{ "%.0f"|format(result.text_prediction.confidence.requires_doctor * 100) }}%)</span></div>
              <div><span class="muted">Sport</span><br>{{ result.text_prediction.sport }} <span class="muted">({{ "%.0f"|format(result.text_prediction.confidence.sport * 100) }}%)</span></div>
            </div>
            {% if result.text_prediction_mismatch %}
              <p class="muted">Some description guesses differ from the selected form values, so the final prediction keeps the form values.</p>
            {% else %}
              <p class="muted">The description analysis agrees with the selected form values.</p>
            {% endif %}
          </div>
        {% elif result.problem_description %}
          <p class="muted">Text model files were not found, so the description is shown only as context.</p>
        {% endif %}

        <div class="protocol">
          <strong>Care Protocol: {{ result.care_protocol.body_part }} / {{ result.care_protocol.triage_level.title() }}</strong>
          <p class="muted">This protocol is selected from the rule-based recovery library in <code>serve_model.py</code>.</p>
        </div>

        <div>
          <strong>Protocol Phases</strong>
          <ul>
            {% for phase in result.care_protocol.phases %}
              <li><strong>{{ phase.phase }}:</strong> {{ phase.guidance }}</li>
            {% endfor %}
          </ul>
        </div>

        <div>
          <strong>Protocol Exercises</strong>
          <ul>{% for item in result.care_protocol.exercises %}<li>{{ item }}</li>{% endfor %}</ul>
        </div>

        <div>
          <strong>Nutrition Tips</strong>
          <ul>{% for item in result.nutrition_tips %}<li>{{ item }}</li>{% endfor %}</ul>
        </div>
      {% else %}
        <p class="empty">Fill the form and submit it to see triage level, doctor guidance, recovery estimate, rule-based recovery protocol, exercises, and nutrition tips.</p>
      {% endif %}
    </section>
  </main>
</body>
</html>
"""

DEFAULT_FORM = {
    "sport": "Cricket",
    "body_part": "Shoulder",
    "self_severity": "moderate",
    "age": "19",
    "weight_kg": "68",
    "fatigue_score": "7",
    "training_days_per_week": "5",
    "training_hours_per_week": "18",
    "previous_injury": "0",
    "match_within_48h": "1",
    "problem_description": "",
}


def build_payload(form):
    return {
        "sport": form["sport"],
        "body_part": form["body_part"],
        "self_severity": form["self_severity"],
        "age": int(form["age"]),
        "weight_kg": float(form["weight_kg"]),
        "fatigue_score": int(form["fatigue_score"]),
        "training_days_per_week": int(form["training_days_per_week"]),
        "training_hours_per_week": float(form["training_hours_per_week"]),
        "previous_injury": int(form["previous_injury"]),
        "match_within_48h": int(form["match_within_48h"]),
        "problem_description": form.get("problem_description", "").strip(),
    }


def make_prediction(data):
    text_prediction = predict_from_text(data.get("problem_description", ""))
    model_data = data.copy()

    X = encode_input(model_data)
    triage_idx = int(triage_clf.predict(X)[0])
    triage_probs = triage_clf.predict_proba(X)[0].tolist()
    triage_label = ["mild", "moderate", "severe"][triage_idx]

    X_d = X.copy()
    X_d["f10"] = np.float32(triage_idx)
    doc_idx = int(doctor_clf.predict(X_d)[0])
    doc_prob = float(doctor_clf.predict_proba(X_d)[0][1])
    rec_days = int(max(1, min(120, recovery_reg.predict(X_d)[0])))

    protocol = RECOVERY_PROTOCOLS.get(model_data["body_part"], {}).get(triage_label, {})
    phases = protocol.get("phases", [])
    exercises = protocol.get("exercises", [])
    return {
        "triage_level": triage_label,
        "triage_probs": {
            "mild": round(triage_probs[0], 3),
            "moderate": round(triage_probs[1], 3),
            "severe": round(triage_probs[2], 3),
        },
        "requires_doctor": bool(doc_idx),
        "doctor_confidence": round(doc_prob, 3),
        "recovery_days": rec_days,
        "problem_description": data.get("problem_description", ""),
        "text_prediction": text_prediction,
        "text_prediction_mismatch": bool(
            text_prediction
            and (
                text_prediction["sport"] != model_data["sport"]
                or text_prediction["body_part"] != model_data["body_part"]
                or text_prediction["triage_level"] != model_data["self_severity"]
            )
        ),
        "inputs_used": {
            "sport": model_data["sport"],
            "body_part": model_data["body_part"],
            "self_severity": model_data["self_severity"],
        },
        "care_protocol": {
            "body_part": model_data["body_part"],
            "triage_level": triage_label,
            "phases": phases,
            "exercises": exercises,
        },
        "recovery_phases": phases,
        "recovery_exercises": exercises,
        "nutrition_tips": NUTRITION_BY_TRIAGE[triage_label],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    form = DEFAULT_FORM.copy()
    result = None
    error = None

    if request.method == "POST":
        form.update(request.form.to_dict())
        try:
            result = make_prediction(build_payload(form))
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        PAGE,
        sports=SPORTS,
        body_parts=BODY_PARTS,
        severities=SEV_LABELS,
        form=form,
        result=result,
        error=error,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(make_prediction(request.get_json(force=True)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"AthleteEdge AI frontend running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
