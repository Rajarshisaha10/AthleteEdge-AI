"""
AthleteEdge AI — Injury Triage ML API Server
=============================================
Run:   python3 serve_model.py
Port:  5050

POST /predict
  Body (JSON):
    {
      "sport":         "Cricket",
      "body_part":     "Shoulder",
      "self_severity": "moderate",
      "age":           19,
      "weight_kg":     68,
      "fatigue_score": 7,
      "training_days_per_week": 5,
      "training_hours_per_week": 18.0,
      "previous_injury": 0,
      "match_within_48h": 1
    }

  Response (JSON):
    {
      "triage_level":    "moderate",
      "triage_probs":    {"mild": 0.05, "moderate": 0.84, "severe": 0.11},
      "requires_doctor": true,
      "doctor_confidence": 0.73,
      "recovery_days":   18,
      "confidence_note": "Self-reported severity is the strongest signal. Keep training logs for better accuracy."
    }
"""

import json, joblib, os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import warnings

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

# Load models with compatibility handling for sklearn version mismatches
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    triage_clf   = joblib.load(os.path.join(MODEL_DIR, "triage_clf.joblib"))
    doctor_clf   = joblib.load(os.path.join(MODEL_DIR, "doctor_clf.joblib"))
    recovery_reg = joblib.load(os.path.join(MODEL_DIR, "recovery_reg.joblib"))
    
    # Fix sklearn compatibility - add missing multi_class attribute if needed
    for model in [triage_clf, doctor_clf, recovery_reg]:
        if hasattr(model, '__dict__') and 'multi_class' not in model.__dict__:
            if hasattr(model, '_estimator_type') and model._estimator_type == 'classifier':
                try:
                    model.multi_class = getattr(model, 'multi_class', 'multinomial')
                except:
                    pass

with open(os.path.join(MODEL_DIR, "label_maps.json")) as f:
    MAPS = json.load(f)

SPORTS      = MAPS["sports"]
BODY_PARTS  = MAPS["body_parts"]
SEV_LABELS  = MAPS["severity_labels"]
FEAT_NAMES  = [f"f{i}" for i in range(10)]

# Rule-based recovery protocol library (body_part × triage_level)
RECOVERY_PROTOCOLS = {
    "Head/Neck": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–2: Rest", "guidance": "Complete rest from activity. No heading, no contact. Monitor for delayed concussion symptoms like nausea or dizziness."},
                {"phase": "Day 3–5: Light return", "guidance": "Light walking, non-contact drills only. If symptom-free for 24h, progress to sport-specific movement."}
            ],
            "exercises": ["Neck isometric holds — 3×10s each direction", "Gentle cervical rotations — 10 reps each side", "Shoulder shrugs for neck tension relief — 3×15"],
        },
        "moderate": {
            "phases": [
                {"phase": "Day 1–3: Strict rest", "guidance": "No training, no screen time > 30 min. Concussion protocol must be followed — consult a doctor before any return to play."},
                {"phase": "Day 4–10: Graduated return", "guidance": "Only return with medical clearance. Follow stepwise concussion return-to-play protocol."}
            ],
            "exercises": ["No exercises until medically cleared", "Vestibular rehab exercises only if prescribed by physio"],
        },
        "severe": {
            "phases": [
                {"phase": "Immediate: See a doctor", "guidance": "Do NOT return to play. This severity requires neurological evaluation. Possible imaging needed."}
            ],
            "exercises": ["No self-guided exercises — await medical evaluation"],
        }
    },
    "Shoulder": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–3: Active rest", "guidance": "Avoid overhead throws and heavy lifting. Ice 15 min × 3/day. For cricket fast bowlers: no bowling for 3 days minimum."},
                {"phase": "Day 4–7: Rehab", "guidance": "Begin rotator cuff strengthening. For cricket, progress from gentle shadow bowling to half-pace bowling only."},
                {"phase": "Day 8–14: Return", "guidance": "Gradual return to full load over 1 week. Monitor for pain during acceleration phase of throw."}
            ],
            "exercises": ["Pendulum swings — 3×30s each direction", "External rotation with band — 3×15 reps", "Scapular retraction — 3×20 reps", "Prone Y-T-W raises — 2×10 reps"],
        },
        "moderate": {
            "phases": [
                {"phase": "Day 1–5: Rest + ice", "guidance": "Sling if needed for comfort. No bowling, throwing, or overhead activity. Kabaddi and wrestling athletes should avoid shoulder tackles."},
                {"phase": "Week 2–3: Progressive load", "guidance": "Physio-guided rotator cuff programme. Pain-free range of motion first before adding resistance."},
                {"phase": "Week 3–4: Sport return", "guidance": "Cricket fast bowlers: 50% pace bowling with bowling coach monitoring action mechanics."}
            ],
            "exercises": ["Isometric shoulder holds — 5×10s", "Supine external rotation — 3×15 (light band)", "Wall slides — 3×10", "Side-lying external rotation — 3×15"],
        },
        "severe": {
            "phases": [
                {"phase": "See doctor first", "guidance": "Possible rotator cuff tear or labral injury. MRI/ultrasound evaluation required. No return to sport until medical clearance."}
            ],
            "exercises": ["Doctor/physio-guided only — do not self-treat"],
        }
    },
    "Elbow/Forearm": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–3: Reduce load", "guidance": "For cricket bowlers: rest from bowling. Ice elbow post-session. Avoid wrist flexion under load — common cause in spin bowlers."},
                {"phase": "Day 4–7: Eccentric rehab", "guidance": "Start Tyler Twist or reverse wrist curl programme. Badminton players: reduce smash volume by 50%."}
            ],
            "exercises": ["Wrist flexor stretch — 3×30s", "Reverse wrist curls — 3×15 (light)", "Supination/pronation — 3×15 reps", "Eccentric wrist extension — 3×10 (Tyler Twist)"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Rest", "guidance": "No throwing or racquet sport. Ice 3×/day. Consider physio evaluation for tennis/golfer's elbow grading."},
                {"phase": "Week 2–3: Eccentric programme", "guidance": "Supervised eccentric loading. Cricket spinners: bowling action analysis recommended to prevent recurrence."}
            ],
            "exercises": ["Eccentric wrist extension — 3×15 daily", "Forearm massage — 5 min daily", "Grip strengthening — 3×20 reps"],
        },
        "severe": {
            "phases": [{"phase": "See doctor", "guidance": "Possible stress fracture or ligament damage (UCL). Common in fast bowlers. X-ray or MRI required."}],
            "exercises": ["Await medical evaluation before any loading"],
        }
    },
    "Wrist/Hand": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–4: Splint + rest", "guidance": "Buddy tape or wrist splint if needed. No batting or catching. Wrestlers: no gripping drills."},
                {"phase": "Day 5–10: Mobility", "guidance": "Gentle range of motion. Progress to light grip work. Cricket batsmen: shadow batting with no impact first."}
            ],
            "exercises": ["Wrist circles — 3×20", "Grip squeeze — 3×15 (soft ball)", "Finger extensions with band — 3×15"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Immobilisation", "guidance": "Splint full time except for hygiene. X-ray to rule out scaphoid fracture — very common and often missed."},
                {"phase": "Week 2–3: Progressive use", "guidance": "Only with pain-free movement. Scaphoid fractures need specialist care — do not rush return."}
            ],
            "exercises": ["Tendon gliding exercises — 3×10", "Wrist flexion/extension — pain-free range only"],
        },
        "severe": {
            "phases": [{"phase": "See doctor — possible fracture", "guidance": "Scaphoid and metacarpal fractures are common and serious. X-ray mandatory before any return."}],
            "exercises": ["None until cleared by doctor"],
        }
    },
    "Chest/Back": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–2: Unload", "guidance": "No heavy lifting or explosive twisting. For cricket fast bowlers: lower back strain is a career risk — treat seriously even if mild."},
                {"phase": "Day 3–7: Core activation", "guidance": "Begin McGill Big 3 (curl-up, bird-dog, side plank). Reduce bowling load by 50% for at least one week."},
                {"phase": "Day 7–14: Return", "guidance": "Bowling load management: 10-ball rule. Do not bowl more than 10 balls per spell until fully pain-free."}
            ],
            "exercises": ["McGill curl-up — 3×8", "Bird-dog — 3×10 each side", "Side plank — 3×20s", "Glute bridge — 3×15"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Rest + physio", "guidance": "Physio evaluation required. For cricketers, a stress fracture of the pars (common in fast bowlers) must be ruled out via MRI or CT."},
                {"phase": "Week 2–4: Structured rehab", "guidance": "Progressive core loading under guidance. Return to bowling only after pain-free batting and fielding."}
            ],
            "exercises": ["Dead bug — 3×10", "Pallof press — 3×12", "Hip hinge practice (pain-free only)", "Thoracic rotation — 3×10 each side"],
        },
        "severe": {
            "phases": [{"phase": "See doctor — possible stress fracture", "guidance": "MRI/CT required to rule out pars interarticularis stress fracture, disc herniation or rib injury. Common in young fast bowlers."}],
            "exercises": ["Await medical clearance"],
        }
    },
    "Thigh/Hamstring": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–2: RICE", "guidance": "Rest, Ice (15 min × 3), Compression, Elevation. No sprinting. Athletes: avoid full-speed runs for 48h minimum."},
                {"phase": "Day 3–5: Active recovery", "guidance": "Light cycling, walking. Begin gentle hamstring lengthening — no aggressive stretching of acute tears."},
                {"phase": "Day 6–10: Progressive return", "guidance": "Jogging, then striding, then sprint progression. Football/Athletics: complete sprint speed test before full return."}
            ],
            "exercises": ["Nordic hamstring curl — 3×8 (eccentric phase only)", "Romanian deadlift — 3×10 (light)", "Single-leg bridge — 3×12 each side", "Prone knee flexion — 3×15"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Rest + ultrasound", "guidance": "Ultrasound to grade tear. Grade 2 tears need 3–6 weeks. Athletics sprinters: complete rehabilitation before return — recurrence risk is high."},
                {"phase": "Week 2–3: Load introduction", "guidance": "Begin Nordic eccentric programme. Running athletes: Askling protocol recommended."},
                {"phase": "Week 3–5: Return to running", "guidance": "Progressive sprint programme. Return to match only with full sprint speed and strength symmetry."}
            ],
            "exercises": ["Nordics — 3×6 (eccentric)", "Stiff-leg deadlift — 3×10", "A-skip and B-skip drills", "Hip extension with band — 3×15"],
        },
        "severe": {
            "phases": [{"phase": "See doctor", "guidance": "Grade 3 tear or avulsion needs immediate imaging. Surgical consultation may be required. Do NOT stretch."}],
            "exercises": ["None until medically evaluated"],
        }
    },
    "Knee": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–3: Reduce load", "guidance": "Avoid deep squats and pivoting. Ice 3×/day. Football and Kabaddi athletes: avoid tackling and direction changes."},
                {"phase": "Day 4–10: Quad + VMO work", "guidance": "Strengthen vastus medialis. Step-downs and terminal knee extension. Pain-free cycling is OK."},
                {"phase": "Day 10–14: Sport return", "guidance": "Single-leg hop test before returning to match play. Any giving-way sensation = see doctor immediately."}
            ],
            "exercises": ["Terminal knee extension (TKE) — 3×20 with band", "Step-down — 3×12 each leg", "Wall sit — 3×30s", "Single-leg balance — 3×30s", "VMO quad set — 3×20"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Rest + evaluation", "guidance": "Physiotherapy evaluation mandatory. Swelling + instability = possible ligament tear. Lachman and pivot shift tests needed."},
                {"phase": "Week 2–4: Rehab", "guidance": "Guided quad/hamstring programme. Kabaddi and football athletes: ACL prevention programme (FIFA 11+) must be part of return."},
                {"phase": "Week 4–6: Return", "guidance": "Functional movement screen + single-leg hop test before return to match."}
            ],
            "exercises": ["SLR (straight leg raise) — 3×20", "Clamshells — 3×20 each side", "Stationary cycling — 15 min (pain-free only)", "Hip abduction with band — 3×15"],
        },
        "severe": {
            "phases": [{"phase": "See doctor — possible ACL/meniscus injury", "guidance": "Knee swelling + giving way = likely ligament injury. MRI required. Do NOT return to sport. ACL reconstruction may be needed."}],
            "exercises": ["None — await surgical or physio evaluation"],
        }
    },
    "Ankle/Foot": {
        "mild":     {
            "phases": [
                {"phase": "Day 1–2: RICE + tape", "guidance": "Rest, Ice, Compression, Elevation. Supportive taping. Most Grade 1 ankle sprains allow weight-bearing. Football athletes: tape for 2 weeks after return."},
                {"phase": "Day 3–5: Balance work", "guidance": "Single-leg balance, wobble board. Early proprioception training reduces re-injury risk by 50%."},
                {"phase": "Day 5–10: Return", "guidance": "Jog → cut → sprint progression. Tape or brace for first 4 weeks back on pitch."}
            ],
            "exercises": ["Single-leg balance — 3×30s (eyes open, then closed)", "Resistance band dorsiflexion — 3×20", "Heel raises — 3×20", "Figure-of-8 walking drills"],
        },
        "moderate": {
            "phases": [
                {"phase": "Week 1: Rest + X-ray", "guidance": "Ottawa rules: X-ray to rule out fracture (pain over malleolus bone + inability to bear weight = X-ray needed). Grade 2 sprain: 2–4 weeks."},
                {"phase": "Week 2–3: Proprioception", "guidance": "Wobble board, BOSU balance. Calf and peroneals strengthening. Athletics sprinters: ensure Achilles and peroneal tendons are pain-free before sprinting."},
                {"phase": "Week 3–4: Return to sport", "guidance": "Football/Kabaddi: agility drills before match clearance. Tape or brace for first month."}
            ],
            "exercises": ["Resistance band eversion — 3×20", "Peroneal strengthening — 3×15", "Calf raises (eccentric) — 3×15", "Star excursion balance test practice"],
        },
        "severe": {
            "phases": [{"phase": "See doctor — possible fracture", "guidance": "Inability to bear weight, bone tenderness over malleolus or 5th metatarsal = X-ray required. High ankle sprains (syndesmosis) need MRI."}],
            "exercises": ["None until fracture excluded by X-ray"],
        }
    }
}

NUTRITION_BY_TRIAGE = {
    "mild": [
        "Haldi doodh (turmeric milk) at night — curcumin reduces inflammation naturally",
        "Moong dal khichdi — easy protein + carbs to keep energy up during rest",
        "Banana + peanut butter as a snack — potassium for muscle cramps, protein for repair",
        "Amla (Indian gooseberry) or nimbu pani — Vitamin C speeds up collagen repair"
    ],
    "moderate": [
        "3 eggs or 150g paneer daily — protein is critical for tissue repair. Aim for 1.5g/kg bodyweight",
        "Haldi + adrak (ginger) chai morning and evening — both are natural anti-inflammatories",
        "Rajma or chana (chickpeas) with rice — complete protein + iron for recovery",
        "Avoid fried food and maida completely during recovery — they increase inflammatory markers"
    ],
    "severe": [
        "High protein diet is mandatory — 2g protein per kg bodyweight daily. Eggs, dal, paneer, chicken, fish",
        "Vitamin D: sit in morning sunlight 20 min/day + eat ragi (finger millet) — supports bone healing",
        "Haldi (turmeric) 1 tsp with black pepper in food daily — bioavailability of curcumin increases 2000% with piperine",
        "Hydration: 3–4 litres water daily — healing tissues need constant hydration. Add electrolytes via coconut water or ORS"
    ]
}


def encode_input(data):
    sport_enc   = SPORTS.index(data["sport"])
    part_enc    = BODY_PARTS.index(data["body_part"])
    sev_enc     = SEV_LABELS.index(data["self_severity"])
    row = [
        sport_enc, part_enc, sev_enc,
        data["age"], data["weight_kg"], data["fatigue_score"],
        data["training_days_per_week"], data["training_hours_per_week"],
        data["previous_injury"], data["match_within_48h"]
    ]
    return pd.DataFrame([row], columns=FEAT_NAMES, dtype=np.float32)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate
        required = ["sport","body_part","self_severity","age","weight_kg",
                    "fatigue_score","training_days_per_week","training_hours_per_week",
                    "previous_injury","match_within_48h"]
        for key in required:
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400

        X = encode_input(data)

        # Triage
        triage_idx   = int(triage_clf.predict(X)[0])
        triage_probs = triage_clf.predict_proba(X)[0].tolist()
        triage_label = ["mild","moderate","severe"][triage_idx]

        # Doctor flag (stacked)
        X_d = X.copy(); X_d["f10"] = np.float32(triage_idx)
        doc_idx   = int(doctor_clf.predict(X_d)[0])
        doc_prob  = doctor_clf.predict_proba(X_d)[0][1]
        req_doc   = bool(doc_idx)

        # Recovery days
        rec_days = int(max(1, min(120, recovery_reg.predict(X_d)[0])))

        # Protocol lookup
        protocol = RECOVERY_PROTOCOLS.get(data["body_part"], {}).get(triage_label, {})
        nutrition = NUTRITION_BY_TRIAGE[triage_label]

        return jsonify({
            "triage_level":    triage_label,
            "triage_probs":    {
                "mild":     round(triage_probs[0], 3),
                "moderate": round(triage_probs[1], 3),
                "severe":   round(triage_probs[2], 3)
            },
            "requires_doctor":   req_doc,
            "doctor_confidence": round(float(doc_prob), 3),
            "recovery_days":     rec_days,
            "recovery_phases":   protocol.get("phases", []),
            "recovery_exercises":protocol.get("exercises", []),
            "nutrition_tips":    nutrition,
            "confidence_note":   "Self-reported severity is the strongest model signal. Fatigue and training history improve accuracy."
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": ["triage_clf","doctor_clf","recovery_reg"]})


if __name__ == "__main__":
    print("AthleteEdge AI — Model Server")
    print("Listening on http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)