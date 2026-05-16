"""
AthleteEdge AI — Injury Triage ML Model Training
=================================================
Exports:
  models/triage_clf.joblib      - triage level (mild/moderate/severe)
  models/doctor_clf.joblib      - requires_doctor flag
  models/recovery_reg.joblib    - recovery days estimate
  models/triage_clf.json        - XGBoost native JSON (for mobile/ONNX later)
  models/doctor_clf.json
  models/recovery_reg.json
  models/label_maps.json        - encode/decode for the app
  models/model_report.txt
"""

import json, os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import warnings; warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
DATASET_PATH = os.path.join(BASE, "dataset.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
REPORT = []
def log(m): print(m); REPORT.append(str(m))

df = pd.read_csv(DATASET_PATH)
log(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

FEATURES = [
    "sport_enc", "body_part_enc", "self_severity_enc",
    "age", "weight_kg", "fatigue_score",
    "training_days_per_week", "training_hours_per_week",
    "previous_injury", "match_within_48h"
]
# Use f0..f9 column names to satisfy XGBoost native JSON export
FEAT_NAMES = [f"f{i}" for i in range(len(FEATURES))]

X = df[FEATURES].astype(np.float32)
X.columns = FEAT_NAMES

y_triage   = df["triage_level"].astype(int)
y_doctor   = df["requires_doctor"].astype(int)
y_recovery = df["recovery_days"].astype(np.float32)

X_tr, X_te, yt_tr, yt_te, yd_tr, yd_te, yr_tr, yr_te = train_test_split(
    X, y_triage, y_doctor, y_recovery,
    test_size=0.2, random_state=42, stratify=y_triage
)
log(f"Train: {len(X_tr)}  Test: {len(X_te)}")

# ── 1. Triage Classifier ────────────────────────────────────────────────────
log("\n=== TRIAGE CLASSIFIER ===")
triage_clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42, n_jobs=-1
)
triage_clf.fit(X_tr, yt_tr, eval_set=[(X_te, yt_te)], verbose=False)
yt_pred = triage_clf.predict(X_te)
log(classification_report(yt_te, yt_pred, target_names=["mild","moderate","severe"]))
cv = cross_val_score(triage_clf, X, y_triage, cv=5, scoring="f1_weighted")
log(f"5-fold CV F1: {cv.mean():.3f} ± {cv.std():.3f}")
joblib.dump(triage_clf, os.path.join(MODEL_DIR, "triage_clf.joblib"))
triage_clf.save_model(os.path.join(MODEL_DIR, "triage_clf.json"))
log("Saved: triage_clf.joblib + triage_clf.json")

# ── 2. Doctor Classifier (stacked on triage prediction) ─────────────────────
log("\n=== REQUIRES-DOCTOR CLASSIFIER ===")
X_tr_d = X_tr.copy(); X_tr_d["f10"] = triage_clf.predict(X_tr).astype(np.float32)
X_te_d  = X_te.copy(); X_te_d["f10"]  = yt_pred.astype(np.float32)

doctor_clf = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.1,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
doctor_clf.fit(X_tr_d, yd_tr, eval_set=[(X_te_d, yd_te)], verbose=False)
yd_pred = doctor_clf.predict(X_te_d)
log(classification_report(yd_te, yd_pred, target_names=["no_doctor","see_doctor"]))
joblib.dump(doctor_clf, os.path.join(MODEL_DIR, "doctor_clf.joblib"))
doctor_clf.save_model(os.path.join(MODEL_DIR, "doctor_clf.json"))
log("Saved: doctor_clf.joblib + doctor_clf.json")

# ── 3. Recovery Regressor ────────────────────────────────────────────────────
log("\n=== RECOVERY DAYS REGRESSOR ===")
recovery_reg = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
recovery_reg.fit(X_tr_d, yr_tr, eval_set=[(X_te_d, yr_te)], verbose=False)
yr_pred = np.clip(recovery_reg.predict(X_te_d), 1, 120)
log(f"MAE: {mean_absolute_error(yr_te, yr_pred):.1f} days  R²: {r2_score(yr_te, yr_pred):.3f}")
joblib.dump(recovery_reg, os.path.join(MODEL_DIR, "recovery_reg.joblib"))
recovery_reg.save_model(os.path.join(MODEL_DIR, "recovery_reg.json"))
log("Saved: recovery_reg.joblib + recovery_reg.json")

# ── Feature importance ───────────────────────────────────────────────────────
log("\n=== FEATURE IMPORTANCE (triage model) ===")
for name, score in sorted(zip(FEATURES, triage_clf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(score * 200)
    log(f"  {name:<30s} {score:.4f}  {bar}")

# ── Label maps ───────────────────────────────────────────────────────────────
label_maps = {
    "features": FEATURES,
    "feature_names_model": FEAT_NAMES,
    "sports":      ["Cricket","Football","Kabaddi","Athletics","Wrestling","Badminton"],
    "body_parts":  ["Head/Neck","Shoulder","Elbow/Forearm","Wrist/Hand",
                    "Chest/Back","Thigh/Hamstring","Knee","Ankle/Foot"],
    "severity_labels": ["mild","moderate","severe"],
    "triage_labels":   ["mild","moderate","severe"],
    "doctor_labels":   ["no_doctor","see_doctor"],
    "usage_note": "Encode sport/body_part/severity by index; pass 11-feature vector (f0..f9 + f10=triage_pred) to doctor/recovery models"
}
with open(os.path.join(MODEL_DIR, "label_maps.json"),"w") as f:
    json.dump(label_maps, f, indent=2)
log("Saved: label_maps.json")

# ── Model sizes ──────────────────────────────────────────────────────────────
log("\n=== MODEL FILE SIZES ===")
for fname in ["triage_clf.json","doctor_clf.json","recovery_reg.json",
              "triage_clf.joblib","doctor_clf.joblib","recovery_reg.joblib"]:
    path = os.path.join(MODEL_DIR, fname)
    size = os.path.getsize(path) / 1024
    log(f"  {fname:<30s} {size:.0f} KB")

# ── Smoke test ───────────────────────────────────────────────────────────────
log("\n=== SMOKE TEST ===")
sample = pd.DataFrame([[2,6,1,19,72,7,5,18,0,1]], columns=FEAT_NAMES)
t_pred = triage_clf.predict(sample)[0]
t_prob = triage_clf.predict_proba(sample)[0]
log(f"Input: Kabaddi / Knee / moderate / fatigue=7 / training_days=5")
log(f"Triage: {['mild','moderate','severe'][t_pred]}  probs={[round(p,3) for p in t_prob]}")

sample_d = sample.copy(); sample_d["f10"] = t_pred
d_pred = doctor_clf.predict(sample_d)[0]
log(f"Doctor: {'SEE DOCTOR' if d_pred else 'No doctor needed'}")
r_pred = int(max(1, recovery_reg.predict(sample_d)[0]))
log(f"Recovery: ~{r_pred} days")

with open(os.path.join(MODEL_DIR, "model_report.txt"),"w") as f:
    f.write("\n".join(REPORT))
log(f"\nDone - all models in {MODEL_DIR}")
