import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE, "injury_text_dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(DATASET_PATH)

# Dataset columns:
# text, body_part, triage_level, requires_doctor, sport

required_columns = ["text", "body_part", "triage_level", "requires_doctor", "sport"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns].dropna().copy()

# =========================
# TEXT PREPROCESSING
# =========================

def clean_text(text):
    text = str(text).lower()

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["text"] = df["text"].apply(clean_text)

# =========================
# LABEL ENCODING
# =========================

body_encoder = LabelEncoder()
triage_encoder = LabelEncoder()
sport_encoder = LabelEncoder()

df["body_part"] = body_encoder.fit_transform(df["body_part"])
df["triage_level"] = triage_encoder.fit_transform(df["triage_level"])
df["sport"] = sport_encoder.fit_transform(df["sport"])

# requires_doctor already numeric

# =========================
# FEATURES + TARGETS
# =========================

X = df["text"]

y = df[[
    "body_part",
    "triage_level",
    "requires_doctor",
    "sport"
]]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL PIPELINE
# =========================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultiOutputClassifier(
            LogisticRegression(max_iter=2000)
        )
    )
])

# =========================
# TRAIN MODEL
# =========================

print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")

# =========================
# EVALUATION
# =========================

predictions = model.predict(X_test)

target_names = [
    "body_part",
    "triage_level",
    "requires_doctor",
    "sport"
]

for i, target in enumerate(target_names):

    print(f"\n====================")
    print(f"{target.upper()} REPORT")
    print(f"====================")

    print(
        classification_report(
            y_test.iloc[:, i],
            predictions[:, i]
        )
    )

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, os.path.join(MODEL_DIR, "sports_injury_nlp_model.pkl"))

joblib.dump(body_encoder, os.path.join(MODEL_DIR, "body_encoder.pkl"))
joblib.dump(triage_encoder, os.path.join(MODEL_DIR, "triage_encoder.pkl"))
joblib.dump(sport_encoder, os.path.join(MODEL_DIR, "sport_encoder.pkl"))

print("\nModel saved successfully!")

# =========================
# PREDICTION FUNCTION
# =========================

def predict_injury(text):

    # clean input
    text = clean_text(text)

    # predict
    pred = model.predict([text])[0]

    # decode outputs
    body_part = body_encoder.inverse_transform([pred[0]])[0]
    triage_level = triage_encoder.inverse_transform([pred[1]])[0]
    requires_doctor = int(pred[2])
    sport = sport_encoder.inverse_transform([pred[3]])[0]

    print("\n====================")
    print("PREDICTION")
    print("====================")

    print("Body Part :", body_part)
    print("Triage    :", triage_level)
    print("Doctor    :", requires_doctor)
    print("Sport     :", sport)


# =========================
# TEST PREDICTION
# =========================

sample_text = """
knee twisted during football match,
huge swelling and cannot walk properly
"""

predict_injury(sample_text)
