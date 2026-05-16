"""
AthleteEdge AI — Athlete Injury Text Description Dataset
=========================================================
5,000 rows of realistic free-text symptom descriptions written the way
Indian grassroots athletes actually describe pain — informal English,
Hindi-English mix, sport-specific language, varying detail levels.

Labels per row:
  body_part      : anatomical location
  triage_level   : mild / moderate / severe
  requires_doctor: 0 / 1
  sport          : sport context
"""

import csv, os, random, itertools
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Template bank ────────────────────────────────────────────────────────────
# Each entry: (template_string, body_part, triage, requires_doctor, sport_tags)
# {sport} {name} {time} placeholders filled at generation time

TEMPLATES = [

# ══════════════════════════════════════════════════════
# HEAD / NECK
# ══════════════════════════════════════════════════════
("got hit on the head while batting, feeling dizzy now, slight headache",
 "Head/Neck", "mild", 0, ["Cricket"]),
("headed the ball and felt a strong jerk in my neck, pain when turning left",
 "Head/Neck", "mild", 0, ["Football"]),
("neck pain since morning, slept in a bad position after match",
 "Head/Neck", "mild", 0, ["Cricket","Football","Kabaddi"]),
("got elbowed on my head during kabaddi, small cut but feeling fine, no dizziness",
 "Head/Neck", "mild", 0, ["Kabaddi"]),
("mild headache after fielding in sun for 3 hours, feeling tired",
 "Head/Neck", "mild", 0, ["Cricket"]),
("neck stiff since 2 days, pain when looking right side, no injury just overtraining",
 "Head/Neck", "mild", 0, ["Athletics","Wrestling"]),
("took a knock on the head during wrestling practice, felt okay but head hurts now",
 "Head/Neck", "moderate", 1, ["Wrestling"]),
("header collision with another player, felt confused for few seconds, headache started",
 "Head/Neck", "moderate", 1, ["Football"]),
("got hit by bouncer on helmet, neck snapped back, feeling nauseous and head is heavy",
 "Head/Neck", "moderate", 1, ["Cricket"]),
("collision with teammate while running for catch, both heads clashed, feeling lightheaded",
 "Head/Neck", "moderate", 1, ["Cricket","Football"]),
("neck pain very bad after being thrown in wrestling, cannot move neck fully, sharp pain",
 "Head/Neck", "moderate", 1, ["Wrestling"]),
("head clash in kabaddi raid, got up but vomited once, blurry vision for 1 minute",
 "Head/Neck", "severe", 1, ["Kabaddi"]),
("took a bad fall, hit back of head on ground, lost consciousness briefly, teammates say I was out for few seconds",
 "Head/Neck", "severe", 1, ["Football","Cricket","Kabaddi"]),
("batsman drove hard ball hit me on temple, no helmet, I fell down, everything went black",
 "Head/Neck", "severe", 1, ["Cricket"]),
("severe neck pain after tackle, cannot move my neck at all, tingling in both arms",
 "Head/Neck", "severe", 1, ["Football","Kabaddi","Wrestling"]),

# ══════════════════════════════════════════════════════
# SHOULDER
# ══════════════════════════════════════════════════════
("shoulder pain when bowling fast, pain only at top of action when I release the ball",
 "Shoulder", "mild", 0, ["Cricket"]),
("little bit of shoulder soreness after long batting session, goes away with rest",
 "Shoulder", "mild", 0, ["Cricket"]),
("shoulder feeling tight during smash, no sharp pain just discomfort",
 "Shoulder", "mild", 0, ["Badminton"]),
("shoulder pain only when lifting arm above head, fine otherwise, started 3 days ago",
 "Shoulder", "mild", 0, ["Cricket","Badminton","Athletics"]),
("minor shoulder ache after wrestling bout, no swelling, I can move it fully",
 "Shoulder", "mild", 0, ["Wrestling"]),
("right shoulder pain for 1 week, slight ache during throwing, came gradually not from one injury",
 "Shoulder", "mild", 0, ["Cricket","Football"]),
("shoulder popped during kabaddi escape, heard a click, moderate pain now, arm feels weak",
 "Shoulder", "moderate", 1, ["Kabaddi"]),
("took a fall on outstretched hand, shoulder hurts bad when I try to raise arm",
 "Shoulder", "moderate", 1, ["Cricket","Football"]),
("bowling shoulder pain getting worse every session, now hurts even while combing hair",
 "Shoulder", "moderate", 1, ["Cricket"]),
("shoulder pain after smash, swelling visible on front of joint, painful to touch",
 "Shoulder", "moderate", 1, ["Badminton"]),
("shoulder thrown out during wrestling, coach pushed it back in, very sore now, limited movement",
 "Shoulder", "moderate", 1, ["Wrestling"]),
("fell on shoulder while diving in field, heard a pop, cannot lift arm above chest level, lot of pain",
 "Shoulder", "severe", 1, ["Cricket","Football"]),
("shoulder dislocated during bout, popped back by coach but arm is totally weak, any movement is painful",
 "Shoulder", "severe", 1, ["Wrestling","Kabaddi"]),
("sharp pain in shoulder during bowling, felt something tear, cannot bowl at all, severe pain",
 "Shoulder", "severe", 1, ["Cricket"]),
("shoulder smash injury, unable to serve even, pain at night waking me up, swelling increasing",
 "Shoulder", "severe", 1, ["Badminton"]),

# ══════════════════════════════════════════════════════
# ELBOW / FOREARM
# ══════════════════════════════════════════════════════
("outer elbow pain after long batting session, pain when holding bat tight",
 "Elbow/Forearm", "mild", 0, ["Cricket"]),
("elbow soreness after many smashes in badminton, goes away overnight",
 "Elbow/Forearm", "mild", 0, ["Badminton"]),
("forearm tightness after long bowling spell, not a sharp pain just fatigue",
 "Elbow/Forearm", "mild", 0, ["Cricket"]),
("elbow clicks sometimes when throwing, no pain, been happening for 2 weeks",
 "Elbow/Forearm", "mild", 0, ["Cricket","Football"]),
("tennis elbow kind of pain on outer side, started slowly over last month",
 "Elbow/Forearm", "mild", 0, ["Badminton","Cricket"]),
("bowling a lot this week, inside elbow hurts during follow through",
 "Elbow/Forearm", "mild", 0, ["Cricket"]),
("elbow pain very bad after bowling 20 overs in match, swollen slightly, tender to touch",
 "Elbow/Forearm", "moderate", 1, ["Cricket"]),
("elbow swollen and painful after opposing player landed on my arm in kabaddi",
 "Elbow/Forearm", "moderate", 1, ["Kabaddi"]),
("forearm pain and weakness, cannot grip racquet properly, pain radiates to wrist",
 "Elbow/Forearm", "moderate", 1, ["Badminton"]),
("spin bowling causing sharp inside elbow pain on every delivery, 3 weeks now not getting better",
 "Elbow/Forearm", "moderate", 1, ["Cricket"]),
("elbow bent backwards in wrestling accident, extreme pain, deformity visible",
 "Elbow/Forearm", "severe", 1, ["Wrestling"]),
("heard a loud pop in elbow during fast bowling, cannot straighten arm, severe pain",
 "Elbow/Forearm", "severe", 1, ["Cricket"]),
("elbow hit hard by ball, bone very tender, possible fracture, cannot move it",
 "Elbow/Forearm", "severe", 1, ["Cricket","Football"]),

# ══════════════════════════════════════════════════════
# WRIST / HAND
# ══════════════════════════════════════════════════════
("wrist pain after batting, pain on thumb side when I grip bat",
 "Wrist/Hand", "mild", 0, ["Cricket"]),
("finger sprain while catching, buddy taped it, mild swelling",
 "Wrist/Hand", "mild", 0, ["Cricket","Football","Kabaddi"]),
("wrist sore after lots of smashes, pain on backhand side only",
 "Wrist/Hand", "mild", 0, ["Badminton"]),
("index finger jammed while keeping wickets, swollen and stiff but I can move it",
 "Wrist/Hand", "mild", 0, ["Cricket"]),
("wrist hurts mildly when doing push-ups, fine during play",
 "Wrist/Hand", "mild", 0, ["Wrestling","Athletics"]),
("hand swollen after being stood on during kabaddi, bruised badly",
 "Wrist/Hand", "moderate", 1, ["Kabaddi"]),
("wrist fell hard on ground diving, painful to rotate, swelling increasing",
 "Wrist/Hand", "moderate", 1, ["Cricket","Football"]),
("thumb bent wrong way in wrestling grip, pain when pinching, cannot make full fist",
 "Wrist/Hand", "moderate", 1, ["Wrestling"]),
("wrist pain on thumb side for 3 weeks, gets worse with any twisting movement",
 "Wrist/Hand", "moderate", 1, ["Badminton","Cricket"]),
("finger bent completely backwards catching hard throw, immediate severe pain and deformity",
 "Wrist/Hand", "severe", 1, ["Cricket","Football"]),
("wrist cannot bear any weight after fall, bone tender over small bump near thumb, possible scaphoid",
 "Wrist/Hand", "severe", 1, ["Cricket","Football","Kabaddi"]),
("hand crushed under opponent weight in wrestling, multiple fingers swollen, cannot move them",
 "Wrist/Hand", "severe", 1, ["Wrestling"]),

# ══════════════════════════════════════════════════════
# CHEST / BACK
# ══════════════════════════════════════════════════════
("lower back tight after bowling 15 overs, goes with rest and hot water bottle",
 "Chest/Back", "mild", 0, ["Cricket"]),
("back stiffness in morning after hard training, better after warmup",
 "Chest/Back", "mild", 0, ["Cricket","Football","Athletics","Kabaddi"]),
("rib soreness after being tackled, tender to touch but breathing ok",
 "Chest/Back", "mild", 0, ["Football","Kabaddi"]),
("upper back tight from too many smashes this week",
 "Chest/Back", "mild", 0, ["Badminton"]),
("mild lower back ache after running 15km in practice, nothing serious I think",
 "Chest/Back", "mild", 0, ["Athletics"]),
("lower back pain while bowling for last 2 weeks, not improving, slight pain at night too",
 "Chest/Back", "moderate", 1, ["Cricket"]),
("back spasm during heavy lifting session, cannot bend forward fully",
 "Chest/Back", "moderate", 1, ["Wrestling","Athletics"]),
("rib pain when breathing deep after getting hit in kabaddi, worried it might be cracked",
 "Chest/Back", "moderate", 1, ["Kabaddi"]),
("lower back pain radiating to left leg, like electric shock when I sneeze",
 "Chest/Back", "moderate", 1, ["Athletics","Cricket"]),
("severe lower back pain after landing awkwardly from jump, cannot stand straight",
 "Chest/Back", "severe", 1, ["Athletics","Basketball"]),
("back pain so bad cannot get out of bed, happened after bowling on hard pitch, shooting pain down leg",
 "Chest/Back", "severe", 1, ["Cricket"]),
("took knee to ribs in kabaddi, sharp chest pain, difficult to breathe fully",
 "Chest/Back", "severe", 1, ["Kabaddi","Football"]),
("stress fracture suspected in lower back, MRI pending, extreme pain every ball I bowl",
 "Chest/Back", "severe", 1, ["Cricket"]),

# ══════════════════════════════════════════════════════
# THIGH / HAMSTRING
# ══════════════════════════════════════════════════════
("thigh feels tight after sprint training, goes away with stretching",
 "Thigh/Hamstring", "mild", 0, ["Athletics","Football"]),
("mild pull feeling in hamstring after quick single, not bad just uncomfortable",
 "Thigh/Hamstring", "mild", 0, ["Cricket"]),
("quad sore from kabaddi session, DOMS type pain not injury",
 "Thigh/Hamstring", "mild", 0, ["Kabaddi"]),
("hamstring tightness only in morning, fine once warmed up",
 "Thigh/Hamstring", "mild", 0, ["Athletics","Football"]),
("felt a small pull in back of thigh while sprinting, stopped running, dull ache now",
 "Thigh/Hamstring", "moderate", 1, ["Athletics","Football","Cricket"]),
("hamstring popped during sprint race, fell down immediately, pain moderate, bruising started",
 "Thigh/Hamstring", "moderate", 1, ["Athletics"]),
("inner thigh strain from kabaddi splits, walking with limp, swollen",
 "Thigh/Hamstring", "moderate", 1, ["Kabaddi"]),
("hamstring has been niggling for 2 weeks, now got worse during match sprint",
 "Thigh/Hamstring", "moderate", 1, ["Football","Athletics"]),
("heard a loud pop in hamstring during sprint, collapsed, cannot put weight on leg, severe bruising forming",
 "Thigh/Hamstring", "severe", 1, ["Athletics","Football"]),
("complete hamstring rupture feeling, back of thigh has a gap I can feel, unbearable pain",
 "Thigh/Hamstring", "severe", 1, ["Athletics"]),
("groin tear during kabaddi escape, shooting pain from groin to knee, cannot lift leg",
 "Thigh/Hamstring", "severe", 1, ["Kabaddi"]),

# ══════════════════════════════════════════════════════
# KNEE
# ══════════════════════════════════════════════════════
("knee pain after long batting session, front of knee hurts going down stairs",
 "Knee", "mild", 0, ["Cricket"]),
("knee slightly sore after football practice, no swelling, fine after ice",
 "Knee", "mild", 0, ["Football"]),
("kneecap area pain when running, getting better with rest, nothing alarming",
 "Knee", "mild", 0, ["Athletics","Football"]),
("inner knee discomfort after kabaddi match, mild, no swelling, can do full squat",
 "Knee", "mild", 0, ["Kabaddi"]),
("knee pain only when climbing stairs, flat ground fine, no trauma",
 "Knee", "mild", 0, ["Athletics","Cricket"]),
("runner's knee kind of pain in badminton, front of knee hurts during lunges",
 "Knee", "mild", 0, ["Badminton"]),
("knee twisted in kabaddi tackle, moderate pain, some swelling visible, walking with slight limp",
 "Knee", "moderate", 1, ["Kabaddi"]),
("knee swollen after football, happened when I landed from jump, no pop heard",
 "Knee", "moderate", 1, ["Football"]),
("kneecap shifted during wrestling throw, very painful, I pushed it back, now very sore",
 "Knee", "moderate", 1, ["Wrestling"]),
("knee pain for 3 weeks not going away, giving way sometimes on uneven ground",
 "Knee", "moderate", 1, ["Football","Cricket","Kabaddi"]),
("knee locked up during match, could not straighten it, had to be helped off",
 "Knee", "moderate", 1, ["Football","Cricket"]),
("heard loud pop in knee changing direction, fell immediately, knee swelling rapidly, cannot bear weight",
 "Knee", "severe", 1, ["Football","Kabaddi","Cricket"]),
("knee twisted badly in wrestling, huge swelling within 10 minutes, completely unstable",
 "Knee", "severe", 1, ["Wrestling"]),
("ACL injury I think, same feeling as my teammate described, knee gave out completely during sprint turn",
 "Knee", "severe", 1, ["Football","Athletics"]),
("knee hit hard by ball while fielding, bone very tender, swelling huge, cannot bend at all",
 "Knee", "severe", 1, ["Cricket"]),
("meniscus tear feeling, knee locked completely, excruciating pain, cannot walk at all",
 "Knee", "severe", 1, ["Football","Kabaddi"]),

# ══════════════════════════════════════════════════════
# ANKLE / FOOT
# ══════════════════════════════════════════════════════
("ankle twisted while running between wickets, mild sprain, slight swelling",
 "Ankle/Foot", "mild", 0, ["Cricket"]),
("foot blisters from new football boots, painful but not an injury as such",
 "Ankle/Foot", "mild", 0, ["Football"]),
("ankle little sore after match, same ankle I sprained 3 months ago",
 "Ankle/Foot", "mild", 0, ["Football","Kabaddi"]),
("heel pain in morning, first steps very painful, goes away after 10 minutes",
 "Ankle/Foot", "mild", 0, ["Athletics","Cricket"]),
("ankle stiff after long athletics training, goes with ice and stretching",
 "Ankle/Foot", "mild", 0, ["Athletics"]),
("minor ankle roll during badminton lunge, taped it, playing tomorrow",
 "Ankle/Foot", "mild", 0, ["Badminton"]),
("ankle rolled badly in kabaddi, pain moderate, cannot run at full speed, swelling on outer side",
 "Ankle/Foot", "moderate", 1, ["Kabaddi"]),
("foot pain on outside after long run, tender bone area, might be stress fracture",
 "Ankle/Foot", "moderate", 1, ["Athletics"]),
("ankle sprain during football match, swollen and bruised, limping but can bear weight",
 "Ankle/Foot", "moderate", 1, ["Football"]),
("Achilles very tight and painful after sprint session, painful to walk on toes",
 "Ankle/Foot", "moderate", 1, ["Athletics","Cricket"]),
("ankle twisted and heard a crack or pop, fell down, outer ankle very swollen and bruised",
 "Ankle/Foot", "severe", 1, ["Football","Kabaddi","Cricket"]),
("foot stepped in hole on ground, heard snap, cannot put any weight at all, severe pain",
 "Ankle/Foot", "severe", 1, ["Cricket","Football","Athletics"]),
("Achilles tendon snap during sprint, felt like someone kicked back of my leg, fell down, can't walk",
 "Ankle/Foot", "severe", 1, ["Athletics","Football"]),
("ankle completely rolled over in wrestling, immediate severe swelling, bone deformity visible",
 "Ankle/Foot", "severe", 1, ["Wrestling"]),

]

# ── Augmentation helpers ─────────────────────────────────────────────────────

SPORT_CONTEXTS = {
    "Cricket":   ["after bowling a long spell","during batting practice","while fielding","during a match","after net session","in a tournament game","while running between wickets"],
    "Football":  ["during a match","in training session","while sprinting","after a tackle","during a corner jump","in 5-a-side game","during dribbling drill"],
    "Kabaddi":   ["during a raid","while defending","in a match","during practice bout","while attempting a tackle","during district tournament","in school match"],
    "Athletics": ["during sprint race","in 100m practice","after long run","during hurdles","in field training","while doing intervals","after 5km run"],
    "Wrestling": ["during a bout","in practice match","while being thrown","during grappling drill","in district championship","while attempting takedown","during mat training"],
    "Badminton": ["during a rally","while playing smash","during footwork drill","in match","while playing doubles","during training session","after long practice"],
}

DURATION_PHRASES = [
    "since yesterday","for 2 days now","since this morning","after last match","for a week now",
    "since practice yesterday","started 3 days ago","from this evening","since last night",
    "for past 4 days","since the tournament","started 2 weeks ago","from this afternoon",
]

HEDGE_PHRASES = [
    "","","","",  # most have no hedge
    "I think ","not sure but ","feels like ","might be ",
]

SEVERITY_MODS_MILD = [
    "not too serious","nothing major","manageable pain","mild discomfort",
    "slight pain","little bit of pain","minor issue I think","small problem",
    "dull ache","low level pain",
]

SEVERITY_MODS_MODERATE = [
    "moderate pain","quite painful","significant discomfort","can't play properly",
    "affecting my game","limiting my movement","painful enough to slow me down",
    "noticeable pain","bothering me a lot",
]

SEVERITY_MODS_SEVERE = [
    "very severe pain","unbearable","cannot continue playing","extremely painful",
    "worst pain I've felt","need help urgently","very serious","cannot walk/move",
    "sharp stabbing pain","immediate severe pain",
]

HINDI_INSERTS = [
    ("",""),  # most entries are pure English
    ("bahut","very"),
    ("thoda","a little"),
    ("dard","pain"),
    ("problem","problem"),
    ("theek nahi","not okay"),
]

AGE_DESCRIPTIONS = [
    "I am 16 years old","I'm 19","18 year old","20 yr old","I'm 22","17 years","I'm 24",
    "I am 15","23 years old","I'm 17",
]

def augment(template_text, body_part, triage, requires_doctor, sports):
    sport = random.choice(sports)
    context = random.choice(SPORT_CONTEXTS.get(sport, ["during practice"]))
    duration = random.choice(DURATION_PHRASES) if random.random() < 0.5 else ""
    hedge = random.choice(HEDGE_PHRASES)
    age_desc = random.choice(AGE_DESCRIPTIONS) if random.random() < 0.25 else ""

    text = hedge + template_text
    if duration:
        text = text + ", " + duration
    if age_desc:
        text = age_desc + ". " + text
    if random.random() < 0.3:
        text = text + " " + context

    # minor noise: capitalisation and punctuation variation
    if random.random() < 0.3:
        text = text[0].upper() + text[1:]
    if random.random() < 0.2:
        text = text.rstrip(".") + "."

    return {
        "text": text,
        "body_part": body_part,
        "triage_level": triage,
        "requires_doctor": requires_doctor,
        "sport": sport,
    }

# ── Generate dataset ─────────────────────────────────────────────────────────
TARGET = 5000
rows = []

# First pass: one row per template per sport mentioned
for (tmpl, bp, tri, doc, sports) in TEMPLATES:
    for sport in sports:
        rows.append(augment(tmpl, bp, tri, doc, [sport]))

print(f"After first pass: {len(rows)} rows")

# Augment until we hit TARGET
while len(rows) < TARGET:
    tmpl, bp, tri, doc, sports = random.choice(TEMPLATES)
    rows.append(augment(tmpl, bp, tri, doc, sports))

random.shuffle(rows)
rows = rows[:TARGET]

# ── Write CSV ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "injury_text_dataset.csv")
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text","body_part","triage_level","requires_doctor","sport"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT}  ({len(rows)} rows)")

import pandas as pd
df = pd.read_csv(OUT)
print("\nClass distribution — triage_level:")
print(df["triage_level"].value_counts())
print("\nClass distribution — body_part:")
print(df["body_part"].value_counts())
print("\nClass distribution — requires_doctor:")
print(df["requires_doctor"].value_counts())
print("\nSample descriptions:")
for _, row in df.sample(8, random_state=1).iterrows():
    print(f'  [{row["triage_level"]:8s}] [{row["body_part"]:18s}] {row["text"]}')
