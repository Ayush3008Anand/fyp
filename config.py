from pathlib import Path

MIMIC = Path(
    r"E:\downloads\mimic-iv-3.1\mimic-iv-3.1"
)

PROJECT = Path(r"E:\endsemproject\project")

HOSP = MIMIC / "hosp"
ICU = MIMIC / "icu"

DATA = PROJECT / "data"

CANDIDATES = DATA / "candidates"
TROPONIN = DATA / "troponin"
ADJUDICATION = DATA / "adjudication"
FINAL = DATA / "final"

for folder in [
    CANDIDATES,
    TROPONIN,
    ADJUDICATION,
    FINAL
]:
    folder.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000

VITAL_ITEM_IDS = {
    220045, 220050, 220051, 220052,
    220179, 220180, 220181, 220277
}

PREDICTION_HOURS = 6