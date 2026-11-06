
import sys
from pathlib import Path

import pandas as pd

# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import CANDIDATES, TROPONIN, ADJUDICATION

DIAGNOSIS_FILE = CANDIDATES / "cardiac_admission_diagnoses.csv"
ADMISSIONS_FILE = CANDIDATES / "cardiac_admissions.csv"
TROPONIN_FILE = TROPONIN / "cardiac_patient_troponin.csv"

OUTPUT_FILE = ADJUDICATION / "cardiac_adjudication_review.csv"


# ==================================================
# STEP 1: LOAD CARDIAC ADMISSIONS
# ==================================================

print("\n[1/5] Loading cardiac admissions...")

admissions = pd.read_csv(
    ADMISSIONS_FILE,
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime"
    ],
    dtype={
        "subject_id": "int32",
        "hadm_id": "int32"
    }
)

admissions = admissions.drop_duplicates(
    subset=["subject_id", "hadm_id"]
).copy()

admissions["admittime"] = pd.to_datetime(
    admissions["admittime"],
    errors="coerce"
)

admissions["dischtime"] = pd.to_datetime(
    admissions["dischtime"],
    errors="coerce"
)

print("Candidate admissions:", len(admissions))
print("Distinct patients:", admissions["subject_id"].nunique())


# ==================================================
# STEP 2: IDENTIFY AMI DIAGNOSIS CODES
# ==================================================

print("\n[2/5] Identifying AMI diagnosis codes...")

diagnoses = pd.read_csv(
    DIAGNOSIS_FILE,
    usecols=[
        "subject_id",
        "hadm_id",
        "icd_code",
        "icd_version"
    ],
    dtype={
        "subject_id": "int32",
        "hadm_id": "int32",
        "icd_code": "string",
        "icd_version": "Int64"
    }
)

# Normalize ICD codes by removing decimal points.
diagnoses["icd_code"] = (
    diagnoses["icd_code"]
    .str.upper()
    .str.replace(".", "", regex=False)
    .str.strip()
)

# ICD-10: I21 and I22
# ICD-9: 410
version = diagnoses["icd_version"]
code = diagnoses["icd_code"]

ami_mask = (
    ((version == 10) & code.str.match(r"^I(?:21|22)", na=False))
    |
    ((version == 9) & code.str.match(r"^410", na=False))
)

ami_diagnoses = diagnoses.loc[ami_mask].copy()

# Summarize AMI diagnosis codes for each admission.
ami_summary = (
    ami_diagnoses.groupby(["subject_id", "hadm_id"])
    .agg(
        ami_code_count=("icd_code", "count"),
        ami_codes=(
            "icd_code",
            lambda x: "; ".join(sorted(set(x.dropna())))
        )
    )
    .reset_index()
)

admissions = admissions.merge(
    ami_summary,
    on=["subject_id", "hadm_id"],
    how="left"
)

admissions["ami_code_count"] = (
    admissions["ami_code_count"]
    .fillna(0)
    .astype(int)
)

admissions["ami_codes"] = (
    admissions["ami_codes"].fillna("")
)

admissions["ami_diagnosis_code_found"] = (
    admissions["ami_code_count"] > 0
)

print("Admissions with AMI diagnosis codes:",
      admissions["ami_diagnosis_code_found"].sum())


# ==================================================
# STEP 3: SUMMARIZE TROPONIN MEASUREMENTS
# ==================================================

print("\n[3/5] Processing troponin measurements...")

if TROPONIN_FILE.exists():

    troponin = pd.read_csv(
        TROPONIN_FILE,
        usecols=[
            "subject_id",
            "hadm_id",
            "charttime",
            "itemid",
            "test_label",
            "valuenum",
            "value",
            "valueuom"
        ],
        dtype={
            "subject_id": "Int64",
            "hadm_id": "Int64"
        }
    )

    troponin["charttime"] = pd.to_datetime(
        troponin["charttime"],
        errors="coerce"
    )

    troponin["valuenum"] = pd.to_numeric(
        troponin["valuenum"],
        errors="coerce"
    )

    # Keep only measurements belonging to candidate admissions.
    troponin = troponin.merge(
        admissions[["subject_id", "hadm_id"]],
        on=["subject_id", "hadm_id"],
        how="inner"
    )

    troponin_summary = (
        troponin.groupby(["subject_id", "hadm_id"])
        .agg(
            troponin_measurement_count=("itemid", "count"),
            troponin_numeric_count=("valuenum", "count"),
            first_troponin_time=("charttime", "min"),
            last_troponin_time=("charttime", "max"),
            troponin_test_names=(
                "test_label",
                lambda x: "; ".join(sorted(set(x.dropna().astype(str))))
            )
        )
        .reset_index()
    )

    admissions = admissions.merge(
        troponin_summary,
        on=["subject_id", "hadm_id"],
        how="left"
    )

else:

    print("WARNING: Troponin file not found.")

    admissions["troponin_measurement_count"] = 0
    admissions["troponin_numeric_count"] = 0
    admissions["first_troponin_time"] = pd.NaT
    admissions["last_troponin_time"] = pd.NaT
    admissions["troponin_test_names"] = ""

# Fill missing summary values.
for column in [
    "troponin_measurement_count",
    "troponin_numeric_count"
]:
    admissions[column] = (
        admissions[column].fillna(0).astype(int)
    )

admissions["troponin_test_names"] = (
    admissions["troponin_test_names"].fillna("")
)

print(
    "Admissions with troponin measurements:",
    (admissions["troponin_measurement_count"] > 0).sum()
)


# ==================================================
# STEP 4: CREATE CLINICAL ADJUDICATION COLUMNS
# ==================================================

print("\n[4/5] Creating clinical review columns...")

# These must be completed using clinical evidence.
# Do not automatically classify based on diagnosis codes
# or troponin values alone.

admissions["cardiac_condition_present_at_admission"] = "uncertain"

admissions["cardiac_condition_verified"] = "uncertain"

admissions["ami_status"] = "uncertain"

admissions["ami_onset_time"] = ""

admissions["evidence_reference"] = ""

admissions["review_notes"] = ""


# ==================================================
# STEP 5: SAVE REVIEW FILE
# ==================================================

print("\n[5/5] Preparing final review CSV...")

output_columns = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",

    "ami_diagnosis_code_found",
    "ami_code_count",
    "ami_codes",

    "troponin_measurement_count",
    "troponin_numeric_count",
    "troponin_test_names",
    "first_troponin_time",
    "last_troponin_time",

    "cardiac_condition_present_at_admission",
    "cardiac_condition_verified",

    "ami_status",
    "ami_onset_time",

    "evidence_reference",
    "review_notes"
]

review = admissions[output_columns].copy()

# Place AMI-coded admissions first to help prioritize review.
review = review.sort_values(
    by=["ami_diagnosis_code_found", "subject_id"],
    ascending=[False, True]
)

# Protect existing manual review work.
if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"\nReview file already exists:\n{OUTPUT_FILE}\n"
        "It has NOT been overwritten. Back it up or rename it "
        "before generating another review file."
    )

review.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n========== SCRIPT 03 SUMMARY ==========")

print("Total candidate admissions:", len(review))

print(
    "Distinct patients:",
    review["subject_id"].nunique()
)

print(
    "Admissions with AMI diagnosis codes:",
    review["ami_diagnosis_code_found"].sum()
)

print(
    "Admissions without AMI diagnosis codes:",
    (~review["ami_diagnosis_code_found"]).sum()
)

print(
    "Admissions with troponin measurements:",
    (review["troponin_measurement_count"] > 0).sum()
)

print(
    "Admissions without troponin measurements:",
    (review["troponin_measurement_count"] == 0).sum()
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n======================================")
