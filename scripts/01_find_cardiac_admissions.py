
import sys
from pathlib import Path
import pandas as pd

# Locate the project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python's import path
sys.path.append(str(PROJECT_ROOT))

# Import paths and settings from config.py
from config import HOSP, CANDIDATES, CHUNK_SIZE

# MIMIC diagnosis file
diagnosis_file = HOSP / "diagnoses_icd.csv.gz"

diagnosis_chunks = []

for chunk in pd.read_csv(
    diagnosis_file,
    usecols=[
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd_code",
        "icd_version"
    ],
    chunksize=CHUNK_SIZE
):
    codes = (
        chunk["icd_code"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    versions = pd.to_numeric(
        chunk["icd_version"],
        errors="coerce"
    )

    # ICD-10 cardiac diagnosis families.
    icd10_cardiac = (
        versions.eq(10) &
        (
            codes.str.match(r"^I0[5-9]") |
            codes.str.match(r"^I2[0-5]") |
            codes.str.match(r"^I[3-5][0-9]")
        )
    )

    # ICD-9 cardiac diagnosis families.
    icd9_prefix = pd.to_numeric(
        codes.str.extract(r"^(\d{3})")[0],
        errors="coerce"
    )

    icd9_cardiac = (
        versions.eq(9) &
        icd9_prefix.between(393, 398) |
        (
            versions.eq(9) &
            icd9_prefix.between(410, 414)
        ) |
        (
            versions.eq(9) &
            icd9_prefix.between(420, 429)
        )
    )

    matched = chunk[
        icd10_cardiac | icd9_cardiac
    ].copy()

    if not matched.empty:
        diagnosis_chunks.append(matched)

if not diagnosis_chunks:
    raise RuntimeError("No cardiac diagnosis records found.")

cardiac_diagnoses = pd.concat(
    diagnosis_chunks,
    ignore_index=True
).drop_duplicates()

# Attach diagnosis descriptions.
dictionary = pd.read_csv(
    HOSP / "d_icd_diagnoses.csv.gz",
    dtype={"icd_code": str}
)

cardiac_diagnoses["icd_code"] = (
    cardiac_diagnoses["icd_code"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.strip()
)

dictionary["icd_code"] = (
    dictionary["icd_code"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.strip()
)

cardiac_diagnoses = cardiac_diagnoses.merge(
    dictionary,
    on=["icd_code", "icd_version"],
    how="left"
)

# Attach hospital admission timestamps.
admissions = pd.read_csv(
    HOSP / "admissions.csv.gz",
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type"
    ]
)

cardiac_admissions = cardiac_diagnoses.merge(
    admissions,
    on=["subject_id", "hadm_id"],
    how="inner"
)

cardiac_admissions = cardiac_admissions.drop_duplicates(
    subset=["subject_id", "hadm_id", "icd_code", "icd_version"]
)

cardiac_admissions.to_csv(
    CANDIDATES / "cardiac_admission_diagnoses.csv",
    index=False
)

# One row per patient-admission.
admission_summary = (
    cardiac_admissions
    .groupby(["subject_id", "hadm_id"], as_index=False)
    .agg(
        admittime=("admittime", "first"),
        dischtime=("dischtime", "first"),
        diagnosis_count=("icd_code", "nunique")
    )
)

admission_summary.to_csv(
    CANDIDATES / "cardiac_admissions.csv",
    index=False
)

print("Cardiac diagnosis records:",
      len(cardiac_admissions))

print("Unique cardiac admissions:",
      admission_summary["hadm_id"].nunique())

print("Unique cardiac patients:",
      admission_summary["subject_id"].nunique())
