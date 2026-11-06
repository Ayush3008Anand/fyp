
import pandas as pd
from pathlib import Path

# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIMIC = Path(r"E:\downloads\mimic-iv-3.1\mimic-iv-3.1")
ICU = MIMIC / "icu"

FINAL = PROJECT_ROOT / "data" / "final"
FINAL.mkdir(parents=True, exist_ok=True)

COHORT_FILE = FINAL / "selected_90_patients.csv"
CHARTEVENTS_FILE = ICU / "chartevents.csv.gz"
ITEMS_FILE = ICU / "d_items.csv.gz"

OUTPUT_FILE = FINAL / "cardiac_patient_labelled_vitals.csv"
TEMP_FILE = FINAL / "cardiac_patient_labelled_vitals.tmp.csv"

CHUNK_SIZE = 500_000

# MIMIC-IV vital-sign item IDs
VITAL_ITEM_IDS = {
    220045,  # Heart rate
    220050,  # Systolic BP
    220051,  # Diastolic BP
    220052,  # Mean BP
    220179,  # Systolic BP
    220180,  # Diastolic BP
    220181,  # Mean BP
    220277   # Oxygen saturation (SpO2)
}


# =====================================================
# 2. MAIN FUNCTION
# =====================================================

def main():

    print("=" * 60)
    print("EXTRACTING LABELLED VITAL SIGNS")
    print("=" * 60)

    # Check required files
    for file in [COHORT_FILE, CHARTEVENTS_FILE, ITEMS_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file}")

    # Remove incomplete temporary output from an earlier run
    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    # -------------------------------------------------
    # Load selected cohort
    # -------------------------------------------------

    print("\nLoading selected cohort...")

    cohort = pd.read_csv(COHORT_FILE, low_memory=False)

    required_columns = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "ami_during_admission"
    ]

    missing = [c for c in required_columns if c not in cohort.columns]

    if missing:
        raise ValueError(f"Missing columns in cohort file: {missing}")

    cohort["subject_id"] = pd.to_numeric(
        cohort["subject_id"], errors="coerce"
    )

    cohort["hadm_id"] = pd.to_numeric(
        cohort["hadm_id"], errors="coerce"
    )

    cohort["admittime"] = pd.to_datetime(
        cohort["admittime"], errors="coerce"
    )

    cohort["dischtime"] = pd.to_datetime(
        cohort["dischtime"], errors="coerce"
    )

    cohort["ami_during_admission"] = pd.to_numeric(
        cohort["ami_during_admission"], errors="coerce"
    )

    cohort = cohort.dropna(subset=required_columns).copy()

    cohort["subject_id"] = cohort["subject_id"].astype("int64")
    cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
    cohort["ami_during_admission"] = (
        cohort["ami_during_admission"].astype("int64")
    )

    # Validate cohort
    if len(cohort) != 90:
        raise ValueError(f"Expected 90 records, found {len(cohort)}")

    if cohort["subject_id"].nunique() != 90:
        raise ValueError("Expected 90 distinct patients.")

    if cohort["hadm_id"].nunique() != 90:
        raise ValueError("Expected 90 distinct admissions.")

    label_counts = cohort["ami_during_admission"].value_counts()

    if label_counts.get(1, 0) != 45 or label_counts.get(0, 0) != 45:
        raise ValueError("Expected exactly 45 positive and 45 control records.")

    print("Total patients:", cohort["subject_id"].nunique())
    print("AMI-code-positive:", label_counts.get(1, 0))
    print("Control:", label_counts.get(0, 0))

    admission_info = cohort[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "ami_during_admission"
        ]
    ].drop_duplicates(["subject_id", "hadm_id"])

    selected_subjects = set(cohort["subject_id"].unique())

    # -------------------------------------------------
    # Load vital-sign dictionary
    # -------------------------------------------------

    print("\nLoading vital-sign dictionary...")

    items = pd.read_csv(
        ITEMS_FILE,
        usecols=["itemid", "label"],
        low_memory=False
    )

    items["itemid"] = pd.to_numeric(items["itemid"], errors="coerce")

    items = items[
        items["itemid"].isin(VITAL_ITEM_IDS)
    ].drop_duplicates("itemid")

    print(items.to_string(index=False))

    # -------------------------------------------------
    # Prepare extraction
    # -------------------------------------------------

    input_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "valuenum",
        "valueuom"
    ]

    output_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "charttime",
        "itemid",
        "label",
        "valuenum",
        "valueuom",
        "admittime",
        "dischtime",
        "ami_during_admission"
    ]

    total_scanned = 0
    total_measurements = 0
    chunk_number = 0

    matched_patients = set()
    matched_admissions = set()

    print("\nStarting chunk-based extraction...")
    print("Chunk size:", f"{CHUNK_SIZE:,}")
    print("This process scans the complete chartevents file.")

    # -------------------------------------------------
    # Read chartevents in chunks
    # -------------------------------------------------

    for chunk in pd.read_csv(
        CHARTEVENTS_FILE,
        usecols=input_columns,
        compression="gzip",
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk_number += 1
        total_scanned += len(chunk)

        # Filter selected patients and vital-sign item IDs
        chunk = chunk[
            chunk["subject_id"].isin(selected_subjects)
            & chunk["itemid"].isin(VITAL_ITEM_IDS)
        ]

        if chunk.empty:
            print(
                f"Chunk {chunk_number} | "
                f"Scanned: {total_scanned:,} | "
                f"Extracted: {total_measurements:,}"
            )
            continue

        # Match exact patient and admission
        chunk = chunk.merge(
            admission_info,
            on=["subject_id", "hadm_id"],
            how="inner"
        )

        if chunk.empty:
            continue

        # Convert values and timestamps
        chunk["charttime"] = pd.to_datetime(
            chunk["charttime"], errors="coerce"
        )

        chunk["valuenum"] = pd.to_numeric(
            chunk["valuenum"], errors="coerce"
        )

        # Retain measurements during the selected admission
        chunk = chunk[
            (chunk["charttime"] >= chunk["admittime"])
            & (chunk["charttime"] <= chunk["dischtime"])
        ]

        chunk = chunk.dropna(subset=["charttime", "valuenum"])

        if chunk.empty:
            continue

        # Attach readable vital-sign names
        chunk = chunk.merge(
            items,
            on="itemid",
            how="left"
        )

        chunk = chunk[output_columns]

        # Append to temporary CSV
        chunk.to_csv(
            TEMP_FILE,
            mode="a",
            header=not TEMP_FILE.exists(),
            index=False
        )

        total_measurements += len(chunk)

        matched_patients.update(chunk["subject_id"].unique())

        matched_admissions.update(
            zip(chunk["subject_id"], chunk["hadm_id"])
        )

        print(
            f"Chunk {chunk_number} | "
            f"Scanned: {total_scanned:,} | "
            f"Extracted: {total_measurements:,} | "
            f"Patients matched: {len(matched_patients)}"
        )

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    print("Total chartevents rows scanned:", f"{total_scanned:,}")
    print("Total measurements extracted:", f"{total_measurements:,}")
    print("Patients with measurements:", len(matched_patients))
    print(
        "Patients without measurements:",
        len(selected_subjects - matched_patients)
    )
    print("Admissions with measurements:", len(matched_admissions))

    if total_measurements > 0:

        # Replace output only after successful extraction
        TEMP_FILE.replace(OUTPUT_FILE)

        print("\nDataset created successfully!")
        print("Output:", OUTPUT_FILE)

    else:
        print("\nNo matching measurements found.")

        if TEMP_FILE.exists():
            TEMP_FILE.unlink()


if __name__ == "__main__":
    main()
