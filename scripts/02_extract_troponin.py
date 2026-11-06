
import sys
import time
from pathlib import Path

import pandas as pd

# Import project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import HOSP, CANDIDATES, TROPONIN, CHUNK_SIZE


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

LABEVENTS_FILE = HOSP / "labevents.csv.gz"
LABITEMS_FILE = HOSP / "d_labitems.csv.gz"

CANDIDATE_FILE = CANDIDATES / "cardiac_admissions.csv"

OUTPUT_FILE = TROPONIN / "cardiac_patient_troponin.csv"
AVAILABLE_TESTS_FILE = TROPONIN / "available_troponin_tests.csv"


# --------------------------------------------------
# STEP 1: IDENTIFY TROPONIN ITEM IDS
# --------------------------------------------------

print("\n[1/4] Identifying troponin test IDs...")

lab_items = pd.read_csv(
    LABITEMS_FILE,
    usecols=["itemid", "label"]
)

troponin_items = lab_items[
    lab_items["label"].str.contains(
        "troponin", case=False, na=False
    )
].copy()

troponin_item_ids = set(
    troponin_items["itemid"].astype(int)
)

troponin_items.to_csv(
    AVAILABLE_TESTS_FILE,
    index=False
)

print("Troponin test IDs found:", len(troponin_item_ids))


# --------------------------------------------------
# STEP 2: LOAD CANDIDATE ADMISSIONS
# --------------------------------------------------

print("\n[2/4] Loading candidate admissions...")

admissions = pd.read_csv(
    CANDIDATE_FILE,
    usecols=["subject_id", "hadm_id"],
    dtype={
        "subject_id": "int32",
        "hadm_id": "int32"
    }
).drop_duplicates()

valid_hadm_ids = set(admissions["hadm_id"])

print("Candidate admissions:", len(valid_hadm_ids))
print("Candidate patients:", admissions["subject_id"].nunique())


# --------------------------------------------------
# STEP 3: SCAN LABEVENTS IN CHUNKS
# --------------------------------------------------

print("\n[3/4] Scanning labevents.csv.gz...")

use_columns = [
    "subject_id",
    "hadm_id",
    "itemid",
    "charttime",
    "valuenum",
    "value",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag"
]

total_rows = 0
total_chunks = 0
matched_rows = 0

unique_patients = set()

start_time = time.time()

# Remove previous output to avoid appending duplicate records
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

first_write = True

reader = pd.read_csv(
    LABEVENTS_FILE,
    compression="gzip",
    usecols=use_columns,
    chunksize=CHUNK_SIZE,
    dtype={
        "subject_id": "Int64",
        "hadm_id": "Int64",
        "itemid": "int32"
    }
)

for chunk in reader:

    total_chunks += 1
    total_rows += len(chunk)

    # Filter candidate admissions
    chunk = chunk[
        chunk["hadm_id"].isin(valid_hadm_ids)
    ]

    # Filter troponin tests
    chunk = chunk[
        chunk["itemid"].isin(troponin_item_ids)
    ]

    if not chunk.empty:

        # Verify subject_id and hadm_id combinations
        chunk = chunk.merge(
            admissions,
            on=["subject_id", "hadm_id"],
            how="inner"
        )

        # Add readable laboratory test labels
        label_mapping = dict(
            zip(troponin_items["itemid"],
                troponin_items["label"])
        )

        chunk["test_label"] = chunk["itemid"].map(
            label_mapping
        )

        matched_rows += len(chunk)

        unique_patients.update(
            chunk["subject_id"].dropna().unique()
        )

        # Incrementally save records
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

    # Live progress
    elapsed = time.time() - start_time

    print(
        f"Chunk: {total_chunks} | "
        f"Rows scanned: {total_rows:,} | "
        f"Troponin records: {matched_rows:,} | "
        f"Unique patients: {len(unique_patients):,} | "
        f"Time: {elapsed / 60:.2f} min",
        flush=True
    )


# --------------------------------------------------
# STEP 4: FINAL SUMMARY
# --------------------------------------------------

print("\n[4/4] Scanning completed!")

total_time = time.time() - start_time

print("\n========== FINAL SUMMARY ==========")

print("Total chunks processed:", total_chunks)
print("Total rows scanned:", f"{total_rows:,}")
print("Matching troponin records:", f"{matched_rows:,}")
print("Unique patients with troponin:", len(unique_patients))

print(f"Total execution time: {total_time / 60:.2f} minutes")

print("\nOutput files:")
print(AVAILABLE_TESTS_FILE)
print(OUTPUT_FILE)

print("\n===================================")
