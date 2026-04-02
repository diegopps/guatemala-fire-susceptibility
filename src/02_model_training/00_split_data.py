"""
split_data.py
-------------
Splits the input dataset into train, test, and validation subsets using
stratified sampling to preserve the class ratio of the 'fire' column.

Outputs a single index CSV file indicating which subset each row belongs to.
Downstream scripts read index to determine subset.
"""

# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_DIR = "/home/diegopab/projects/ForestFire_GT/paperwriting/ForestFireGT"
INPUT_DATA_PATH = f"{PROJECT_DIR}/data/indata/data_trainingset_2017-2024/allvars_2017-2024.csv"
OUTPUT_INDEX_PATH = f"{PROJECT_DIR}/data/processed/split_index.csv"

RANDOM_SEED = 107

# Split proportions
TRAIN_SIZE = 0.65
TEST_SIZE = 0.25
VAL_SIZE = 0.10

# Column used for stratification
STRATIFY_COLUMN = "fire"

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Validate split proportions
    assert abs(TRAIN_SIZE + TEST_SIZE + VAL_SIZE - 1.0) < 1e-9, \
        "TRAIN_SIZE, TEST_SIZE, and VAL_SIZE must sum to 1.0"

    print(f"Loading data from: {INPUT_DATA_PATH}")
    df = pd.read_csv(INPUT_DATA_PATH)
    print(f"  Total rows: {len(df):,}")

    # Print class distribution
    class_counts = df[STRATIFY_COLUMN].value_counts()
    print(f"\nClass distribution in '{STRATIFY_COLUMN}':")
    for label, count in class_counts.items():
        print(f"  {label}: {count:,} ({count / len(df) * 100:.4f}%)")

    # --- Step 1: Split off validation set ---
    # We first carve out the validation set from the full dataset,
    # then split the remainder into train and test.
    val_relative_size = VAL_SIZE  # relative to full dataset
    train_test_relative_size = 1.0 - VAL_SIZE

    df_traintest, df_val = train_test_split(
        df,
        test_size=val_relative_size,
        random_state=RANDOM_SEED,
        stratify=df[STRATIFY_COLUMN]
    )

    # --- Step 2: Split train/test from the remaining data ---
    # test_size relative to the train+test subset
    test_relative_size = TEST_SIZE / train_test_relative_size

    df_train, df_test = train_test_split(
        df_traintest,
        test_size=test_relative_size,
        random_state=RANDOM_SEED,
        stratify=df_traintest[STRATIFY_COLUMN]
    )

    # --- Step 3: Build index file ---
    split_index = pd.Series("", index=df.index, name="split", dtype=str)
    split_index.loc[df_train.index] = "train"
    split_index.loc[df_test.index] = "test"
    split_index.loc[df_val.index] = "validation"

    split_index_df = split_index.reset_index()
    split_index_df.columns = ["row_index", "split"]

    # --- Step 4: Save index file ---
    split_index_df.to_csv(OUTPUT_INDEX_PATH, index=False)
    print(f"\nSplit index saved to: {OUTPUT_INDEX_PATH}")

    # --- Step 5: Summary report ---
    print("\nSubset sizes:")
    for subset in ["train", "test", "validation"]:
        subset_df = df.loc[split_index_df[split_index_df["split"] == subset]["row_index"]]
        n = len(subset_df)
        pos = subset_df[STRATIFY_COLUMN].sum()
        neg = n - pos
        print(f"  {subset:<12}: {n:>10,} rows | "
              f"positives: {int(pos):>7,} ({pos/n*100:.4f}%) | "
              f"negatives: {int(neg):>10,} ({neg/n*100:.4f}%)")


if __name__ == "__main__":
    main()
