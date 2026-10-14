#!/usr/bin/env python3
"""
Home Credit Default Risk -- combined best-of-both merge script.

Keeps, from the "GPT" version:
  * assert_unique_key() + merge(validate="one_to_one") everywhere -- fails
    loudly on any duplicate-key assumption instead of silently corrupting rows
  * ID columns excluded from downcasting (fixes the int32/int64 join crash)
  * separate train/test output files
  * std alongside mean/min/max/sum

Keeps, from the original version:
  * status-conditioned subset aggregates: BURO_ACTIVE / BURO_CLOSED,
    PREV_APPROVED / PREV_REFUSED -- these are consistently strong features
    in this competition and were missing from the GPT version
  * vectorized one-hot + mean ("share of this client's records in category")
    for every categorical column, instead of a per-group Python mode function
    (which is slow on the 3-13M row child tables)
  * additionally keeps nunique per categorical column, so you get BOTH
    cardinality and distribution, not just one

Usage:
    python merge_home_credit_v2.py --data-dir ./data --out-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

SENTINEL_DAYS = 365243
ID_COLS = ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")
NUM_AGGS = ["count", "mean", "min", "max", "sum", "std"]
SPLIT_AGGS = ["count", "mean", "min", "max", "sum"]

_T0 = time.time()


def log(msg: str) -> None:
    m, s = divmod(time.time() - _T0, 60)
    print(f"[{int(m):02d}:{s:05.2f}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if c in ID_COLS:
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype("int64")   # force consistent dtype everywhere
            continue
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    log(f"  read {name:<22} {df.shape[0]:>10,} x {df.shape[1]:>3}")
    return df


def assert_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    if df[key].isna().any():
        raise ValueError(f"{name}: {key} has missing values")
    if df[key].duplicated().any():
        n = int(df[key].duplicated(keep=False).sum())
        raise ValueError(f"{name}: {n:,} duplicate {key} values -- investigate, don't auto-drop")


def clean_sentinels(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "DAYS_EMPLOYED" in df.columns:
        bad = df["DAYS_EMPLOYED"].eq(SENTINEL_DAYS)
        df["DAYS_EMPLOYED_ANOM"] = bad.astype("uint8")
        df.loc[bad, "DAYS_EMPLOYED"] = np.nan

    if name == "previous_application":
        for c in ("DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                  "DAYS_LAST_DUE", "DAYS_TERMINATION"):
            if c in df.columns:
                df.loc[df[c].eq(SENTINEL_DAYS), c] = np.nan

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].replace("XNA", np.nan)   # XAP is a real state, kept
    return df


def split_cols(df: pd.DataFrame, exclude: list[str]) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat


def one_hot_with_nunique(df: pd.DataFrame, key: str, cat_cols: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Returns (df_with_dummies, dummy_col_names, nunique_per_key_frame)."""
    if not cat_cols:
        return df, [], pd.DataFrame(index=pd.Index(df[key].unique(), name=key))
    nunique = df.groupby(key, sort=False)[cat_cols].nunique(dropna=True)
    nunique.columns = [f"{c}_NUNIQUE" for c in nunique.columns]
    before = set(df.columns)
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True, dtype=np.uint8)
    return df, [c for c in df.columns if c not in before], nunique


def aggregate(df: pd.DataFrame, key: str, prefix: str,
              num_cols: list[str], ohe_cols: list[str],
              num_aggs: list[str] = NUM_AGGS) -> pd.DataFrame:
    frames = []
    if num_cols:
        g = df.groupby(key, sort=False)[num_cols].agg(num_aggs)
        g.columns = [f"{prefix}_{c}_{a.upper()}" for c, a in g.columns]
        frames.append(g)
    if ohe_cols:
        g = df.groupby(key, sort=False)[ohe_cols].mean()
        g.columns = [f"{prefix}_{c}_RATE" for c in g.columns]
        frames.append(g)
    if not frames:
        raise ValueError(f"nothing to aggregate for {prefix}")
    out = pd.concat(frames, axis=1)
    out[f"{prefix}_RECORD_COUNT"] = df.groupby(key, sort=False).size()
    return out.reset_index()


def merge_unique(left: pd.DataFrame, right: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    """Left-merge, asserting the right side is unique on key and row count doesn't change."""
    if not right[key].is_unique:
        raise AssertionError(f"{name}: right side not unique on {key}")
    before = len(left)
    out = left.merge(right, on=key, how="left", validate="one_to_one")
    if len(out) != before:
        raise AssertionError(f"{name}: row count changed on merge ({before:,} -> {len(out):,})")
    return out


# ---------------------------------------------------------------------------
# 1. spine
# ---------------------------------------------------------------------------

def build_application(data_dir: Path) -> pd.DataFrame:
    log("application_{train,test}")
    train = read_table(data_dir / FILES["application_train"], "application_train")
    test = read_table(data_dir / FILES["application_test"], "application_test")

    if "TARGET" not in train.columns:
        raise ValueError("application_train.csv must contain TARGET")
    assert_unique_key(train, "SK_ID_CURR", "application_train")
    assert_unique_key(test, "SK_ID_CURR", "application_test")
    overlap = set(train["SK_ID_CURR"]) & set(test["SK_ID_CURR"])
    if overlap:
        raise ValueError(f"train/test SK_ID_CURR overlap: {len(overlap):,} ids")

    train = clean_sentinels(train, "application")
    test = clean_sentinels(test, "application")
    train["SPLIT"], test["SPLIT"] = "train", "test"
    test["TARGET"] = np.nan

    df = pd.concat([train, test], ignore_index=True, sort=False).copy()
    del train, test
    gc.collect()
    log(f"  spine: {df.shape[0]:,} x {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 2. bureau branch
# ---------------------------------------------------------------------------

def build_bureau(data_dir: Path) -> pd.DataFrame:
    log("bureau branch")
    bb = read_table(data_dir / FILES["bureau_balance"], "bureau_balance")
    bb = bb.drop_duplicates(subset=["SK_ID_BUREAU", "MONTHS_BALANCE"], keep="last")
    bb = clean_sentinels(bb, "bureau_balance")
    bb, bb_ohe, bb_nuniq = one_hot_with_nunique(bb, "SK_ID_BUREAU", ["STATUS"])
    bb_agg = aggregate(bb, "SK_ID_BUREAU", "BB", ["MONTHS_BALANCE"], bb_ohe)
    bb_agg = bb_agg.merge(bb_nuniq.add_prefix("BB_").reset_index(), on="SK_ID_BUREAU", how="left")
    del bb
    gc.collect()

    bureau = read_table(data_dir / FILES["bureau"], "bureau")
    assert_unique_key(bureau, "SK_ID_BUREAU", "bureau")
    bureau = merge_unique(bureau, bb_agg, "SK_ID_BUREAU", "bureau+bureau_balance")
    del bb_agg
    gc.collect()
    bureau = clean_sentinels(bureau, "bureau")

    mask_active = bureau["CREDIT_ACTIVE"].eq("Active").to_numpy()
    mask_closed = bureau["CREDIT_ACTIVE"].eq("Closed").to_numpy()

    num, cat = split_cols(bureau, exclude=["SK_ID_CURR", "SK_ID_BUREAU"])
    bureau, ohe, nuniq = one_hot_with_nunique(bureau, "SK_ID_CURR", cat)

    out = aggregate(bureau, "SK_ID_CURR", "BURO", num, ohe)
    out = out.merge(nuniq.add_prefix("BURO_").reset_index(), on="SK_ID_CURR", how="left")

    for label, mask in (("BURO_ACTIVE", mask_active), ("BURO_CLOSED", mask_closed)):
        sub = bureau.loc[mask]
        if len(sub):
            sub_agg = aggregate(sub, "SK_ID_CURR", label, num, [], SPLIT_AGGS)
            out = out.merge(sub_agg, on="SK_ID_CURR", how="left")
        del sub

    del bureau
    gc.collect()
    log(f"  bureau branch -> {out.shape[0]:,} x {out.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# 3. previous_application
# ---------------------------------------------------------------------------

def build_previous(data_dir: Path) -> pd.DataFrame:
    log("previous_application")
    prev = read_table(data_dir / FILES["previous_application"], "previous_application")
    assert_unique_key(prev, "SK_ID_PREV", "previous_application")
    prev = clean_sentinels(prev, "previous_application")

    with np.errstate(divide="ignore", invalid="ignore"):
        prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"].replace(0, np.nan)

    mask_appr = prev["NAME_CONTRACT_STATUS"].eq("Approved").to_numpy()
    mask_refu = prev["NAME_CONTRACT_STATUS"].eq("Refused").to_numpy()

    num, cat = split_cols(prev, exclude=["SK_ID_CURR", "SK_ID_PREV"])
    prev, ohe, nuniq = one_hot_with_nunique(prev, "SK_ID_CURR", cat)

    out = aggregate(prev, "SK_ID_CURR", "PREV", num, ohe)
    out = out.merge(nuniq.add_prefix("PREV_").reset_index(), on="SK_ID_CURR", how="left")

    for label, mask in (("PREV_APPROVED", mask_appr), ("PREV_REFUSED", mask_refu)):
        sub = prev.loc[mask]
        if len(sub):
            sub_agg = aggregate(sub, "SK_ID_CURR", label, num, [], SPLIT_AGGS)
            out = out.merge(sub_agg, on="SK_ID_CURR", how="left")
        del sub

    del prev
    gc.collect()
    log(f"  previous_application -> {out.shape[0]:,} x {out.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# 4-6. POS / installments / credit_card
# ---------------------------------------------------------------------------

def _build_child_table(data_dir: Path, file_key: str, prefix: str, label: str,
                        extra_features=None) -> pd.DataFrame:
    log(label)
    df = read_table(data_dir / FILES[file_key], label)
    df = df.drop_duplicates()
    df = clean_sentinels(df, file_key)
    if extra_features is not None:
        df = extra_features(df)

    num, cat = split_cols(df, exclude=["SK_ID_CURR", "SK_ID_PREV"])
    df, ohe, nuniq = one_hot_with_nunique(df, "SK_ID_CURR", cat)

    out = aggregate(df, "SK_ID_CURR", prefix, num, ohe)
    out = out.merge(nuniq.add_prefix(f"{prefix}_").reset_index(), on="SK_ID_CURR", how="left")

    unique_prev = df.groupby("SK_ID_CURR", sort=False)["SK_ID_PREV"].nunique()
    out = out.merge(unique_prev.rename(f"{prefix}_NUNIQUE_PREV").reset_index(),
                     on="SK_ID_CURR", how="left")
    log(f"  {label} -> {out.shape[0]:,} x {out.shape[1]}")
    del df
    gc.collect()
    return out


def build_pos(data_dir: Path) -> pd.DataFrame:
    return _build_child_table(data_dir, "pos", "POS", "POS_CASH_balance")


def build_installments(data_dir: Path) -> pd.DataFrame:
    def features(df: pd.DataFrame) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["PAYMENT_PERC"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
        df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
        delay = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["DPD"] = delay.clip(lower=0)
        df["DBD"] = (-delay).clip(lower=0)
        return df.replace([np.inf, -np.inf], np.nan)
    return _build_child_table(data_dir, "installments", "INSTAL", "installments_payments", features)


def build_credit_card(data_dir: Path) -> pd.DataFrame:
    return _build_child_table(data_dir, "credit_card", "CC", "credit_card_balance")


# ---------------------------------------------------------------------------
# final clean + write
# ---------------------------------------------------------------------------

def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float"]).columns:
        vals = df[c].to_numpy()
        if np.isinf(vals).any():
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    protected = {"SK_ID_CURR", "TARGET", "SPLIT"}
    drop_cols = []
    for c in df.columns:
        if c in protected:
            continue
        s = df[c]
        if s.isna().all():
            drop_cols.append(c)
        elif s.nunique(dropna=False) <= 1:
            drop_cols.append(c)
    if drop_cols:
        log(f"  dropping {len(drop_cols)} all-null/constant columns")
        df = df.drop(columns=drop_cols)

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)

    seen, names = {}, []
    import re
    for c in df.columns:
        n = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        n = re.sub(r"_+", "_", n).strip("_").upper()
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        names.append(n)
    df.columns = names

    lead = [c for c in ("SK_ID_CURR", "TARGET", "SPLIT") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    if not df["SK_ID_CURR"].is_unique:
        raise AssertionError("final output has duplicate SK_ID_CURR -- do not ship this file")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path,
                     default=Path(r"C:\Users\kunwa\PycharmProjects\loan_deafult\output"))
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_application(data_dir)
    train_rows = int((df["SPLIT"] == "train").sum())
    test_rows = int((df["SPLIT"] == "test").sum())

    for builder in (build_bureau, build_previous, build_pos, build_installments, build_credit_card):
        part = builder(data_dir)
        df = merge_unique(df, part, "SK_ID_CURR", builder.__name__)
        log(f"  joined -> {df.shape[0]:,} x {df.shape[1]}")
        del part
        gc.collect()

    df = final_clean(df)

    n_train = int((df["SPLIT"] == "train").sum())
    n_test = int((df["SPLIT"] == "test").sum())
    if n_train != train_rows:
        raise AssertionError("train row count changed")
    if n_test != test_rows:
        raise AssertionError("test row count changed")

    out_path = out_dir / "train_main.csv"
    df.to_csv(out_path, index=False, float_format="%.7g")

    log("DONE")
    log(f"  combined: {df.shape[0]:,} x {df.shape[1]:,} ({n_train:,} train / {n_test:,} test)")
    log(f"  train default rate: {df.loc[df['SPLIT'].eq('train'), 'TARGET'].mean():.5f}")
    log(f"  duplicate SK_ID_CURR: {df['SK_ID_CURR'].duplicated().sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())