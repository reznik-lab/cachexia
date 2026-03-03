#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm


PROJECT_ROOT = os.environ.get("CACHEXIA_PROJECT_ROOT", os.getcwd())
REV_INPUTS = os.environ.get("CACHEXIA_INPUTS_DIR", os.path.join(PROJECT_ROOT, "data"))
REV_RESULTS = os.environ.get("CACHEXIA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
REV_PLOTS = os.environ.get("CACHEXIA_PLOTS_DIR", os.path.join(PROJECT_ROOT, "plots"))

DATE_STAMP = os.environ.get("CACHEXIA_DATE_STAMP", "YYYYMMDD")

dx_fp = os.environ.get("CACHEXIA_DX_FP", os.path.join(REV_INPUTS, f"dx_cohort_metadata_{DATE_STAMP}.csv"))
bmi_fp = os.environ.get("CACHEXIA_BMI_FP", os.path.join(REV_INPUTS, f"bmi_final_{DATE_STAMP}.csv"))

# Episode detection settings (match your current code defaults)
ALPHA = float(os.environ.get("CACHEXIA_EWMA_ALPHA", "0.2"))
SMOOTH_COL = os.environ.get("CACHEXIA_SMOOTH_COL", "bmi")
TIME_COL = os.environ.get("CACHEXIA_TIME_COL", "days_since_anchor")
BMI_LOG_COL = os.environ.get("CACHEXIA_BMI_LOG_COL", "log_smoothed_BMI")

WL_RUNS = {"WL5": 0.05, "WL10": 0.10, "WL15": 0.15}

# QC settings
MIN_DUR = int(os.environ.get("CACHEXIA_MIN_DUR", "15"))
MIN_WL_PCT = float(os.environ.get("CACHEXIA_MIN_WL_PCT", "2"))

# Edema QC settings
EDEMA_WINDOW_DAYS = int(os.environ.get("CACHEXIA_EDEMA_WINDOW_DAYS", "30"))
UP_FRAC = float(os.environ.get("CACHEXIA_EDEMA_UP_FRAC", "0.05"))
RETURN_FRAC = float(os.environ.get("CACHEXIA_EDEMA_RETURN_FRAC", "0.02"))

# Which episode detector to use:
# - "standard": identify_cachexia_episodes (fixed wl_frac)
# - "bmi20_rule": identify_cachexia_episodes_2 (2% WL if baseline BMI<20 when enabled)
EPISODE_MODE = os.environ.get("CACHEXIA_EPISODE_MODE", "standard").strip().lower()
BMI20_RULE = os.environ.get("CACHEXIA_BMI20_RULE", "0").strip() in ("1", "true", "yes", "y")

os.makedirs(REV_RESULTS, exist_ok=True)
os.makedirs(REV_PLOTS, exist_ok=True)



# ------------- Smoothing ----------------------
def smooth_bmi_ewma(df, smooth_col, alpha):
    df["smoothed_BMI"] = df[smooth_col].ewm(alpha=alpha, ignore_na=True).mean()
    return df

# ------------- Recovery Episode Identification ----------------------
def identify_recovery_episodes(patient_data, merged_episodes_df, time_col, bmi_col):
   
    merged_episodes_df["recovery_day"] = None
    merged_episodes_df["recovery_date"] = None
    merged_episodes_df["recovery_bmi"] = None

    merged_episodes_df["recovery_smoothed_bmi"] = None
    if not merged_episodes_df["start_day"].isna().all():
        for i in range(merged_episodes_df.shape[0]):
            t0 = merged_episodes_df["end_day"][i]
            if i < merged_episodes_df.shape[0] - 1:
                t1 = merged_episodes_df["start_day"][i + 1]
            else:
                t1 = patient_data[time_col].max()

            end_index = patient_data.loc[patient_data[time_col] == t0].index[0]
            b0 = patient_data[bmi_col][end_index]

            recovery_data = patient_data[
                (patient_data[time_col] >= t0)
                & (patient_data[time_col] <= t1)
                & (patient_data[bmi_col] > b0 + np.log(1.05))
            ]

            if not recovery_data.empty:
                recovery_day = recovery_data[time_col].min()
                recovery_date = recovery_data.loc[recovery_data[time_col] == recovery_day, "datetime"].values[0]
                recovery_bmi = recovery_data.loc[recovery_data[time_col] == recovery_day, bmi_col].values[0]
                recovery_smoothed_bmi = recovery_data.loc[
                    recovery_data[time_col] == recovery_day, "smoothed_BMI"
                ].values[0]

                merged_episodes_df.at[i, "recovery_day"] = recovery_day
                merged_episodes_df.at[i, "recovery_date"] = recovery_date
                merged_episodes_df.at[i, "recovery_bmi"] = recovery_bmi
                merged_episodes_df.at[i, "recovery_smoothed_bmi"] = recovery_smoothed_bmi 

    return merged_episodes_df


# ------------- Cachexia Episode Identification 5% , 2% if BMI<20----------------------
def identify_cachexia_episodes_2(df, time_col, bmi_col, recovery=True, wl_frac=0.05, bmi20_rule=False):
   
    results = []
    df.reset_index(drop=True, inplace=True)

    wl_log_default = np.log(1 - wl_frac)  # e.g., log(0.95)
    wl_log_lowbmi = np.log(0.98)  # 2% WL threshold for BMI<20


    for i in range(df.shape[0]):
        t0 = df[time_col][i]
        b0 = df[bmi_col][i]  # baseline log-smoothed BMI
        baseline_bmi = df["smoothed_BMI"][i]  # baseline BMI in linear space (smoothed)

        current_date = df["datetime"][i]
        current_bmi = df["bmi"][i]

        df_window = df[(df[time_col] > t0) & (df[time_col] <= t0 + 180)]

        wl_log_i = wl_log_lowbmi if (bmi20_rule and baseline_bmi < 20) else wl_log_default
        df_filtered = df_window[df_window[bmi_col] < b0 + wl_log_i]

        if not df_filtered.empty:
            onset_day = df_filtered[time_col].min()
            onset_date = (
                df_filtered.loc[df_filtered[time_col] == onset_day, "datetime"].values[0]
                if not df_filtered.loc[df_filtered[time_col] == onset_day].empty
                else None
            )
            onset_bmi = (
                df_filtered.loc[df_filtered[time_col] == onset_day, "smoothed_BMI"].values[0]
                if not df_filtered.loc[df_filtered[time_col] == onset_day].empty
                else None
            )

            max_t = df_filtered[time_col].max()
            end_date = (
                df_filtered.loc[df_filtered[time_col] == max_t, "datetime"].values[0]
                if not df_filtered.loc[df_filtered[time_col] == max_t].empty
                else None
            )
            end_bmi = (
                df_filtered.loc[df_filtered[time_col] == max_t, "smoothed_BMI"].values[0]
                if not df_filtered.loc[df_filtered[time_col] == max_t].empty
                else None
            )

            results.append(
                {
                    "start_day": t0,
                    "start_date": current_date,
                    "start_bmi": current_bmi,
                    "onset_day": onset_day,
                    "onset_date": onset_date,
                    "onset_bmi": onset_bmi,
                    "end_day": max_t,
                    "end_date": end_date,
                    "end_bmi": end_bmi,
                }
            )

    if len(results) == 0:
        results.append(
            {
                "start_day": None,
                "start_date": None,
                "start_bmi": None,
                "onset_day": None,
                "onset_date": None,
                "onset_bmi": None,
                "end_day": None,
                "end_date": None,
                "end_bmi": None,
            }
        )

    merged_episodes_df = merge_episodes(pd.DataFrame(results), start_col="start_day", end_col="end_day")

    if recovery:
        merged_episodes_df = identify_recovery_episodes(df, merged_episodes_df, time_col, bmi_col)

    return merged_episodes_df


# ------------- Merging Episodes ----------------------
def merge_episodes(df, start_col, end_col):
    """
    Merges overlapping cachexia episodes, keeping start, onset, end, and associated dates/BMI.
    """
    merged_episodes = []
    current_episode = None

    for i in range(df.shape[0]):
        if current_episode is None:
            current_episode = df.iloc[i].to_dict()
        elif df[start_col][i] <= current_episode["end_day"]:
            current_episode["end_day"] = max(current_episode["end_day"], df[end_col][i])
            current_episode["end_date"] = df.loc[i, "end_date"]
            current_episode["end_bmi"] = df.loc[i, "end_bmi"]
        else:
            merged_episodes.append(current_episode)
            current_episode = df.iloc[i].to_dict()

    if current_episode is not None:
        merged_episodes.append(current_episode)

    return pd.DataFrame(merged_episodes)


# ---------------- Edema QC helpers ----------------
def identify_edema_windows_log(df_mrn, time_col, log_bmi_col, window_days=30, up_frac=0.05, return_frac=0.02):
    """
    Detect edema-like reversible *increase* windows:
      baseline -> >= baseline + log(1+up_frac) -> <= baseline + log(1+return_frac)
    within [baseline_day, baseline_day + window_days].

    Returns DataFrame with: ed_start_day, ed_peak_day, ed_end_day
    """
    df_mrn = df_mrn[[time_col, log_bmi_col]].dropna().sort_values(time_col).reset_index(drop=True)
    if df_mrn.shape[0] < 3:
        return pd.DataFrame(columns=["ed_start_day", "ed_peak_day", "ed_end_day"])

    t = df_mrn[time_col].to_numpy()
    x = df_mrn[log_bmi_col].to_numpy()

    up_log = np.log(1.0 + up_frac)
    ret_log = np.log(1.0 + return_frac)

    out = []
    n = len(t)

    for i in range(n - 2):
        baseline_day = t[i]
        baseline_log = x[i]
        t_end = baseline_day + window_days

        j_end = np.searchsorted(t, t_end, side="right")
        if j_end <= i + 1:
            continue

        win_t = t[i + 1 : j_end]
        win_x = x[i + 1 : j_end]

        inc_idx_rel = np.where(win_x >= (baseline_log + up_log))[0]
        if inc_idx_rel.size == 0:
            continue

        peak_rel = inc_idx_rel[0]
        peak_day = win_t[peak_rel]

        after_x = win_x[peak_rel + 1 :]
        after_t = win_t[peak_rel + 1 :]
        if after_x.size == 0:
            continue

        ret_idx_rel = np.where(after_x <= (baseline_log + ret_log))[0]
        if ret_idx_rel.size == 0:
            continue

        end_day = after_t[ret_idx_rel[0]]
        out.append((baseline_day, peak_day, end_day))

    if len(out) == 0:
        return pd.DataFrame(columns=["ed_start_day", "ed_peak_day", "ed_end_day"])

    return pd.DataFrame(out, columns=["ed_start_day", "ed_peak_day", "ed_end_day"])


def merge_overlapping_windows(ed):
    """Merge overlapping edema windows per MRN (ed_start_day/ed_end_day)."""
    if ed.empty:
        return ed
    ed = ed.sort_values("ed_start_day").reset_index(drop=True)
    merged = []
    cur_s = ed.loc[0, "ed_start_day"]
    cur_e = ed.loc[0, "ed_end_day"]

    for k in range(1, ed.shape[0]):
        s = ed.loc[k, "ed_start_day"]
        e = ed.loc[k, "ed_end_day"]
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return pd.DataFrame(merged, columns=["ed_start_day", "ed_end_day"])


def flag_episode_edema_overlap(ep_mrn, ed_mrn):
    """Add boolean edema_overlap for episodes that overlap any edema window."""
    ep_mrn = ep_mrn.copy()
    ep_mrn["edema_overlap"] = False

    if ed_mrn.empty or ep_mrn.empty:
        return ep_mrn

    for idx in ep_mrn.index:
        s = ep_mrn.at[idx, "start_day"]
        e = ep_mrn.at[idx, "end_day"]
        if pd.isna(s) or pd.isna(e):
            continue
        ep_mrn.at[idx, "edema_overlap"] = bool(((ed_mrn["ed_start_day"] <= e) & (ed_mrn["ed_end_day"] >= s)).any())

    return ep_mrn




# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    # ---- load ----
    dx_final = pd.read_csv(dx_fp, low_memory=False)
    bmi_final = pd.read_csv(bmi_fp)

    # ---- minimal hygiene ----
    dx_final.columns = dx_final.columns.str.strip()
    bmi_final.columns = bmi_final.columns.str.strip()

    dx_final["MRN"] = pd.to_numeric(dx_final["MRN"], errors="coerce").astype("Int64")
    bmi_final["MRN"] = pd.to_numeric(bmi_final["MRN"], errors="coerce").astype("Int64")

    dx_final["anchor_final"] = pd.to_datetime(dx_final["anchor_final"], errors="coerce")
    bmi_final["datetime"] = pd.to_datetime(bmi_final["datetime"], errors="coerce")
    bmi_final["bmi"] = pd.to_numeric(bmi_final["bmi"], errors="coerce")
    bmi_final["anchor_final"] = pd.to_datetime(bmi_final["anchor_final"], errors="coerce")

    # ---- anchor-filter BMI ----
    bmi_ts = bmi_final.dropna(subset=["MRN", "datetime", "bmi", "anchor_final"]).copy()

    bmi_ts["days_since_anchor"] = (bmi_ts["datetime"] - bmi_ts["anchor_final"]).dt.days
    bmi_ts = bmi_ts[bmi_ts["days_since_anchor"] >= 0].copy()

    bmi_ts = (
        bmi_ts.sort_values(["MRN", "datetime", "bmi"])
        .drop_duplicates(subset=["MRN", "datetime"], keep="first")
        .sort_values(["MRN", "datetime"])
        .reset_index(drop=True)
    )

    print("bmi_final unique MRN:", bmi_final["MRN"].nunique(dropna=True))
    print("bmi_final rows:", bmi_final.shape[0])
    print("bmi_ts unique MRN:", bmi_ts["MRN"].nunique(dropna=True))
    print("bmi_ts rows:", bmi_ts.shape[0])


    # ----------------------------
    # Smooth once (alpha fixed)
    # ----------------------------
    patient_data = bmi_ts.groupby("MRN").apply(smooth_bmi_ewma, SMOOTH_COL, ALPHA).reset_index(drop=True)
    patient_data[BMI_LOG_COL] = np.log(patient_data["smoothed_BMI"])

    stamp = os.environ.get("CACHEXIA_RUN_STAMP", datetime.now().strftime("%Y%m%d"))
    smoothed_bmi_file = os.path.join(REV_RESULTS, f"smoothed_bmi_all_patients_{stamp}_alpha{ALPHA}.csv")
    patient_data.to_csv(smoothed_bmi_file, index=False)
    print("Saved:", smoothed_bmi_file)

    # ----------------------------
    # Detect episodes (WL5/WL10/WL15)
    # ----------------------------
    MRNS = patient_data["MRN"].unique()

    for label, wl_frac in WL_RUNS.items():
        df_episodes_all = pd.DataFrame()

        for mrn in tqdm(MRNS, desc=f"Processing patients {label}"):
            df_mrn = patient_data[patient_data["MRN"] == mrn].copy()
            df_mrn.reset_index(drop=True, inplace=True)

            df_episodes_merged = identify_cachexia_episodes_2(
                df_mrn,
                time_col=TIME_COL,
                bmi_col=BMI_LOG_COL,
                recovery=True,
                wl_frac=wl_frac,
                bmi20_rule=(EPISODE_MODE == "bmi20_rule" or BMI20_RULE),
            )

            df_episodes_merged["MRN"] = mrn
            df_episodes_merged["wl_threshold"] = label

            df_episodes_all = pd.concat([df_episodes_all, df_episodes_merged], ignore_index=True)

    cols = ["MRN"] + [c for c in df_episodes_all.columns if c != "MRN"]
    df_episodes_all = df_episodes_all[cols]

    out_fp = os.path.join(REV_RESULTS, f"df_episodes_all_precomp_{label}_{stamp}.csv")
    df_episodes_all.to_csv(out_fp, index=False)
    print("Saved:", out_fp)
    print("Rows:", df_episodes_all.shape[0], "| Unique MRN:", df_episodes_all["MRN"].nunique())

    n_total = df_episodes_all["MRN"].nunique()
    n_any = df_episodes_all.dropna(subset=["start_day"])["MRN"].nunique()
    print(f"Incidence (>=1 episode, pre-QC): {n_any:,}/{n_total:,} ({100*n_any/n_total:.2f}%)")

    # ----------------------------
    # Episodes Pre-Edema QC: duration + WL%
    # ----------------------------
    WL_LABELS = ["WL5", "WL10", "WL15"]

    for label in WL_LABELS:
        episodes_file = os.path.join(REV_RESULTS, f"df_episodes_all_precomp_{label}_{stamp}.csv")
        df_episodes_all = pd.read_csv(episodes_file)

        all_patients = pd.DataFrame(df_episodes_all["MRN"].unique(), columns=["MRN"])
        n_total = all_patients["MRN"].nunique()

        episodes_only = df_episodes_all.dropna(subset=["start_day", "end_day"]).copy()

        n_any_pre = episodes_only["MRN"].nunique()
        print(f"Pre-QC incidence (>=1 episode): {n_any_pre:,}/{n_total:,} ({100*n_any_pre/n_total:.2f}%)")
        print(f"Pre-QC episode rows: {episodes_only.shape[0]:,}")

        valid_episodes = episodes_only.copy()
        valid_episodes["episode_duration"] = (valid_episodes["end_day"] - valid_episodes["start_day"]).astype(int)
        valid_episodes = valid_episodes[valid_episodes["episode_duration"] >= MIN_DUR].copy()

        valid_episodes["weight_loss"] = (
            (valid_episodes["start_bmi"] - valid_episodes["end_bmi"]) / valid_episodes["start_bmi"] * 100
        )
        valid_episodes = valid_episodes[valid_episodes["weight_loss"] >= MIN_WL_PCT].copy()

        print(f"Valid episode rows after QC: {valid_episodes.shape[0]:,}")

        n_any_post = valid_episodes["MRN"].nunique()
        print(f"Post-QC incidence (>=1 valid episode): {n_any_post:,}/{n_total:,} ({100*n_any_post/n_total:.2f}%)")

        episode_summary = all_patients.merge(valid_episodes, on="MRN", how="left")
        episode_summary["has_cachexia_valid"] = episode_summary["MRN"].isin(valid_episodes["MRN"]).astype(int)

        out_all = os.path.join(REV_RESULTS, f"episode_summary_valid_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}.csv")
        out_valid = os.path.join(REV_RESULTS, f"valid_episodes_only_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}.csv")

        episode_summary.to_csv(out_all, index=False)
        valid_episodes.to_csv(out_valid, index=False)

        print("Saved:", out_all)
        print("Saved:", out_valid)

    # ----------------------------
    # Edema QC (LOG-space reversible increase windows)
    # ----------------------------
    SMOOTH_FP = smoothed_bmi_file
    print("SMOOTH_FP:", SMOOTH_FP)

    patient_data2 = pd.read_csv(SMOOTH_FP)
    patient_data2[TIME_COL] = pd.to_numeric(patient_data2[TIME_COL], errors="coerce")
    patient_data2[BMI_LOG_COL] = pd.to_numeric(patient_data2[BMI_LOG_COL], errors="coerce")

    edema_dict = {}
    edema_out = []

    for mrn, df_mrn in tqdm(patient_data2.groupby("MRN"), desc="Computing edema windows (log)"):
        ed = identify_edema_windows_log(
            df_mrn,
            time_col=TIME_COL,
            log_bmi_col=BMI_LOG_COL,
            window_days=EDEMA_WINDOW_DAYS,
            up_frac=UP_FRAC,
            return_frac=RETURN_FRAC,
        )
        if not ed.empty:
            merged = merge_overlapping_windows(ed[["ed_start_day", "ed_end_day"]])
            edema_dict[mrn] = merged
            tmp_ed = merged.copy()
            tmp_ed["MRN"] = mrn
            edema_out.append(tmp_ed)
        else:
            edema_dict[mrn] = pd.DataFrame(columns=["ed_start_day", "ed_end_day"])

    edema_df = pd.concat(edema_out, ignore_index=True) if len(edema_out) else pd.DataFrame(columns=["MRN", "ed_start_day", "ed_end_day"])
    edema_fp = os.path.join(
        REV_RESULTS,
        f"edema_windows_{stamp}_LOG_w{EDEMA_WINDOW_DAYS}_up{int(UP_FRAC*100)}_ret{int(RETURN_FRAC*100)}.csv",
    )
    edema_df.to_csv(edema_fp, index=False)
    print("Saved:", edema_fp, "| rows:", edema_df.shape[0])

    inc_postQC = {}
    inc_postQC_edema = {}

    for label in WL_LABELS:
        print("\n==============================")
        print("Edema QC apply:", label)
        print("==============================")

        valid_fp = os.path.join(REV_RESULTS, f"valid_episodes_only_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}.csv")
        denom_fp = os.path.join(REV_RESULTS, f"episode_summary_valid_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}.csv")

        df_valid = pd.read_csv(valid_fp)
        denom = pd.read_csv(denom_fp)[["MRN"]].drop_duplicates()

        n_total = denom["MRN"].nunique()
        n_any_postQC = df_valid["MRN"].nunique()
        inc_postQC[label] = 100 * n_any_postQC / n_total if n_total else 0.0

        flagged = []
        for mrn, ep_mrn in tqdm(df_valid.groupby("MRN"), desc=f"Flagging overlap {label}"):
            ed_mrn = edema_dict.get(mrn, pd.DataFrame(columns=["ed_start_day", "ed_end_day"]))
            flagged.append(flag_episode_edema_overlap(ep_mrn, ed_mrn))

        df_flagged = pd.concat(flagged, ignore_index=True) if len(flagged) else df_valid.copy()
        df_noed = df_flagged.loc[~df_flagged["edema_overlap"]].copy()

        n_any_post_ed = df_noed["MRN"].nunique()
        inc_postQC_edema[label] = 100 * n_any_post_ed / n_total if n_total else 0.0

        episode_summary_edema = denom.merge(df_noed, on="MRN", how="left")
        episode_summary_edema["has_cachexia_valid_edemaQC"] = episode_summary_edema["MRN"].isin(df_noed["MRN"]).astype(int)

        out_valid = os.path.join(
            REV_RESULTS,
            f"valid_episodes_only_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}_edemaQC_LOG_w{EDEMA_WINDOW_DAYS}_up{int(UP_FRAC*100)}_ret{int(RETURN_FRAC*100)}.csv",
        )
        out_sum = os.path.join(
            REV_RESULTS,
            f"episode_summary_valid_{label}_{stamp}_dur{MIN_DUR}_wl{int(MIN_WL_PCT)}_edemaQC_LOG_w{EDEMA_WINDOW_DAYS}_up{int(UP_FRAC*100)}_ret{int(RETURN_FRAC*100)}.csv",
        )

        df_noed.to_csv(out_valid, index=False)
        episode_summary_edema.to_csv(out_sum, index=False)

        print(f"{label} post-QC incidence:       {n_any_postQC:,}/{n_total:,} ({inc_postQC[label]:.2f}%)")
        print(f"{label} post-QC+edema incidence: {n_any_post_ed:,}/{n_total:,} ({inc_postQC_edema[label]:.2f}%)")


    