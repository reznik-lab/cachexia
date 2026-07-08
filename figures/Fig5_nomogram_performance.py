#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-cancer-type nomogram performance: reduced multivariate Cox model (covariates
significant at FDR<0.1 from Fig5_multivariate_mutation_cox.py), evaluated with:
  - optimism-corrected bootstrap C-index (Harrell's method, 200 resamples)
  - repeated 50/50-split time-dependent AUC at 365/730/1095 days (200 splits)
  - a single 50/50-split calibration plot (quartile-binned predicted vs observed)

Adapted from 0303_ccx_revisions/rev_code/{nomogram_test.py, c_index_check.py} -
merged into one script since the two originals were near-duplicates that had
diverged (c_index_check.py had the correct optimism-correction and a covariate
dummy-name-to-column mapper that nomogram_test.py lacked; nomogram_test.py had
the calibration/AUC plots that c_index_check.py lacked). Also fixes the same
CRC Sidedness bug as Fig5_multivariate_mutation_cox.py.

Usage: python Fig5_nomogram_performance.py <cancer_type>
  e.g. python Fig5_nomogram_performance.py "Lung Adenocarcinoma"
       python Fig5_nomogram_performance.py "Colorectal Adenocarcinoma"
       python Fig5_nomogram_performance.py "Prostate Adenocarcinoma"
"""
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

CANCER_TYPE = sys.argv[1]

BASE_REV   = "."
REV_INPUTS = os.path.join(BASE_REV, "rev_inputs")
REV_MUTS   = os.path.join(BASE_REV, "rev_muts")
REV_PLOTS  = os.path.join(BASE_REV, "rev_plots")
DATE_STAMP = "20260706"
WL_LABEL   = "WL5_BMIlt20"

mut_run_dir = os.path.join(REV_MUTS, f"results_mutation_{WL_LABEL}_{DATE_STAMP}")
out_dir = os.path.join(mut_run_dir, f"nomogram_{WL_LABEL}_{DATE_STAMP}")
os.makedirs(out_dir, exist_ok=True)
fig_dir = os.path.join(REV_PLOTS, "fearon_definition", "Fig5")
os.makedirs(fig_dir, exist_ok=True)

HORIZONS = (365, 730, 1095)
BOOT_B = 200
REPEAT_SPLITS = 200
FDR_THR = 0.1
safe_ct = re.sub(r"[^A-Za-z0-9]", "_", CANCER_TYPE)


def map_sig_covariates_to_df_columns(sig_covs, df_columns):
    """Collapse dummy-encoded lifelines formula terms (e.g. 'MSI_TYPE[T.Instable]')
    back to their base column name (e.g. 'MSI_TYPE'), so the reduced-model formula
    references real columns instead of silently dropping categorical hits."""
    mapped = []
    for c in sig_covs:
        if c in df_columns:
            mapped.append(c)
            continue
        base = re.sub(r"\[T\..*\]$", "", c)
        if base in df_columns and base not in mapped:
            mapped.append(base)
    return mapped


# ---------------------------- data prep (same as multivariate script) ----------------------------
metadata_fp = os.path.join(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
metadata = pd.read_csv(metadata_fp, header=0)
metadata = metadata.rename(columns={"anchor_final": "Tumor Diagnosis Date"})

bmi_fp = os.path.join(REV_INPUTS, "bmi_final_20260129.csv")
bmi = pd.read_csv(bmi_fp, header=0)
bmi = bmi.merge(metadata[["MRN", "Tumor Diagnosis Date"]], on="MRN", how="left")
bmi["Days_Since_Diagnosis"] = (pd.to_datetime(bmi["datetime"]) - pd.to_datetime(bmi["Tumor Diagnosis Date"])).dt.days
bmi = bmi.dropna(subset=["Days_Since_Diagnosis"], axis=0)

cachexia_data = pd.read_csv(
    os.path.join(mut_run_dir, f"cachexia_data_survival_mutation_{WL_LABEL}_{DATE_STAMP}.csv"), low_memory=False
)
cachexia_data["Tumor Diagnosis Date"] = pd.to_datetime(cachexia_data["Tumor Diagnosis Date"])
cachexia_data["PT_BIRTH_DTE"] = pd.to_datetime(cachexia_data["PT_BIRTH_DTE"])
cachexia_data["age_at_diagnosis"] = (cachexia_data["Tumor Diagnosis Date"] - cachexia_data["PT_BIRTH_DTE"]).dt.days / 3652.5
cachexia_data = cachexia_data.dropna(
    subset=["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR"], axis=0
)
cachexia_data.loc[cachexia_data["MSI_TYPE"] == "Stable", "MSI_TYPE"] = "AStable"

ecog = pd.read_csv(os.path.join(REV_INPUTS, "data_timeline_ecog_kps.txt"), sep="\t", header=0)
ecog = ecog.sort_values(by=["PATIENT_ID", "START_DATE"])
ecog.reset_index(inplace=True)
ecog_baseline = ecog.groupby("PATIENT_ID", group_keys=False).apply(lambda x: x.loc[(x["START_DATE"]).abs().idxmin()])
ecog_baseline.rename(columns={"PATIENT_ID": "DMP_ID"}, inplace=True)
cachexia_data = cachexia_data.merge(ecog_baseline[["DMP_ID", "ECOG_KPS"]], on="DMP_ID", how="inner")

left_right_tumor = pd.read_csv(os.path.join(REV_INPUTS, "CRC_Primary_Tumor_Location_01_24_25.csv"), encoding="utf-8-sig")
left_right_tumor = left_right_tumor.rename(columns={"PATIENT_ID": "DMP_ID", "Sidedness": "Sidedness_crcfile"})
cachexia_data = cachexia_data.merge(left_right_tumor[["DMP_ID", "Sidedness_crcfile"]], on="DMP_ID", how="left")
if "Sidedness" not in cachexia_data.columns:
    cachexia_data["Sidedness"] = np.nan
cachexia_data["Sidedness"] = cachexia_data["Sidedness_crcfile"].fillna(cachexia_data["Sidedness"])
cachexia_data = cachexia_data.drop(columns=["Sidedness_crcfile"])

bmi_baseline = bmi.groupby("MRN", group_keys=False).apply(lambda x: x.loc[(x["Days_Since_Diagnosis"]).abs().idxmin()])
bmi_baseline.reset_index(inplace=True, drop=True)
cachexia_data = cachexia_data.merge(bmi_baseline[["MRN", "bmi"]], on="MRN", how="inner")
cachexia_data.rename(columns={"bmi": "start_BMI"}, inplace=True)
cachexia_data["start_BMI"] = cachexia_data["start_BMI"] / 5

# ---------------------------- reduced-model covariate selection ----------------------------
mv_fp = os.path.join(mut_run_dir, f"multivariate_{WL_LABEL}_{DATE_STAMP}", "cancer_types", f"{CANCER_TYPE}.csv")
if not os.path.exists(mv_fp):
    print(f"[SKIP] No multivariate CSV for {CANCER_TYPE} at {mv_fp}")
    sys.exit(0)

mv = pd.read_csv(mv_fp, index_col=0)
sig_covs_raw = mv[mv["p_adj"] < FDR_THR].index.tolist()

df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == CANCER_TYPE].copy()
sig_covs = map_sig_covariates_to_df_columns(sig_covs_raw, df.columns)
print(f"[{CANCER_TYPE}] significant covariates (raw): {sig_covs_raw}")
print(f"[{CANCER_TYPE}] mapped to df columns: {sig_covs}")

if not sig_covs:
    print(f"[SKIP] No significant covariates for {CANCER_TYPE}")
    sys.exit(0)

use_cols = sig_covs + ["time_to_cachexia", "cachexia_event"]
df = df.dropna(subset=use_cols).reset_index(drop=True)
print(f"[{CANCER_TYPE}] n={df.shape[0]}, events={df['cachexia_event'].sum()}")

formula = " + ".join(sig_covs)
cph_full = CoxPHFitter()
cph_full.fit(df, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
c_full = cph_full.concordance_index_
print(f"[{CANCER_TYPE}] Full-data C-index: {c_full:.3f}")

# ---------------------------- optimism-corrected bootstrap C-index ----------------------------
rng = np.random.default_rng(42)
optimism_list = []
for b in range(BOOT_B):
    boot_idx = rng.choice(df.index, size=len(df), replace=True)
    # reset_index is essential: df.loc[boot_idx] has a duplicate (non-unique)
    # index from with-replacement sampling, and lifelines' cached
    # .concordance_index_ property computes silently wrong (~0.50, near-random)
    # on non-unique-index data - manually recomputing concordance_index on the
    # same predictions gives the correct value once the index is unique again.
    df_boot = df.loc[boot_idx].reset_index(drop=True)
    try:
        cph_b = CoxPHFitter()
        cph_b.fit(df_boot, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
    except Exception:
        continue
    c_boot_boot = cph_b.concordance_index_
    pred_orig = cph_b.predict_partial_hazard(df)
    c_boot_orig = concordance_index(df["time_to_cachexia"], -pred_orig, df["cachexia_event"])
    optimism_list.append(c_boot_boot - c_boot_orig)

optimism = np.array(optimism_list)
c_corrected = c_full - optimism.mean()
c_ci = np.percentile(c_full - optimism, [2.5, 97.5])
print(f"[{CANCER_TYPE}] Optimism-corrected C-index: {c_corrected:.3f} (95% CI {c_ci[0]:.3f}-{c_ci[1]:.3f}), n_boot={len(optimism_list)}")

# ---------------------------- repeated 50/50 split time-dependent AUC ----------------------------
auc_records = []
for s in range(REPEAT_SPLITS):
    try:
        train, test = train_test_split(df, test_size=0.5, random_state=s)
        cph_s = CoxPHFitter()
        cph_s.fit(train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

        surv_train = Surv.from_arrays(train["cachexia_event"].astype(bool), train["time_to_cachexia"])
        surv_test = Surv.from_arrays(test["cachexia_event"].astype(bool), test["time_to_cachexia"])
        risk_test = cph_s.predict_partial_hazard(test)

        max_train_time = train["time_to_cachexia"].max()
        eval_horizons = [h for h in HORIZONS if h < max_train_time]
        if not eval_horizons:
            continue
        auc_vals, mean_auc = cumulative_dynamic_auc(surv_train, surv_test, risk_test, eval_horizons)
        rec = {h: np.nan for h in HORIZONS}
        for h, a in zip(eval_horizons, auc_vals):
            rec[h] = a
        auc_records.append(rec)
    except Exception:
        continue

auc_df = pd.DataFrame(auc_records)
auc_summary = {}
for h in HORIZONS:
    vals = auc_df[h].dropna()
    if len(vals) > 0:
        auc_summary[h] = (vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    else:
        auc_summary[h] = (np.nan, np.nan, np.nan)
    print(f"[{CANCER_TYPE}] AUC @ {h}d: {auc_summary[h][0]:.3f} (95% CI {auc_summary[h][1]:.3f}-{auc_summary[h][2]:.3f}), n_splits={len(vals)}")

# ---------------------------- save summary ----------------------------
summary = pd.DataFrame([{
    "cancer_type": CANCER_TYPE,
    "n": df.shape[0],
    "n_events": int(df["cachexia_event"].sum()),
    "c_index_full": c_full,
    "c_index_corrected": c_corrected,
    "c_index_ci_low": c_ci[0],
    "c_index_ci_high": c_ci[1],
    "auc_365": auc_summary[365][0], "auc_365_low": auc_summary[365][1], "auc_365_high": auc_summary[365][2],
    "auc_730": auc_summary[730][0], "auc_730_low": auc_summary[730][1], "auc_730_high": auc_summary[730][2],
    "auc_1095": auc_summary[1095][0], "auc_1095_low": auc_summary[1095][1], "auc_1095_high": auc_summary[1095][2],
    "sig_covariates": ";".join(sig_covs),
}])
summary_fp = os.path.join(out_dir, f"model_perf_summary_{safe_ct}_{DATE_STAMP}.csv")
summary.to_csv(summary_fp, index=False)
print(f"\n[SAVED] {summary_fp}")

# ---------------------------- calibration (all 3 horizons, bootstrap CI) + ROC (all 3 horizons) ----------------------------
# Matches 0303_ccx_revisions/rev_code/nomogram_test.py's calibration_half_split /
# auc_half_split plots exactly (same 50/50 split, same quartile binning off the
# 1-year risk, same bootstrap-CI procedure, same colors). These are SFig5 panels
# (D-F calibration, G-I ROC), not Fig5.
sfig_dir = os.path.join(REV_PLOTS, "fearon_definition", "SFig5")
os.makedirs(sfig_dir, exist_ok=True)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 9

HORIZON_COLORS = {365: "#6388B4FF", 730: "#FFAE34FF", 1095: "#EF6F6AFF"}
HORIZON_MARKERS = {365: "o", 730: "s", 1095: "^"}

train, test = train_test_split(df, test_size=0.5, random_state=42)
cph_cal = CoxPHFitter()
cph_cal.fit(train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

sf = cph_cal.predict_survival_function(test)
times_avail = sf.index.to_numpy(float)
sel = {h: float(times_avail[np.abs(times_avail - h).argmin()]) for h in HORIZONS if h <= times_avail.max()}
time_points = sorted(sel.keys())
print(f"[{CANCER_TYPE}] calibration/ROC horizons used (nearest available day): {sel}")

test = test.copy()
for h, t_near in sel.items():
    test[f"predicted_cachexia_{h}d"] = 1 - sf.loc[t_near].values

pivot_h = 365 if 365 in sel else time_points[0]
test["risk_bin"] = pd.qcut(test[f"predicted_cachexia_{pivot_h}d"], q=4, labels=False, duplicates="drop")

from lifelines import KaplanMeierFitter
n_bootstrap_cal = 200
observed, ci_lower, ci_upper = {h: [] for h in time_points}, {h: [] for h in time_points}, {h: [] for h in time_points}

for grp in sorted(test["risk_bin"].dropna().unique()):
    grp_data = test[test["risk_bin"] == grp]
    for h in time_points:
        kmf = KaplanMeierFitter()
        kmf.fit(grp_data["time_to_cachexia"], event_observed=grp_data["cachexia_event"])
        observed[h].append(1 - kmf.predict(h))
        boot_ests = []
        for bi in range(n_bootstrap_cal):
            rs = grp_data.sample(n=len(grp_data), replace=True, random_state=bi)
            kmf_b = KaplanMeierFitter()
            kmf_b.fit(rs["time_to_cachexia"], event_observed=rs["cachexia_event"])
            boot_ests.append(1 - kmf_b.predict(h))
        ci_lower[h].append(np.percentile(boot_ests, 2.5))
        ci_upper[h].append(np.percentile(boot_ests, 97.5))

cal_data_rows = []
for h in time_points:
    pred_means = test.groupby("risk_bin")[f"predicted_cachexia_{h}d"].mean().values
    for i in range(len(pred_means)):
        cal_data_rows.append({
            "horizon_days": h, "risk_bin": i,
            "pred_mean": pred_means[i], "observed": observed[h][i],
            "ci_low": ci_lower[h][i], "ci_high": ci_upper[h][i],
        })
cal_df = pd.DataFrame(cal_data_rows)
cal_fp = os.path.join(out_dir, f"calibration_data_{safe_ct}_{DATE_STAMP}.csv")
cal_df.to_csv(cal_fp, index=False)

fig, ax = plt.subplots(figsize=(3.5, 3.3))
for h in time_points:
    sub = cal_df[cal_df.horizon_days == h]
    ax.errorbar(sub["pred_mean"], sub["observed"],
                yerr=[sub["observed"] - sub["ci_low"], sub["ci_high"] - sub["observed"]],
                fmt=HORIZON_MARKERS[h], label=f"{round(h/365)}-year incidence",
                color=HORIZON_COLORS[h], capsize=3)
ax.plot([0, 1], [0, 1], "--", label="Perfect Calibration", color="black")
ax.set_xlabel("Nomogram-Predicted Probability")
ax.set_ylabel("Cumulative Incidence")
ax.set_title(CANCER_TYPE)
ax.legend(fontsize=7, markerscale=1, frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
cal_plot_fp = os.path.join(sfig_dir, f"calibration_{safe_ct}_{DATE_STAMP}.pdf")
plt.savefig(cal_plot_fp)
plt.close()
print(f"[SAVED] {cal_plot_fp}")

# ---------------------------- ROC curves (same 50/50 split, all 3 horizons) ----------------------------
from sklearn.metrics import roc_curve

surv_train = Surv.from_arrays(train["cachexia_event"].astype(bool), train["time_to_cachexia"])
surv_test = Surv.from_arrays(test["cachexia_event"].astype(bool), test["time_to_cachexia"])

fig, ax = plt.subplots(figsize=(2.8, 2.8))
for h in time_points:
    auc_h, _ = cumulative_dynamic_auc(surv_train, surv_test, test[f"predicted_cachexia_{h}d"], [h])
    test[f"cachexia_event_{h}d"] = (test["cachexia_event"] & (test["time_to_cachexia"] <= h)).astype(int)
    fpr, tpr, _ = roc_curve(test[f"cachexia_event_{h}d"], test[f"predicted_cachexia_{h}d"])
    ax.plot(fpr, tpr, color=HORIZON_COLORS[h], label=f"{round(h/365)}-Year AUC: {auc_h[0]:.3f}")

ax.plot([0, 1], [0, 1], "--", color="black", label="Random Chance")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title(CANCER_TYPE)
ax.legend(fontsize=7, markerscale=1, frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
roc_plot_fp = os.path.join(sfig_dir, f"roc_{safe_ct}_{DATE_STAMP}.pdf")
plt.savefig(roc_plot_fp)
plt.close()
print(f"[SAVED] {roc_plot_fp}")

print(f"\n[{CANCER_TYPE}] Done.")
