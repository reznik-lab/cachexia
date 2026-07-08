#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFCI NSCLC nomogram performance (matches manuscript's "DFCI C-index 0.596,
n=1159, 516 events"). Same optimism-corrected bootstrap C-index + repeated
50/50-split time-dependent AUC procedure as Fig5_nomogram_performance.py,
applied to the DFCI external validation cohort instead of MSK.
"""
import os
import re
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

BASE_REV   = "."
REV_MUTS   = os.path.join(BASE_REV, "rev_muts")
DATE_STAMP = "20260706"
THRESHOLD  = "5WL_BMI20"
CANCER_TYPE_DFCI = "NSCLC"

dfci_run_dir = os.path.join(REV_MUTS, f"results_dfci_{THRESHOLD}_{DATE_STAMP}")
out_dir = os.path.join(dfci_run_dir, f"nomogram_{THRESHOLD}_{DATE_STAMP}")
os.makedirs(out_dir, exist_ok=True)

HORIZONS = (365, 730, 1095)
BOOT_B = 200
REPEAT_SPLITS = 200
FDR_THR = 0.1


def map_sig_covariates_to_df_columns(sig_covs, df_columns):
    mapped = []
    for c in sig_covs:
        if c in df_columns:
            mapped.append(c)
            continue
        base = re.sub(r"\[T\..*\]$", "", c)
        if base in df_columns and base not in mapped:
            mapped.append(base)
    return mapped


mv_fp = os.path.join(dfci_run_dir, f"multivariate_{THRESHOLD}", "cancer_types",
                      f"{CANCER_TYPE_DFCI}_multivariate_{THRESHOLD}.csv")
mv = pd.read_csv(mv_fp, index_col=0)
sig_covs_raw = mv[mv["p_adj"] < FDR_THR].index.tolist()
print(f"[{CANCER_TYPE_DFCI}] significant covariates (raw): {sig_covs_raw}")

# rebuild the same design-matrix-based df used for the DFCI multivariate fit
import glob
DFCI_ROOT = os.path.join(REV_MUTS, "FILE_SHARED_WITH_MSKCC_V1", "deidentified_data")
COMBINATION = "diagdate_1ca_tier1or2or3_bmi"
NON_GENE_COLS = {
    "patient_id", "cachexia_event", "time_to_cachexia", "os_days", "os_event",
    "event", "time", "age_at_diagnosis_binned", "STAGE_CDM_DERIVED_GRANULAR",
    "CANCER_TYPE_DETAILED", "CVR_TMB_SCORE", "SAMPLE_TYPE", "GENDER",
    "ANCESTRY_LABEL", "cancer_type", "combination", "threshold",
}

fp = os.path.join(DFCI_ROOT, f"{CANCER_TYPE_DFCI}_cachexia", COMBINATION, THRESHOLD, "patient_data_deidentified.csv")
raw = pd.read_csv(fp, low_memory=False)

covars = [c for c in ["age_at_diagnosis_binned", "GENDER", "STAGE_CDM_DERIVED_GRANULAR",
                       "SAMPLE_TYPE", "ANCESTRY_LABEL", "CVR_TMB_SCORE"] if c in raw.columns]
gene_covs = [c for c in sig_covs_raw if c in raw.columns and c not in NON_GENE_COLS]
variables = gene_covs + covars

df = raw[raw["time_to_cachexia"] >= 0].copy()
use_cols = variables + ["time_to_cachexia", "cachexia_event"]
df = df.dropna(subset=[c for c in use_cols if c in df.columns])

df_design = pd.get_dummies(df[variables], drop_first=True)
for c in df_design.columns:
    if df_design[c].value_counts(normalize=True).iloc[0] > 0.99:
        df_design = df_design.drop(columns=[c])
df_design["time_to_cachexia"] = df["time_to_cachexia"].values
df_design["cachexia_event"] = df["cachexia_event"].values
df_design = df_design.loc[:, ~df_design.columns.duplicated()]
df_design = df_design.astype({c: float for c in df_design.columns if df_design[c].dtype == bool})
df_design = df_design.reset_index(drop=True)

sig_covs = map_sig_covariates_to_df_columns(sig_covs_raw, df_design.columns)
print(f"[{CANCER_TYPE_DFCI}] mapped to design columns: {sig_covs}")
if not sig_covs:
    print("[SKIP] No significant covariates map to design columns")
    raise SystemExit(0)

formula = " + ".join(sig_covs)
df_design = df_design.dropna(subset=sig_covs + ["time_to_cachexia", "cachexia_event"]).reset_index(drop=True)
print(f"[{CANCER_TYPE_DFCI}] n={df_design.shape[0]}, events={int(df_design['cachexia_event'].sum())}")

cph_full = CoxPHFitter()
cph_full.fit(df_design, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
c_full = cph_full.concordance_index_
print(f"[{CANCER_TYPE_DFCI}] Full-data C-index: {c_full:.3f}")

rng = np.random.default_rng(42)
optimism_list = []
for b in range(BOOT_B):
    boot_idx = rng.choice(df_design.index, size=len(df_design), replace=True)
    df_boot = df_design.loc[boot_idx].reset_index(drop=True)
    try:
        cph_b = CoxPHFitter()
        cph_b.fit(df_boot, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
    except Exception:
        continue
    c_boot_boot = cph_b.concordance_index_
    pred_orig = cph_b.predict_partial_hazard(df_design)
    c_boot_orig = concordance_index(df_design["time_to_cachexia"], -pred_orig, df_design["cachexia_event"])
    optimism_list.append(c_boot_boot - c_boot_orig)

optimism = np.array(optimism_list)
c_corrected = c_full - optimism.mean()
c_ci = np.percentile(c_full - optimism, [2.5, 97.5])
print(f"[{CANCER_TYPE_DFCI}] Optimism-corrected C-index: {c_corrected:.3f} (95% CI {c_ci[0]:.3f}-{c_ci[1]:.3f}), n_boot={len(optimism_list)}")

auc_records = []
for s in range(REPEAT_SPLITS):
    try:
        train, test = train_test_split(df_design, test_size=0.5, random_state=s)
        cph_s = CoxPHFitter()
        cph_s.fit(train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
        surv_train = Surv.from_arrays(train["cachexia_event"].astype(bool), train["time_to_cachexia"])
        surv_test = Surv.from_arrays(test["cachexia_event"].astype(bool), test["time_to_cachexia"])
        risk_test = cph_s.predict_partial_hazard(test)
        max_train_time = train["time_to_cachexia"].max()
        eval_horizons = [h for h in HORIZONS if h < max_train_time]
        if not eval_horizons:
            continue
        auc_vals, _ = cumulative_dynamic_auc(surv_train, surv_test, risk_test, eval_horizons)
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
    print(f"[{CANCER_TYPE_DFCI}] AUC @ {h}d: {auc_summary[h][0]:.3f} (95% CI {auc_summary[h][1]:.3f}-{auc_summary[h][2]:.3f}), n_splits={len(vals)}")

summary = pd.DataFrame([{
    "cancer_type": f"{CANCER_TYPE_DFCI} (DFCI)",
    "n": df_design.shape[0], "n_events": int(df_design["cachexia_event"].sum()),
    "c_index_full": c_full, "c_index_corrected": c_corrected,
    "c_index_ci_low": c_ci[0], "c_index_ci_high": c_ci[1],
    "auc_365": auc_summary[365][0], "auc_730": auc_summary[730][0], "auc_1095": auc_summary[1095][0],
    "sig_covariates": ";".join(sig_covs),
}])
summary_fp = os.path.join(out_dir, f"model_perf_summary_{CANCER_TYPE_DFCI}_DFCI_{DATE_STAMP}.csv")
summary.to_csv(summary_fp, index=False)
print(f"\n[SAVED] {summary_fp}")

# ---------------------------- calibration (J) + ROC (K), DFCI-specific 3-color scheme ----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 9

REV_PLOTS = os.path.join(BASE_REV, "rev_plots")
sfig_dir = os.path.join(REV_PLOTS, "fearon_definition", "SFig5")
os.makedirs(sfig_dir, exist_ok=True)

DFCI_HORIZON_COLORS = {365: "#4FA3A3", 730: "#B5A22A", 1095: "#B0578D"}
DFCI_HORIZON_MARKERS = {365: "o", 730: "s", 1095: "^"}

train, test = train_test_split(df_design, test_size=0.5, random_state=42)
cph_cal = CoxPHFitter()
cph_cal.fit(train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

sf = cph_cal.predict_survival_function(test)
times_avail = sf.index.to_numpy(float)
sel_jk = {h: float(times_avail[np.abs(times_avail - h).argmin()]) for h in HORIZONS if h <= times_avail.max()}
time_points_jk = sorted(sel_jk.keys())
print(f"[{CANCER_TYPE_DFCI}] J/K horizons used (nearest available day): {sel_jk}")

test = test.copy()
for h, t_near in sel_jk.items():
    test[f"predicted_cachexia_{h}d"] = 1 - sf.loc[t_near].values

pivot_h = 365 if 365 in sel_jk else time_points_jk[0]
test["risk_bin"] = pd.qcut(test[f"predicted_cachexia_{pivot_h}d"], q=4, labels=False, duplicates="drop")

n_bootstrap_cal = 200
observed, ci_lower, ci_upper = {h: [] for h in time_points_jk}, {h: [] for h in time_points_jk}, {h: [] for h in time_points_jk}
for grp in sorted(test["risk_bin"].dropna().unique()):
    grp_data = test[test["risk_bin"] == grp]
    for h in time_points_jk:
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

cal_rows = []
for h in time_points_jk:
    pred_means = test.groupby("risk_bin")[f"predicted_cachexia_{h}d"].mean().values
    for i in range(len(pred_means)):
        cal_rows.append({"horizon_days": h, "risk_bin": i, "pred_mean": pred_means[i],
                          "observed": observed[h][i], "ci_low": ci_lower[h][i], "ci_high": ci_upper[h][i]})
cal_df_jk = pd.DataFrame(cal_rows)
cal_df_jk.to_csv(os.path.join(out_dir, f"calibration_data_{CANCER_TYPE_DFCI}_DFCI_{DATE_STAMP}.csv"), index=False)

fig, ax = plt.subplots(figsize=(3.5, 3.3))
for h in time_points_jk:
    sub = cal_df_jk[cal_df_jk.horizon_days == h]
    ax.errorbar(sub["pred_mean"], sub["observed"],
                yerr=[sub["observed"] - sub["ci_low"], sub["ci_high"] - sub["observed"]],
                fmt=DFCI_HORIZON_MARKERS[h], label=f"{round(h/365)}y", color=DFCI_HORIZON_COLORS[h], capsize=3)
ax.plot([0, 1], [0, 1], "--", label="Perfect", color="gray")
ax.set_xlabel("Mean predicted risk")
ax.set_ylabel("Observed event proportion")
ax.set_title(f"{CANCER_TYPE_DFCI} DFCI Test Set")
ax.legend(fontsize=7, markerscale=1, frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
cal_plot_fp = os.path.join(sfig_dir, f"calibration_{CANCER_TYPE_DFCI}_DFCI_{DATE_STAMP}.pdf")
plt.savefig(cal_plot_fp)
plt.close()
print(f"[SAVED] {cal_plot_fp}")

surv_train_jk = Surv.from_arrays(train["cachexia_event"].astype(bool), train["time_to_cachexia"])
surv_test_jk = Surv.from_arrays(test["cachexia_event"].astype(bool), test["time_to_cachexia"])

fig, ax = plt.subplots(figsize=(2.8, 2.8))
for h in time_points_jk:
    auc_h, _ = cumulative_dynamic_auc(surv_train_jk, surv_test_jk, test[f"predicted_cachexia_{h}d"], [h])
    test[f"cachexia_event_{h}d"] = (test["cachexia_event"] & (test["time_to_cachexia"] <= h)).astype(int)
    fpr, tpr, _ = roc_curve(test[f"cachexia_event_{h}d"], test[f"predicted_cachexia_{h}d"])
    n_h = int(test[f"cachexia_event_{h}d"].shape[0])
    ax.plot(fpr, tpr, color=DFCI_HORIZON_COLORS[h], label=f"{round(h/365)}y  AUC={auc_h[0]:.3f}  (n={n_h})")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(f"{CANCER_TYPE_DFCI} DFCI Test Set")
ax.legend(fontsize=6.5, markerscale=1, frameon=False, loc="lower right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
roc_plot_fp = os.path.join(sfig_dir, f"roc_{CANCER_TYPE_DFCI}_DFCI_{DATE_STAMP}.pdf")
plt.savefig(roc_plot_fp)
plt.close()
print(f"[SAVED] {roc_plot_fp}")
