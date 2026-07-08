#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-cancer-type, per-gene univariate Cox model for time-to-first-cachexia-episode,
with death-before-cachexia as censoring. Fits a cause-specific Cox model (not a
true Fine-Gray competing-risk regression); the Aalen-Johansen cumulative-incidence
curves used for the CIF plots ARE a proper competing-risk estimator, used only for
visualization, not for the p-value/HR.

Adapted from 0303_ccx_revisions/rev_code/univariate_cif.py, parameterized across
the 3 weight-loss thresholds (the original hardcoded "5pct" everywhere and had no
real way to rerun at 10%/15% - the historical 10pct/15pct outputs referenced
downstream were apparently produced by manually hand-editing that hardcoded string
and were never saved, so this rebuild makes that parameterization real). Also fixes
a save/read filename mismatch in the original (saved with a threshold suffix, read
without it).

Usage: python Fig5_univariate_mutation_cox.py <WL_LABEL>
  e.g. python Fig5_univariate_mutation_cox.py WL5_BMIlt20
       python Fig5_univariate_mutation_cox.py WL10
       python Fig5_univariate_mutation_cox.py WL15

Run with working directory set to the project root (containing rev_inputs/,
rev_results/, rev_plots/, rev_tables/, rev_muts/, rev_code/).
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter, AalenJohansenFitter

WL_LABEL = sys.argv[1]
assert WL_LABEL in ("WL5_BMIlt20", "WL10", "WL15"), "WL_LABEL must be WL5_BMIlt20, WL10, or WL15"

EPISODE_FP_MAP = {
    "WL5_BMIlt20": "episode_summary_valid_WL5_BMIlt20rule_20260706_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv",
    "WL10":        "episode_summary_valid_WL10_20260706_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv",
    "WL15":        "episode_summary_valid_WL15_20260706_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv",
}
MIN_WL_PCT_MAP = {"WL5_BMIlt20": 5, "WL10": 10, "WL15": 15}

BASE_REV    = "."
REV_INPUTS  = os.path.join(BASE_REV, "rev_inputs")
REV_RESULTS = os.path.join(BASE_REV, "rev_results")
REV_MUTS    = os.path.join(BASE_REV, "rev_muts")
REV_PLOTS   = os.path.join(BASE_REV, "rev_plots")
DATE_STAMP  = "20260706"

episodes_fp = os.path.join(REV_RESULTS, EPISODE_FP_MAP[WL_LABEL])
metadata_fp = os.path.join(REV_INPUTS, f"dx_cohort_metadata_20260126_v2.csv")
mutation_fp = os.path.join(REV_INPUTS, "IMPACT_Oncogenic_Table_0919_2024.csv")

results_dir = os.path.join(REV_MUTS, f"results_mutation_{WL_LABEL}_{DATE_STAMP}")
fig_dir     = os.path.join(REV_PLOTS, "fearon_definition", "Fig5" if WL_LABEL == "WL5_BMIlt20" else "SFig5")
cif_dir     = os.path.join(fig_dir, "CIF_plots")
os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
os.makedirs(cif_dir, exist_ok=True)

MIN_CANCER_N   = 200
MIN_MUT_RATE   = 0.05
MIN_EPISODE_DUR = 30
MIN_WL_PCT     = MIN_WL_PCT_MAP[WL_LABEL]
FDR_ALPHA      = 0.10
MAX_TIME_DAYS  = 4000

# ---------------------------- episode / outcome prep ----------------------------
cac_episodes = pd.read_csv(episodes_fp)
cac_episodes["episode_duration"] = cac_episodes["end_day"] - cac_episodes["start_day"]
cac_episodes["weight_loss"] = (
    (cac_episodes["start_bmi"] - cac_episodes["end_bmi"]) / cac_episodes["start_bmi"] * 100
)

metadata = pd.read_csv(metadata_fp, header=0)
if "Tumor Diagnosis Date" not in metadata.columns and "anchor_final" in metadata.columns:
    metadata = metadata.rename(columns={"anchor_final": "Tumor Diagnosis Date"})

metadata["PLA_LAST_CONTACT_DTE"] = pd.to_datetime(metadata["PLA_LAST_CONTACT_DTE"])
metadata["Tumor Diagnosis Date"] = pd.to_datetime(metadata["Tumor Diagnosis Date"])
metadata["os_days"] = (metadata["PLA_LAST_CONTACT_DTE"] - metadata["Tumor Diagnosis Date"]).dt.days
metadata["os_event"] = 0
metadata.loc[metadata["PT_DEATH_DTE"].notna(), "os_event"] = 1

# ---------------------------- mutation data ----------------------------
mutation = pd.read_csv(mutation_fp, header=0)
mutation = mutation.rename(columns={mutation.columns[0]: "DMP_ID"})
mutation["DMP_ID"] = mutation["DMP_ID"].astype(str).str[:9]
mutation = mutation.drop_duplicates(subset="DMP_ID", keep="first")
for gene in mutation.columns[1:]:
    mutation.loc[mutation[gene] > 1, gene] = 1
mutation.columns = [col.replace("-", "_") for col in mutation.columns]

# ---------------------------- build cachexia_event / time_to_cachexia ----------------------------
cac_episodes.loc[
    (cac_episodes["start_day"].notna()) & (cac_episodes["episode_duration"] < MIN_EPISODE_DUR),
    "start_day",
] = None
cac_episodes.loc[
    (cac_episodes["start_day"].notna()) & (cac_episodes["weight_loss"] < MIN_WL_PCT),
    "start_day",
] = None

cachexia_data = cac_episodes.groupby("MRN", as_index=False).apply(
    lambda x: x.loc[x["start_day"].idxmin()] if x["start_day"].notna().any() else x.iloc[0]
).reset_index(drop=True)

cachexia_data["cachexia_event"] = cachexia_data["start_day"].notna().astype(int)
cachexia_data = cachexia_data[["start_day", "MRN", "cachexia_event"]]
cachexia_data = cachexia_data.rename(columns={"start_day": "time_to_cachexia"})

cachexia_data = cachexia_data.merge(metadata, on="MRN", how="left")
cachexia_data = cachexia_data.dropna(subset=["os_days", "os_event"])

cachexia_data.loc[cachexia_data["cachexia_event"] == 0, "time_to_cachexia"] = cachexia_data["os_days"]

cachexia_data["event"] = 0
cachexia_data.loc[cachexia_data["os_event"] == 1, "event"] = 2
cachexia_data.loc[cachexia_data["cachexia_event"] == 1, "event"] = 1
cachexia_data["time"] = cachexia_data["os_days"]
cachexia_data.loc[cachexia_data["cachexia_event"] == 1, "time"] = cachexia_data["time_to_cachexia"]

cachexia_data = cachexia_data.merge(mutation, on="DMP_ID", how="left")
cachexia_data = cachexia_data[cachexia_data["AR"].notna()]

cachexia_data = cachexia_data.loc[cachexia_data["time_to_cachexia"] >= 0]
cachexia_data = cachexia_data.loc[cachexia_data["time_to_cachexia"] < MAX_TIME_DAYS]

cachexia_data.loc[
    cachexia_data["CANCER_TYPE_DETAILED"] == "Colon Adenocarcinoma", "CANCER_TYPE_DETAILED"
] = "Colorectal Adenocarcinoma"
cachexia_data.loc[
    cachexia_data["CANCER_TYPE_DETAILED"] == "Rectal Adenocarcinoma", "CANCER_TYPE_DETAILED"
] = "Colorectal Adenocarcinoma"

surv_fp = os.path.join(results_dir, f"cachexia_data_survival_mutation_{WL_LABEL}_{DATE_STAMP}.csv")
cachexia_data.to_csv(surv_fp, index=False)

# ---------------------------- univariate Cox: cachexia ~ gene ----------------------------
cancer_counts = cachexia_data["CANCER_TYPE_DETAILED"].value_counts().reset_index()
cancer_counts.columns = ["CANCER_TYPE_DETAILED", "count"]
all_genes = mutation.columns[1:]

rows = []
for cancer_type in cancer_counts.loc[cancer_counts["count"] > MIN_CANCER_N, "CANCER_TYPE_DETAILED"]:
    df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]
    mutation_rate = df[all_genes].sum() / df.shape[0]
    filtered_genes = mutation_rate[mutation_rate > MIN_MUT_RATE].index.to_list()
    if not filtered_genes:
        continue
    for gene in filtered_genes:
        data = df[df[gene].notna()]
        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col="time_to_cachexia", event_col="cachexia_event", formula=f"{gene}")
        except Exception:
            continue
        row = cph.summary.iloc[0, :].copy()
        row["detailed_cancer_type"] = cancer_type
        row["mutation"] = gene
        row.name = f"{cancer_type}_{gene}"
        rows.append(row)

cox_df = pd.DataFrame(rows)
cox_df["p_adj"] = multipletests(cox_df["p"], method="fdr_bh")[1]

cox_fp = os.path.join(results_dir, f"mutation_cox_cachexia_results_{WL_LABEL}_{DATE_STAMP}.csv")
cox_df.to_csv(cox_fp)
print(f"[SAVED] {cox_fp}  ({len(cox_df)} cancer-gene pairs, {(cox_df['p_adj'] < FDR_ALPHA).sum()} significant at FDR<{FDR_ALPHA})")

# ---------------------------- volcano plot ----------------------------
cox_df_v = cox_df.copy()
cox_df_v["-log2(p_adj)"] = -np.log2(cox_df_v["p_adj"])
cox_df_v["significant"] = "Not significant"
cox_df_v.loc[cox_df_v["p_adj"] < FDR_ALPHA, "significant"] = "Significant"
cox_df_v = cox_df_v[(cox_df_v["coef"] > -14) & (cox_df_v["coef"] < 14)]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cox_df_v, x="coef", y="-log2(p_adj)", hue="significant")
plt.axhline(y=-np.log2(FDR_ALPHA), color="r", linestyle="--")
plt.axvline(x=0, color="r", linestyle="--")
for i in range(cox_df_v[cox_df_v["p_adj"] < FDR_ALPHA].shape[0]):
    sig_row = cox_df_v[cox_df_v["p_adj"] < FDR_ALPHA].iloc[i]
    plt.annotate(cox_df_v[cox_df_v["p_adj"] < FDR_ALPHA].index[i],
                 (sig_row["coef"], sig_row["-log2(p_adj)"]), fontsize=5)
plt.xlabel("log(Hazard Ratio)")
plt.ylabel("-log2(p_adj)")
plt.title(f"Volcano plot for mutation in cachexia ({WL_LABEL})")
plt.savefig(os.path.join(results_dir, "plots", f"volcano_plot_{WL_LABEL}_{DATE_STAMP}.pdf"))
plt.close()

# ---------------------------- CIF plots for significant genes ----------------------------
cox_df_sig = cox_df_v[cox_df_v["p_adj"] < FDR_ALPHA]
WT_COL, MUT_COL = "#6388B4FF", "#EF6F6AFF"

for i in range(cox_df_sig.shape[0]):
    gene = cox_df_sig.iloc[i]["mutation"]
    cancer_type = cox_df_sig.iloc[i]["detailed_cancer_type"]
    df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]

    ajf_wt = AalenJohansenFitter(calculate_variance=True)
    ajf_mut = AalenJohansenFitter(calculate_variance=True)
    ajf_wt.fit(durations=df[df[gene] == 0]["time"], event_observed=df[df[gene] == 0]["event"], event_of_interest=1)
    ajf_mut.fit(durations=df[df[gene] > 0]["time"], event_observed=df[df[gene] > 0]["event"], event_of_interest=1)

    fig, ax = plt.subplots(figsize=(3, 2.6))
    ajf_wt.plot(ax=ax, label=f"WT {gene}", ci_show=True)
    ajf_mut.plot(ax=ax, label=f"MUT {gene}", ci_show=True)

    lines = ax.get_lines()
    if len(lines) >= 2:
        lines[0].set_color(WT_COL); lines[1].set_color(MUT_COL)
        lines[0].set_linewidth(2.0); lines[1].set_linewidth(2.0)
    from matplotlib.collections import PolyCollection
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    if len(polys) >= 2:
        polys[0].set_facecolor(WT_COL); polys[1].set_facecolor(MUT_COL)
        polys[0].set_alpha(0.18); polys[1].set_alpha(0.18)
        polys[0].set_edgecolor("none"); polys[1].set_edgecolor("none")

    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.3)
    ax.tick_params(axis="both", which="both", top=False, right=False, width=0.3, length=3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.margins(x=0, y=0)
    ax.set_title(f"CIF in {cancer_type}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Cumulative Incidence of cachexia")
    ax.legend(frameon=False)
    plt.savefig(os.path.join(cif_dir, f"cif_{gene}_{cancer_type}_{WL_LABEL}.pdf"))
    plt.close()

print(f"\n[{WL_LABEL}] Wrote univariate mutation Cox outputs to: {results_dir}")
