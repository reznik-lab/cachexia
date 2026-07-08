#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFCI external validation cohort: univariate mutation Cox (discovery), CIF plots,
and multivariate Cox, using DFCI's own pre-computed cachexia labels (an
independent, external pipeline run by DFCI on their own cohort - not affected by
the MSK BMI-smoothing bug, since DFCI's episode detection is entirely separate).

Adapted from 0303_ccx_revisions/rev_code/dfci_muts.py, split into proper stages
(the original bundled 3 stages in one file with inconsistent __main__ guarding -
only the multivariate stage was guarded, so importing this module used to run
univariate+CIF unconditionally). Threshold/combination tags are DFCI's own
folder-naming convention (fixed at 5WL_BMI20 / diagdate_1ca_tier1or2or3_bmi -
their cohort wasn't rerun at other thresholds).

Usage: python Fig5_dfci_validation.py <stage>
  stage: univariate | cif | multivariate | all
"""
import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter, AalenJohansenFitter

BASE_REV   = "."
REV_MUTS   = os.path.join(BASE_REV, "rev_muts")
REV_PLOTS  = os.path.join(BASE_REV, "rev_plots")
DATE_STAMP = "20260706"

DFCI_ROOT = os.path.join(REV_MUTS, "FILE_SHARED_WITH_MSKCC_V1", "deidentified_data")
COMBINATION = "diagdate_1ca_tier1or2or3_bmi"
THRESHOLD   = "5WL_BMI20"

RESULTS_DIR = os.path.join(REV_MUTS, f"results_dfci_{THRESHOLD}_{DATE_STAMP}")
CIF_DIR = os.path.join(REV_PLOTS, "fearon_definition", "Fig5", "CIF_plots_DFCI")
os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_N_CT    = 200
MIN_MUTRATE = 0.05
FDR_ALPHA   = 0.10
TIME_CUTOFF = 4000

FOLDER_MAP = {
    "Bladder Urothelial Carcinoma": "BladderCancer",
    "Invasive Breast Carcinoma": "Breastcancer",
    "Colorectal Adenocarcinoma": "CRC",
    "Endometrial Carcinoma": "EndometrialCancer",
    "Esophageal Adenocarcinoma": "EsophagogastricCarcinoma",
    "Glioblastoma Multiforme": "Glioma",
    "Cutaneous Melanoma": "Melanoma",
    "Lung Adenocarcinoma": "NSCLC",
    "Pancreatic Adenocarcinoma": "Pancreaticcancer",
    "Prostate Adenocarcinoma": "ProstateCancer",
    "Renal Clear Cell Carcinoma": "RCC",
}
FOLDER_MAP_REV = {v: k for k, v in FOLDER_MAP.items()}

NON_GENE_COLS = {
    "patient_id", "cachexia_event", "time_to_cachexia", "os_days", "os_event",
    "event", "time", "age_at_diagnosis_binned", "STAGE_CDM_DERIVED_GRANULAR",
    "CANCER_TYPE_DETAILED", "CVR_TMB_SCORE", "SAMPLE_TYPE", "GENDER",
    "ANCESTRY_LABEL", "cancer_type", "combination", "threshold",
}


def load_all_dfci():
    pattern = os.path.join(DFCI_ROOT, "*_cachexia", COMBINATION, THRESHOLD, "patient_data_deidentified.csv")
    files = sorted(glob.glob(pattern))
    frames = []
    for fp in files:
        folder_ct = fp.split(os.sep)[-4].replace("_cachexia", "")
        df = pd.read_csv(fp, low_memory=False)
        df["cancer_type"] = folder_ct
        if "time" not in df.columns or df["time"].isna().all():
            df["time"] = df["time_to_cachexia"]
            df.loc[df["time"].isna(), "time"] = df["os_days"]
        else:
            df["time"] = df["time"].fillna(df["time_to_cachexia"]).fillna(df["os_days"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gene_cols(df):
    return [c for c in df.columns if c not in NON_GENE_COLS]


def stage_univariate():
    all_df = load_all_dfci()
    rows = []
    for cancer_type, dfg in all_df.groupby("cancer_type"):
        if dfg.shape[0] <= MIN_N_CT:
            continue
        genes = gene_cols(dfg)
        mut_rate = dfg[genes].sum() / dfg.shape[0]
        filtered_genes = mut_rate[mut_rate > MIN_MUTRATE].index.to_list()
        for gene in filtered_genes:
            data = dfg[dfg[gene].notna()]
            cph = CoxPHFitter()
            try:
                cph.fit(data, duration_col="time_to_cachexia", event_col="cachexia_event", formula=f"{gene}")
            except Exception:
                continue
            row = cph.summary.iloc[0, :].copy()
            row["detailed_cancer_type"] = FOLDER_MAP_REV.get(cancer_type, cancer_type)
            row["cancer_type_dfci"] = cancer_type
            row["mutation"] = gene
            row.name = f"{cancer_type}_{gene}"
            rows.append(row)

    cox_df = pd.DataFrame(rows)
    cox_df["p_adj"] = multipletests(cox_df["p"], method="fdr_bh")[1]
    out_fp = os.path.join(RESULTS_DIR, f"mutation_cox_cachexia_results_DFCI_{COMBINATION}_{THRESHOLD}_{DATE_STAMP}.csv")
    cox_df.to_csv(out_fp)
    print(f"[SAVED] {out_fp}  ({len(cox_df)} pairs, {(cox_df['p_adj'] < FDR_ALPHA).sum()} significant at FDR<{FDR_ALPHA})")
    return cox_df, all_df


def stage_cif(cox_df=None, all_df=None):
    if cox_df is None:
        in_fp = os.path.join(RESULTS_DIR, f"mutation_cox_cachexia_results_DFCI_{COMBINATION}_{THRESHOLD}_{DATE_STAMP}.csv")
        cox_df = pd.read_csv(in_fp, index_col=0)
    if all_df is None:
        all_df = load_all_dfci()

    sig = cox_df[cox_df["p_adj"] < FDR_ALPHA]
    WT_COL, MUT_COL = "#6388B4FF", "#EF6F6AFF"

    for _, r in sig.iterrows():
        gene = r["mutation"]
        cancer_type_dfci = r["cancer_type_dfci"]
        dfg = all_df[all_df["cancer_type"] == cancer_type_dfci]
        dfg = dfg[dfg["time"] < TIME_CUTOFF]

        ajf_wt = AalenJohansenFitter(calculate_variance=True)
        ajf_mut = AalenJohansenFitter(calculate_variance=True)
        ajf_wt.fit(durations=dfg[dfg[gene] == 0]["time"], event_observed=dfg[dfg[gene] == 0]["event"], event_of_interest=1)
        ajf_mut.fit(durations=dfg[dfg[gene] > 0]["time"], event_observed=dfg[dfg[gene] > 0]["event"], event_of_interest=1)

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
        ax.set_title(f"CIF in {FOLDER_MAP_REV.get(cancer_type_dfci, cancer_type_dfci)} (DFCI)")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Cumulative Incidence of cachexia")
        ax.legend(frameon=False)
        out_fp = os.path.join(CIF_DIR, f"cif_{gene}_{cancer_type_dfci}_{THRESHOLD}_DFCI.pdf")
        plt.savefig(out_fp)
        plt.close()
        print(f"[SAVED] {out_fp}")


def stage_multivariate(cox_df=None, all_df=None):
    if cox_df is None:
        in_fp = os.path.join(RESULTS_DIR, f"mutation_cox_cachexia_results_DFCI_{COMBINATION}_{THRESHOLD}_{DATE_STAMP}.csv")
        cox_df = pd.read_csv(in_fp, index_col=0)
    if all_df is None:
        all_df = load_all_dfci()

    out_root = os.path.join(RESULTS_DIR, f"multivariate_{THRESHOLD}", "cancer_types")
    os.makedirs(out_root, exist_ok=True)

    n_fit = 0
    for cancer_type_dfci, dfg in all_df.groupby("cancer_type"):
        sig_genes = cox_df.loc[
            (cox_df["cancer_type_dfci"] == cancer_type_dfci) & (cox_df["p_adj"] < FDR_ALPHA), "mutation"
        ].to_list()
        if not sig_genes:
            continue

        df = dfg[dfg["time_to_cachexia"] >= 0].copy() if "time_to_cachexia" in dfg.columns else dfg.copy()
        covars = [c for c in ["age_at_diagnosis_binned", "GENDER", "STAGE_CDM_DERIVED_GRANULAR",
                               "SAMPLE_TYPE", "ANCESTRY_LABEL", "CVR_TMB_SCORE", "start_BMI"]
                  if c in df.columns and df[c].notna().sum() > 0]
        variables = sig_genes + covars
        use_cols = variables + ["time_to_cachexia", "cachexia_event"]
        df = df.dropna(subset=[c for c in use_cols if c in df.columns])
        if df.shape[0] < 20:
            continue

        df_design = pd.get_dummies(df[variables], drop_first=True)
        for c in df_design.columns:
            if df_design[c].value_counts(normalize=True).iloc[0] > 0.99:
                df_design = df_design.drop(columns=[c])
        df_design["time_to_cachexia"] = df["time_to_cachexia"].values
        df_design["cachexia_event"] = df["cachexia_event"].values
        df_design = df_design.loc[:, ~df_design.columns.duplicated()]
        df_design = df_design.astype({c: float for c in df_design.columns if df_design[c].dtype == bool})

        cph = CoxPHFitter()
        try:
            cph.fit(df_design, duration_col="time_to_cachexia", event_col="cachexia_event")
        except Exception as e:
            print(f"[SKIP] {cancer_type_dfci}: {e}")
            continue

        out_df = cph.summary
        out_df["p_adj"] = multipletests(out_df["p"], method="fdr_bh")[1]
        out_fp = os.path.join(out_root, f"{cancer_type_dfci}_multivariate_{THRESHOLD}.csv")
        out_df.to_csv(out_fp)
        n_fit += 1
        print(f"[SAVED] {out_fp}  ({len(sig_genes)} sig genes, n={df.shape[0]})")

    print(f"\nWrote {n_fit} DFCI multivariate Cox fits to: {out_root}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert stage in ("univariate", "cif", "multivariate", "all")

    if stage in ("univariate", "all"):
        cox_df, all_df = stage_univariate()
    else:
        cox_df, all_df = None, None

    if stage in ("cif", "all"):
        stage_cif(cox_df, all_df)

    if stage in ("multivariate", "all"):
        stage_multivariate(cox_df, all_df)
