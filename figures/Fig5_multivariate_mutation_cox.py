#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariable Cox model (cachexia ~ significant genes + clinical covariates) per
cancer type, using the significant hits from Fig5_univariate_mutation_cox.py.

Adapted from 0303_ccx_revisions/rev_code/multivariate_test.py, parameterized
across weight-loss thresholds like the univariate script. Fixes a bug in the
original: the CRC "Sidedness" covariate was referenced in the model formula but
the code that merges Sidedness data in was commented out, so CRC's multivariate
fit would crash as originally written - now actually merges it in.

Usage: python Fig5_multivariate_mutation_cox.py <WL_LABEL>
  e.g. python Fig5_multivariate_mutation_cox.py WL5_BMIlt20

Requires Fig5_univariate_mutation_cox.py to have already been run for the same
WL_LABEL (reads its cachexia_data_survival_mutation_*.csv and
mutation_cox_cachexia_results_*.csv outputs).

Run with working directory set to the project root.
"""
import sys
import os
import pandas as pd
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter

WL_LABEL = sys.argv[1]
assert WL_LABEL in ("WL5_BMIlt20", "WL10", "WL15")

BASE_REV    = "."
REV_INPUTS  = os.path.join(BASE_REV, "rev_inputs")
REV_MUTS    = os.path.join(BASE_REV, "rev_muts")
DATE_STAMP  = "20260706"

mut_run_dir = os.path.join(REV_MUTS, f"results_mutation_{WL_LABEL}_{DATE_STAMP}")
results_dir = os.path.join(mut_run_dir, f"multivariate_{WL_LABEL}_{DATE_STAMP}", "cancer_types")
os.makedirs(results_dir, exist_ok=True)

metadata_fp = os.path.join(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
metadata = pd.read_csv(metadata_fp, header=0)
metadata = metadata.rename(columns={"anchor_final": "Tumor Diagnosis Date"})

bmi_fp = os.path.join(REV_INPUTS, "bmi_final_20260129.csv")
bmi = pd.read_csv(bmi_fp, header=0)
bmi = bmi.merge(metadata[["MRN", "Tumor Diagnosis Date"]], on="MRN", how="left")
bmi["Days_Since_Diagnosis"] = (pd.to_datetime(bmi["datetime"]) - pd.to_datetime(bmi["Tumor Diagnosis Date"])).dt.days
bmi = bmi.dropna(subset=["Days_Since_Diagnosis"], axis=0)

cachexia_data = pd.read_csv(
    os.path.join(mut_run_dir, f"cachexia_data_survival_mutation_{WL_LABEL}_{DATE_STAMP}.csv")
)

cachexia_data["Tumor Diagnosis Date"] = pd.to_datetime(cachexia_data["Tumor Diagnosis Date"])
cachexia_data["PT_BIRTH_DTE"] = pd.to_datetime(cachexia_data["PT_BIRTH_DTE"])
cachexia_data["age_at_diagnosis"] = (cachexia_data["Tumor Diagnosis Date"] - cachexia_data["PT_BIRTH_DTE"]).dt.days / 3652.5
cachexia_data = cachexia_data.dropna(
    subset=["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR"], axis=0
)
cachexia_data.loc[cachexia_data["MSI_TYPE"] == "Stable", "MSI_TYPE"] = "AStable"

# ---- ECOG baseline ----
ecog = pd.read_csv(os.path.join(REV_INPUTS, "data_timeline_ecog_kps.txt"), sep="\t", header=0)
ecog = ecog.sort_values(by=["PATIENT_ID", "START_DATE"])
ecog.reset_index(inplace=True)
ecog_baseline = ecog.groupby("PATIENT_ID", group_keys=False).apply(
    lambda x: x.loc[(x["START_DATE"]).abs().idxmin()]
)
ecog_baseline.rename(columns={"PATIENT_ID": "DMP_ID"}, inplace=True)
cachexia_data = cachexia_data.merge(ecog_baseline[["DMP_ID", "ECOG_KPS"]], on="DMP_ID", how="inner")

# ---- CRC left/right (Sidedness) - fixed: actually merged in now.
# The clinical metadata already carries its own Sidedness field (from the
# univariate script's metadata merge); coalesce with the dedicated CRC file,
# preferring the dedicated file since it's the more specific/curated source.
left_right_tumor = pd.read_csv(os.path.join(REV_INPUTS, "CRC_Primary_Tumor_Location_01_24_25.csv"), encoding="utf-8-sig")
left_right_tumor = left_right_tumor.rename(columns={"PATIENT_ID": "DMP_ID", "Sidedness": "Sidedness_crcfile"})
cachexia_data = cachexia_data.merge(left_right_tumor[["DMP_ID", "Sidedness_crcfile"]], on="DMP_ID", how="left")
cachexia_data["Sidedness"] = cachexia_data["Sidedness_crcfile"].fillna(cachexia_data["Sidedness"])
cachexia_data = cachexia_data.drop(columns=["Sidedness_crcfile"])

# ---- start BMI ----
bmi_baseline = bmi.groupby("MRN", group_keys=False).apply(
    lambda x: x.loc[(x["Days_Since_Diagnosis"]).abs().idxmin()]
)
bmi_baseline.reset_index(inplace=True, drop=True)
cachexia_data = cachexia_data.merge(bmi_baseline[["MRN", "bmi"]], on="MRN", how="inner")
cachexia_data.rename(columns={"bmi": "start_BMI"}, inplace=True)
cachexia_data["start_BMI"] = cachexia_data["start_BMI"] / 5
cachexia_data["start_BMI_cat"] = pd.cut(
    cachexia_data["start_BMI"], bins=[0, 18.5, 25, 30, 100], labels=["underweight", "normal", "overweight", "obese"]
)

# ---- multivariate Cox per cancer type ----
uni_cox = pd.read_csv(
    os.path.join(mut_run_dir, f"mutation_cox_cachexia_results_{WL_LABEL}_{DATE_STAMP}.csv"), index_col=0
)

CANCER_TYPES = [
    "Colorectal Adenocarcinoma", "Prostate Adenocarcinoma", "Lung Adenocarcinoma",
    "Breast Invasive Ductal Carcinoma", "Invasive Breast Carcinoma", "Renal Clear Cell Carcinoma",
    "Endometrial Carcinoma", "Cutaneous Melanoma", "Stomach Adenocarcinoma",
    "Uterine Endometrioid Carcinoma", "Breast Invasive Lobular Carcinoma",
    "Upper Tract Urothelial Carcinoma", "Bladder Urothelial Carcinoma",
]

n_fit = 0
for cancer_type in CANCER_TYPES:
    df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]
    if df.empty:
        continue
    sig_genes = uni_cox.loc[
        (uni_cox["detailed_cancer_type"] == cancer_type) & (uni_cox["p_adj"] < 0.1), "mutation"
    ].to_list()

    if cancer_type in ("Colorectal Adenocarcinoma", "Stomach Adenocarcinoma", "Uterine Endometrioid Carcinoma"):
        df = df.dropna(subset=["MSI_TYPE"], axis=0)
        if cancer_type == "Colorectal Adenocarcinoma":
            df = df.loc[df["Sidedness"].notna()]
            variables = sig_genes + ["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "MSI_TYPE",
                                      "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS",
                                      "Sidedness", "start_BMI"]
        else:
            variables = sig_genes + ["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "MSI_TYPE",
                                      "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", "start_BMI"]
    else:
        variables = sig_genes + ["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "CVR_TMB_SCORE",
                                  "STAGE_CDM_DERIVED_GRANULAR", "ECOG_KPS", "start_BMI"]

    if not sig_genes or df.shape[0] < 20:
        continue

    formula = " + ".join(variables)
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
    except Exception as e:
        print(f"[SKIP] {cancer_type}: {e}")
        continue

    cox_df = cph.summary
    cox_df["p_adj"] = multipletests(cox_df["p"], method="fdr_bh")[1]
    out_fp = f"{results_dir}/{cancer_type.replace('/', '_')}.csv"
    cox_df.to_csv(out_fp)
    n_fit += 1
    print(f"[SAVED] {out_fp}  ({len(sig_genes)} sig genes, n={df.shape[0]})")

print(f"\n[{WL_LABEL}] Wrote {n_fit} multivariate Cox fits to: {results_dir}")
