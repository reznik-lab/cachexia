#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:42:30 2026

@author: castilv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os
import pickle
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import AalenJohansenFitter
from lifelines.datasets import load_waltons
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib.gridspec as gridspec
from sklearn.utils import resample
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import roc_curve


if __name__ == "__main__":
    # ----------------------------Load Data--------------------------------
   
    PROJECT_ROOT = os.environ.get("CACHEXIA_PROJECT_ROOT", os.getcwd())

    REV_INPUTS = os.environ.get("CACHEXIA_INPUTS_DIR", os.path.join(PROJECT_ROOT, "data"))
    REV_RESULTS = os.environ.get("CACHEXIA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
    REV_MUTS = os.environ.get("CACHEXIA_MUTS_DIR", os.path.join(PROJECT_ROOT, "results_mutations"))

    DATE_STAMP = os.environ.get("CACHEXIA_DATE_STAMP", "YYYYMMDD")  # e.g., "20260126"
    WL_TAG = os.environ.get("CACHEXIA_WL_TAG", "5pct")  # e.g., "5pct", "10pct", "15pct"

    metadata_fp = os.environ.get(
        "CACHEXIA_METADATA_FP",
        os.path.join(REV_INPUTS, f"dx_cohort_metadata_{DATE_STAMP}.csv"),
    )
    metadata = pd.read_csv(metadata_fp, header=0)

    if "Tumor Diagnosis Date" not in metadata.columns and "anchor_final" in metadata.columns:
        metadata = metadata.rename(columns={"anchor_final": "Tumor Diagnosis Date"})

    bmi_fp = os.environ.get("CACHEXIA_BMI_FP", os.path.join(REV_INPUTS, "bmi.csv"))
    bmi = pd.read_csv(bmi_fp, header=0)

    bmi = bmi.merge(metadata[["MRN", "Tumor Diagnosis Date"]], on="MRN", how="left")
    bmi["Days_Since_Diagnosis"] = (pd.to_datetime(bmi["datetime"]) - pd.to_datetime(bmi["Tumor Diagnosis Date"])).dt.days
    bmi = bmi.dropna(subset=["Days_Since_Diagnosis"], axis=0)

    mut_run_dir = os.environ.get(
        "CACHEXIA_MUT_RUN_DIR",
        os.path.join(REV_MUTS, f"results_mutation_cac_frozen_{WL_TAG}"),
    )
    run_tag = os.environ.get("CACHEXIA_RUN_TAG", f"multivariate_{WL_TAG}_{DATE_STAMP}")
    results_dir = os.path.join(mut_run_dir, run_tag, "cancer_types")
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    metadata["PLA_LAST_CONTACT_DTE"] = pd.to_datetime(metadata["PLA_LAST_CONTACT_DTE"])
    metadata["Tumor Diagnosis Date"] = pd.to_datetime(metadata["Tumor Diagnosis Date"])
    metadata["os_days"] = (metadata["PLA_LAST_CONTACT_DTE"] - metadata["Tumor Diagnosis Date"]).dt.days
    metadata["os_event"] = 0
    metadata.loc[metadata["PT_DEATH_DTE"].notna(), "os_event"] = 1

    mutation_fp = os.environ.get("CACHEXIA_MUTATION_FP", os.path.join(REV_INPUTS, "oncogenic_table.csv"))
    mutation = pd.read_csv(mutation_fp, header=0)
    mutation = mutation.rename(columns={mutation.columns[0]: "DMP_ID"})
    mutation["DMP_ID"] = mutation["DMP_ID"].astype(str).str[:9]
    mutation = mutation.drop_duplicates(subset="DMP_ID", keep="first")
    for gene in mutation.columns[1:]:
        mutation.loc[mutation[gene] > 1, gene] = 1
    mutation.columns = [col.replace("-", "_") for col in mutation.columns]

    # ----------------------------Baseline Competing risk model--------------------------------
    cachexia_data_fp = os.environ.get(
        "CACHEXIA_CACHEXIA_DATA_FP",
        os.path.join(mut_run_dir, f"cachexia_data_survival_mutation_cac_frozen_{WL_TAG}.csv"),
    )
    cachexia_data = pd.read_csv(cachexia_data_fp)

    # --------------Preprocessing for the multivariate cox model--------------
    # calculate the age at diagnosis
    cachexia_data["Tumor Diagnosis Date"] = pd.to_datetime(cachexia_data["Tumor Diagnosis Date"])
    cachexia_data["PT_BIRTH_DTE"] = pd.to_datetime(cachexia_data["PT_BIRTH_DTE"])
    cachexia_data["age_at_diagnosis"] = (
        (cachexia_data["Tumor Diagnosis Date"] - cachexia_data["PT_BIRTH_DTE"]).dt.days / 3652.5
    )  
    cachexia_data = cachexia_data.dropna(
        subset=["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR"],
        axis=0,
    )

    cachexia_data.loc[cachexia_data["MSI_TYPE"] == "Stable", "MSI_TYPE"] = "AStable"

    # ----- add ECOG status data (baseline = closest START_DATE to 0)
    ecog_fp = os.environ.get("CACHEXIA_ECOG_FP", os.path.join(REV_INPUTS, "ecog_kps_timeline.tsv"))
    ecog = pd.read_csv(ecog_fp, sep="\t", header=0)

    ecog = ecog.sort_values(by=["PATIENT_ID", "START_DATE"])
    ecog.reset_index(inplace=True)

    ecog_baseline = ecog.groupby("PATIENT_ID", group_keys=False).apply(
        lambda x: x.loc[(x["START_DATE"]).abs().idxmin()]
    )
    ecog_baseline.rename(columns={"PATIENT_ID": "DMP_ID"}, inplace=True)

    cachexia_data = cachexia_data.merge(ecog_baseline[["DMP_ID", "ECOG_KPS"]], on="DMP_ID", how="inner")

    # ----- add start BMI (baseline = closest Days_Since_Diagnosis to 0)
    bmi_baseline = bmi.groupby("MRN", group_keys=False).apply(
        lambda x: x.loc[(x["Days_Since_Diagnosis"]).abs().idxmin()]
    )
    bmi_baseline.reset_index(inplace=True, drop=True)

    cachexia_data = cachexia_data.merge(bmi_baseline[["MRN", "bmi"]], on="MRN", how="inner")
    cachexia_data.rename(columns={"bmi": "start_BMI"}, inplace=True)
    cachexia_data["start_BMI"] = cachexia_data["start_BMI"] / 5  # scale by 5

    cachexia_data["start_BMI_cat"] = pd.cut(
        cachexia_data["start_BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    # -----------------------------Multivariate Cox Model: cachexia ~ multi genes--------------------------------

    uni_cox_fp = os.environ.get(
        "CACHEXIA_UNI_COX_FP",
        os.path.join(mut_run_dir, f"mutation_cox_cachexia_results_ccr_updated_{WL_TAG}.csv"),
    )
    uni_cox = pd.read_csv(uni_cox_fp, index_col=0)

    # Cancers to run (override with a comma-separated env var if desired)
    cancers_env = os.environ.get("CACHEXIA_CANCER_TYPES", None)
    if cancers_env is not None and len(cancers_env.strip()) > 0:
        cancer_types = [c.strip() for c in cancers_env.split(",")]
    else:
        cancer_types = [
            "Colorectal Adenocarcinoma",
            "Prostate Adenocarcinoma",
            "Lung Adenocarcinoma",
            "Breast Invasive Ductal Carcinoma",
            "Invasive Breast Carcinoma",
            "Renal Clear Cell Carcinoma",
            "Endometrial Carcinoma",
            "Cutaneous Melanoma",
            "Stomach Adenocarcinoma",
            "Uterine Endometrioid Carcinoma",
            "Breast Invasive Lobular Carcinoma",
            "Upper Tract Urothelial Carcinoma",
            "Bladder Urothelial Carcinoma",
        ]

    for cancer_type in cancer_types:
        df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]

        sig_genes = uni_cox.loc[
            (uni_cox["detailed_cancer_type"] == cancer_type) & (uni_cox["p_adj"] < 0.1), "mutation"
        ].to_list()

        if cancer_type in ["Colorectal Adenocarcinoma", "Stomach Adenocarcinoma", "Uterine Endometrioid Carcinoma"]:
            df = df.dropna(subset=["MSI_TYPE"], axis=0)

            if cancer_type == "Colorectal Adenocarcinoma":
                df = df.loc[df["Sidedness"].notna()]
                variables = sig_genes + [
                    "age_at_diagnosis",
                    "GENDER",
                    "SAMPLE_TYPE",
                    "MSI_TYPE",
                    "CVR_TMB_SCORE",
                    "STAGE_CDM_DERIVED_GRANULAR",
                    "ECOG_KPS",
                    "Sidedness",
                    "start_BMI",
                ]
            else:
                variables = sig_genes + [
                    "age_at_diagnosis",
                    "GENDER",
                    "SAMPLE_TYPE",
                    "MSI_TYPE",
                    "CVR_TMB_SCORE",
                    "STAGE_CDM_DERIVED_GRANULAR",
                    "ECOG_KPS",
                    "start_BMI",
                ]
        else:
            variables = sig_genes + [
                "age_at_diagnosis",
                "GENDER",
                "SAMPLE_TYPE",
                "CVR_TMB_SCORE",
                "STAGE_CDM_DERIVED_GRANULAR",
                "ECOG_KPS",
                "start_BMI",
            ]

        formula = " + ".join(variables)

        cph = CoxPHFitter()
        cph.fit(df, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

        cox_df = cph.summary
        cox_df["p_adj"] = multipletests(cox_df["p"], method="fdr_bh")[1]
        cox_df.to_csv(f"{results_dir}/{cancer_type.replace('/', '_')}.csv")