

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
# from scripts.episodes.utils import plot_BMI, plot_BMI_treatment
import matplotlib.gridspec as gridspec
from sklearn.utils import resample
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import roc_curve


if __name__ == "__main__":
   

    PROJECT_ROOT = os.environ.get("CACHEXIA_PROJECT_ROOT", os.getcwd())
    REV_INPUTS = os.environ.get("CACHEXIA_INPUTS_DIR", os.path.join(PROJECT_ROOT, "data"))
    REV_RESULTS = os.environ.get("CACHEXIA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
    REV_MUTS = os.environ.get("CACHEXIA_MUTS_DIR", os.path.join(PROJECT_ROOT, "results_mutations"))

    DATE_STAMP = os.environ.get("CACHEXIA_DATE_STAMP", "YYYYMMDD")  # e.g., "20260126"
    WL_TAG = os.environ.get("CACHEXIA_WL_TAG", "5pct")  # e.g., "5pct", "10pct", "15pct"

    mut_run_dir = os.path.join(REV_MUTS, f"results_mutation_cac_frozen_{WL_TAG}")

    NOMO_TAG = os.environ.get("CACHEXIA_NOMO_TAG", f"nomogram_{WL_TAG}_{DATE_STAMP}")
    results_dir = os.path.join(mut_run_dir, NOMO_TAG)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

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
    bmi = bmi.dropna(subset=["Days_Since_Diagnosis"]).copy()

    # recalculate overall survival and os_event
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
    cachexia_data["Tumor Diagnosis Date"] = pd.to_datetime(cachexia_data["Tumor Diagnosis Date"])
    cachexia_data["PT_BIRTH_DTE"] = pd.to_datetime(cachexia_data["PT_BIRTH_DTE"])
    cachexia_data["age_at_diagnosis"] = (
        (cachexia_data["Tumor Diagnosis Date"] - cachexia_data["PT_BIRTH_DTE"]).dt.days / 3652.5
    )  # scale by 10 years

    cachexia_data = cachexia_data.dropna(
        subset=["age_at_diagnosis", "GENDER", "SAMPLE_TYPE", "CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR"], axis=0
    )
    cachexia_data = cachexia_data.dropna(subset=["STAGE_CDM_DERIVED_GRANULAR"], axis=0)
    cachexia_data.loc[cachexia_data["MSI_TYPE"] == "Stable", "MSI_TYPE"] = "AStable"

    # ----- add ECOG status data
    ecog_fp = os.environ.get("CACHEXIA_ECOG_FP", os.path.join(REV_INPUTS, "ecog_kps_timeline.tsv"))
    ecog = pd.read_csv(ecog_fp, sep="\t", header=0)
    ecog = ecog.sort_values(by=["PATIENT_ID", "START_DATE"])
    ecog.reset_index(inplace=True)
    ecog_baseline = ecog.groupby("PATIENT_ID", group_keys=False).apply(
        lambda x: x.loc[(x["START_DATE"]).abs().idxmin()]
    )
    ecog_baseline.rename(columns={"PATIENT_ID": "DMP_ID"}, inplace=True)
    cachexia_data = cachexia_data.merge(ecog_baseline[["DMP_ID", "ECOG_KPS"]], on="DMP_ID", how="inner")

    # ----- add start BMI
    bmi_baseline = bmi.groupby("MRN", group_keys=False).apply(
        lambda x: x.loc[(x["Days_Since_Diagnosis"]).abs().idxmin()]
    )
    bmi_baseline.reset_index(inplace=True, drop=True)
    cachexia_data = cachexia_data.merge(bmi_baseline[["MRN", "bmi"]], on="MRN", how="inner")
    cachexia_data.rename(columns={"bmi": "start_BMI"}, inplace=True)
    cachexia_data["start_BMI"] = cachexia_data["start_BMI"] / 5  # scale by 5
    cachexia_data["start_BMI_cat"] = pd.cut(
        cachexia_data["start_BMI"], bins=[0, 18.5, 25, 30, 100], labels=["underweight", "normal", "overweight", "obese"]
    )

    # -----------------------------Build nomogram--------------------------------
    cancer_type = os.environ.get("CACHEXIA_CANCER_TYPE", "Pancreatic Adenocarcinoma")
    df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type].copy()
    out_prefix = f"{cancer_type.replace('/','_').replace(' ','_')}_{WL_TAG}"

    mv_dir = os.path.join(mut_run_dir, f"multivariate_{WL_TAG}_{DATE_STAMP}", "cancer_types")
    mv_fp = os.environ.get("CACHEXIA_MV_FP", os.path.join(mv_dir, f"{cancer_type.replace('/','_')}.csv"))
    mv = pd.read_csv(mv_fp, index_col=0)  # covariate names are the index

    SIG_THR = float(os.environ.get("CACHEXIA_SIG_THR", "0.1"))
    sig_covs = mv.loc[mv["p_adj"] < SIG_THR].index.astype(str).tolist()
    variables = [v for v in sig_covs if v in df.columns]

    df = df.loc[df["time_to_cachexia"] > 0].copy()
    df = df.dropna(subset=variables)

    formula = " + ".join(variables)

    print("Using variables:", variables)
    print("Formula:", formula)

    cph = CoxPHFitter()
    cph.fit(df, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
    cox_df = cph.summary

    # -----------------------------Bootstrapping and validation (C-index)-----------------------------
    n_bootstrap = int(os.environ.get("CACHEXIA_BOOTSTRAP_N", "200"))
    c_indices = []

    for i in range(n_bootstrap):
        df_resampled = resample(df, replace=True, n_samples=len(df), random_state=i)
        df_resampled.reset_index(drop=True, inplace=True)

        c_index = concordance_index(
            df_resampled["time_to_cachexia"],
            -cph.predict_partial_hazard(df_resampled),
            df_resampled["cachexia_event"],
        )
        c_indices.append(c_index)

    c_index_mean = float(np.mean(c_indices))
    c_index_ci = (float(np.percentile(c_indices, 2.5)), float(np.percentile(c_indices, 97.5)))
    print(c_index_mean, c_index_ci)

    # --------------- Calibration Plot (50/50 split) ---------------
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)

    cph_split = CoxPHFitter()
    cph_split.fit(df_train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

    surv = cph_split.predict_survival_function(df_test)  # rows=time, cols=patients
    times = surv.index.to_numpy(float)

    targets = (365, 730, 1095)
    sel = {t: float(times[np.abs(times - t).argmin()]) for t in targets if len(times) and t <= times.max()}

    print("Has exact targets:", {t: (t in surv.index) for t in targets})
    print("Using nearest days:", sel)

    if not sel:
        raise ValueError(
            f"No evaluation horizons available; max follow-up is {float(times.max()) if len(times) else 'N/A'} days."
        )

    for t, t_near in sel.items():
        df_test[f"predicted_cachexia_{t}d"] = 1 - surv.loc[t_near].values

    pivot_t = 365 if 365 in sel else min(sel.keys())
    df_test["risk_bin"] = pd.qcut(df_test[f"predicted_cachexia_{pivot_t}d"], q=4, labels=False, duplicates="drop")

    time_points = sorted(sel.keys())

    # re-bin using 365d if present
    df_test["risk_bin"] = pd.qcut(df_test["predicted_cachexia_365d"], q=4, labels=False, duplicates="drop")

    kmf = KaplanMeierFitter()
    observed_cachexia = {t: [] for t in time_points}
    ci_lower, ci_upper = {t: [] for t in time_points}, {t: [] for t in time_points}

    n_bootstrap_cal = int(os.environ.get("CACHEXIA_CAL_BOOTSTRAP_N", "200"))

    for group in sorted(df_test["risk_bin"].unique()):
        group_data = df_test[df_test["risk_bin"] == group]

        for t in time_points:
            kmf.fit(group_data["time_to_cachexia"], event_observed=group_data["cachexia_event"])
            observed_risk = 1 - kmf.predict(t)
            observed_cachexia[t].append(observed_risk)

            bootstrap_estimates = []
            for j in range(n_bootstrap_cal):
                resampled_data = resample(group_data, replace=True, n_samples=len(group_data), random_state=j)
                kmf.fit(resampled_data["time_to_cachexia"], event_observed=resampled_data["cachexia_event"])
                bootstrap_estimates.append(1 - kmf.predict(t))

            ci_lower[t].append(np.percentile(bootstrap_estimates, 2.5))
            ci_upper[t].append(np.percentile(bootstrap_estimates, 97.5))

    df_calibration = {}
    for t in time_points:
        df_calibration[t] = pd.DataFrame(
            {
                "Predicted Cachexia Risk": df_test.groupby("risk_bin")[f"predicted_cachexia_{t}d"].mean(),
                "Observed Cachexia Risk": observed_cachexia[t],
                "CI Lower": ci_lower[t],
                "CI Upper": ci_upper[t],
            }
        )

    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 9

    plt.figure(figsize=(3.5, 3.3))

    markers = {365: "o", 730: "s", 1095: "^"}
    colors = {365: "#6388B4FF", 730: "#FFAE34FF", 1095: "#EF6F6AFF"}

    for t in time_points:
        plt.errorbar(
            df_calibration[t]["Predicted Cachexia Risk"],
            df_calibration[t]["Observed Cachexia Risk"],
            yerr=[
                df_calibration[t]["Observed Cachexia Risk"] - df_calibration[t]["CI Lower"],
                df_calibration[t]["CI Upper"] - df_calibration[t]["Observed Cachexia Risk"],
            ],
            fmt=markers[t],
            label=f"{round(t/365)}-year cachexia incidence",
            color=colors[t],
            capsize=3,
        )

    plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration", color="black")
    plt.xlabel("Nomogram-Predicted Probability of Cachexia")
    plt.ylabel("Cumulative Incidence of Cachexia")
    plt.title(f"Calibration Plot for {cancer_type}")
    plt.legend(fontsize=7, markerscale=1, frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", f"calibration_half_split_{out_prefix}.pdf"))
    plt.close()

    # ----------------------------- AUC (50/50 split) -----------------------------#
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)

    cph_split = CoxPHFitter()
    cph_split.fit(df_train, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)

    surv = cph_split.predict_survival_function(df_test)
    times = surv.index.to_numpy(float)

    targets = (365, 730, 1095)
    sel = {t: float(times[np.abs(times - t).argmin()]) for t in targets if len(times) and t <= times.max()}

    print("Has exact targets:", {t: (t in surv.index) for t in targets})
    print("Using nearest days:", sel)

    if not sel:
        raise ValueError(
            f"No evaluation horizons available; max follow-up is {float(times.max()) if len(times) else 'N/A'} days."
        )

    for t, t_near in sel.items():
        df_test[f"predicted_cachexia_{t}d"] = 1 - surv.loc[t_near].values

    pivot_t = 365 if 365 in sel else min(sel.keys())
    df_test["risk_bin"] = pd.qcut(df_test[f"predicted_cachexia_{pivot_t}d"], q=4, labels=False, duplicates="drop")

    time_points = sorted(sel.keys())

    survival_train = Surv.from_dataframe(event="cachexia_event", time="time_to_cachexia", data=df_train)
    survival_test = Surv.from_dataframe(event="cachexia_event", time="time_to_cachexia", data=df_test)

    auc_values = {}

    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 9

    plt.figure(figsize=(2.8, 2.8))

    colors = {365: "#6388B4FF", 730: "#FFAE34FF", 1095: "#EF6F6AFF"}

    for t in time_points:
        auc_values[t], _ = cumulative_dynamic_auc(survival_train, survival_test, df_test[f"predicted_cachexia_{t}d"], [t])

        df_test[f"cachexia_event_{t}d"] = (df_test["cachexia_event"] & (df_test["time_to_cachexia"] <= t)).astype(int)

        fpr, tpr, _ = roc_curve(df_test[f"cachexia_event_{t}d"], df_test[f"predicted_cachexia_{t}d"])
        plt.plot(fpr, tpr, color=colors[t], label=f"{round(t/365)}-Year AUC: {auc_values[t][0]:.3f}")

    plt.plot([0, 1], [0, 1], "--", color="black", label="Random Chance")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"Time-Dependent ROC: {cancer_type}")

    plt.legend(fontsize=7, markerscale=1, frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", f"auc_half_split_{out_prefix}.pdf"))
    plt.close()

    for t in time_points:
        print(f"AUC at {round(t/365)} year(s): {auc_values[t][0]:.3f}")