
import pandas as pd
import numpy as np
import seaborn as sns
import os
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


if __name__ == "__main__":
    # ----------------------------Load Data--------------------------------

    PROJECT_ROOT = os.environ.get("CACHEXIA_PROJECT_ROOT", os.getcwd())

    REV_INPUTS = os.environ.get("CACHEXIA_INPUTS_DIR", os.path.join(PROJECT_ROOT, "data"))
    REV_RESULTS = os.environ.get("CACHEXIA_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
    REV_MUTS = os.environ.get("CACHEXIA_MUTS_DIR", os.path.join(PROJECT_ROOT, "results_mutations"))

    DATE_STAMP = os.environ.get("CACHEXIA_DATE_STAMP", "YYYYMMDD") 

    # Inputs
    bmi_fp = os.environ.get("CACHEXIA_BMI_FP", os.path.join(REV_INPUTS, "bmi.csv"))
    episodes_fp = os.environ.get(
        "CACHEXIA_EPISODES_FP",
        os.path.join(REV_RESULTS, "episode_summary.csv"),
    )
    metadata_fp = os.environ.get(
        "CACHEXIA_METADATA_FP",
        os.path.join(REV_INPUTS, f"dx_cohort_metadata_{DATE_STAMP}.csv"),
    )
    mutation_fp = os.environ.get(
        "CACHEXIA_MUTATION_FP",
        os.path.join(REV_INPUTS, "oncogenic_table.csv"),
    )

    bmi = pd.read_csv(bmi_fp, header=0)

    # episode table
    cac_frozen_episodes = pd.read_csv(episodes_fp)
    cac_frozen_episodes["episode_duration"] = cac_frozen_episodes["end_day"] - cac_frozen_episodes["start_day"]
    cac_frozen_episodes["weight_loss"] = (
        (cac_frozen_episodes["start_bmi"] - cac_frozen_episodes["end_bmi"]) / cac_frozen_episodes["start_bmi"] * 100
    )

    metadata = pd.read_csv(metadata_fp, header=0)
    if "Tumor Diagnosis Date" not in metadata.columns and "anchor_final" in metadata.columns:
        metadata = metadata.rename(columns={"anchor_final": "Tumor Diagnosis Date"})

    # Outputs (everything under REV_MUTS)
    results_dir = os.path.join(REV_MUTS, "results_mutation_cac_frozen_5pct")
    try:
        os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    except FileExistsError:
        print("Directory plots already exists.")

    # ----------------------------Overall survival and os_event--------------------------------
    metadata["PLA_LAST_CONTACT_DTE"] = pd.to_datetime(metadata["PLA_LAST_CONTACT_DTE"])
    metadata["Tumor Diagnosis Date"] = pd.to_datetime(metadata["Tumor Diagnosis Date"])
    metadata["os_days"] = (metadata["PLA_LAST_CONTACT_DTE"] - metadata["Tumor Diagnosis Date"]).dt.days
    metadata["os_event"] = 0
    metadata.loc[metadata["PT_DEATH_DTE"].notna(), "os_event"] = 1

    # ----------------------------Mutation data--------------------------------
    mutation = pd.read_csv(mutation_fp, header=0)
    mutation = mutation.rename(columns={mutation.columns[0]: "DMP_ID"})
    mutation["DMP_ID"] = mutation["DMP_ID"].astype(str).str[:9]
    mutation = mutation.drop_duplicates(subset="DMP_ID", keep="first")

    for gene in mutation.columns[1:]:
        mutation.loc[mutation[gene] > 1, gene] = 1
    mutation.columns = [col.replace("-", "_") for col in mutation.columns]

    # ----------------------------Baseline Competing risk model--------------------------------
    cac_frozen_episodes.loc[
        (cac_frozen_episodes["start_day"].notna()) & (cac_frozen_episodes["episode_duration"] < 30),
        "start_day",
    ] = None

    cac_frozen_episodes.loc[
        (cac_frozen_episodes["start_day"].notna()) & (cac_frozen_episodes["weight_loss"] < 5),
        "start_day",
    ] = None

    cachexia_data = cac_frozen_episodes.groupby("MRN", as_index=False).apply(
        lambda x: x.loc[x["start_day"].idxmin()] if x["start_day"].notna().any() else x.iloc[0]
    ).reset_index(drop=True)

    cachexia_data["cachexia_event"] = cachexia_data["start_day"].notna().astype(int)
    cachexia_data = cachexia_data[["start_day", "MRN", "cachexia_event"]]
    cachexia_data = cachexia_data.rename(columns={"start_day": "time_to_cachexia"})

    # merge cachexia_data with metadata to get os_days and os_event
    cachexia_data = cachexia_data.merge(metadata, on="MRN", how="left")
    cachexia_data = cachexia_data.dropna(subset=["os_days", "os_event"])

    # if cachexia_event is 0, set it to os_days
    cachexia_data.loc[cachexia_data["cachexia_event"] == 0, "time_to_cachexia"] = cachexia_data["os_days"]

    # define multi-events
    cachexia_data["event"] = 0
    cachexia_data.loc[cachexia_data["os_event"] == 1, "event"] = 2
    cachexia_data.loc[cachexia_data["cachexia_event"] == 1, "event"] = 1

    # create a new time column
    cachexia_data["time"] = cachexia_data["os_days"]
    cachexia_data.loc[cachexia_data["cachexia_event"] == 1, "time"] = cachexia_data["time_to_cachexia"]

    # merge with mutation data
    cachexia_data = cachexia_data.merge(mutation, on="DMP_ID", how="left")
    cachexia_data = cachexia_data[cachexia_data["AR"].notna()]

    # drop the rows with mismatching os_days
    cachexia_data = cachexia_data.loc[cachexia_data["time_to_cachexia"] >= 0]

    cachexia_data = cachexia_data.loc[cachexia_data["time_to_cachexia"] < 4000]

    # merge colon/rectal into colorectal
    cachexia_data.loc[
        cachexia_data["CANCER_TYPE_DETAILED"] == "Colon Adenocarcinoma", "CANCER_TYPE_DETAILED"
    ] = "Colorectal Adenocarcinoma"
    cachexia_data.loc[
        cachexia_data["CANCER_TYPE_DETAILED"] == "Rectal Adenocarcinoma", "CANCER_TYPE_DETAILED"
    ] = "Colorectal Adenocarcinoma"

    cachexia_data.to_csv(
        os.path.join(results_dir, "cachexia_data_survival_mutation_cac_frozen_5pct.csv"),
        index=False,
    )

    cachexia_data = pd.read_csv(
        os.path.join(results_dir, "cachexia_data_survival_mutation_cac_frozen_5pct.csv"),
        header=0,
    )

    # -----------------------------Cox Proportional Hazard Model: cachexia--------------------------------
    cancer_counts = cachexia_data["CANCER_TYPE_DETAILED"].value_counts().reset_index()
    all_genes = mutation.columns[1:]

    cox_df = pd.DataFrame(
        {
            "coef": [],
            "exp(coef)": [],
            "se(coef)": [],
            "coef lower 95%": [],
            "coef upper 95%": [],
            "exp(coef) lower 95%": [],
            "exp(coef) upper 95%": [],
            "cmp to": [],
            "z": [],
            "p": [],
            "-log2(p)": [],
        }
    )

    for cancer_type in cancer_counts.loc[cancer_counts["count"] > 200, "CANCER_TYPE_DETAILED"]:
        df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]

        mutation_rate = df[all_genes].sum() / df.shape[0]
        filtered_genes = mutation_rate[mutation_rate > 0.05].index.to_list()

        if len(filtered_genes) == 0:
            continue
        else:
            for gene in filtered_genes:
                data = df[df[gene].notna()]
                cph = CoxPHFitter()
                cph.fit(
                    data,
                    duration_col="time_to_cachexia",
                    event_col="cachexia_event",
                    formula=f"{gene}",
                )
                cox_df.loc[f"{cancer_type}_{gene}"] = cph.summary.iloc[0, :]
                cox_df.loc[f"{cancer_type}_{gene}", "detailed_cancer_type"] = cancer_type
                cox_df.loc[f"{cancer_type}_{gene}", "mutation"] = gene

    cox_df["p_adj"] = multipletests(cox_df["p"], method="fdr_bh")[1]
    cox_df.to_csv(os.path.join(results_dir, "mutation_cox_cachexia_results_ccr_updated_5pct.csv"))

    # -----------------------------Dosage effects validation--------------------------------
    cox_df_5 = pd.read_csv(
        os.path.join(REV_MUTS, "results_mutation_cac_frozen_5pct", "mutation_cox_cachexia_results_ccr_updated.csv"),
        index_col=0,
    )
    cox_df_10 = pd.read_csv(
        os.path.join(REV_MUTS, "results_mutation_cac_frozen_10pct", "mutation_cox_cachexia_results_ccr_updated_10pct.csv"),
        index_col=0,
    )
    cox_df_15 = pd.read_csv(
        os.path.join(REV_MUTS, "results_mutation_cac_frozen_15pct", "mutation_cox_cachexia_results_ccr_updated_15pct.csv"),
        index_col=0,
    )

    sig_indices = cox_df_5[cox_df_5["p_adj"] < 0.1].index.to_list()
    cox_df_10 = cox_df_10.loc[sig_indices]
    cox_df_15 = cox_df_15.loc[sig_indices]
    print(cox_df_10.loc[cox_df_10["p"] < 0.05].shape[0])  # 24
    print(cox_df_15.loc[cox_df_15["p"] < 0.05].shape[0])  # 13

    cox_df = pd.read_csv(os.path.join(results_dir, "mutation_cox_cachexia_results_ccr_updated.csv"), index_col=0)

    # --------------------------------plot volcano plot for cachexia--------------------------------
    cox_df_cachexia = cox_df.copy()
    cox_df_cachexia["-log2(p_adj)"] = -np.log2(cox_df_cachexia["p_adj"])
    cox_df_cachexia["significant"] = "Not significant"
    cox_df_cachexia.loc[cox_df_cachexia["p_adj"] < 0.1, "significant"] = "Significant"
    cox_df_cachexia = cox_df_cachexia[(cox_df_cachexia["coef"] > -14) & (cox_df_cachexia["coef"] < 14)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cox_df_cachexia, x="coef", y="-log2(p_adj)", hue="significant")
    plt.axhline(y=-np.log2(0.1), color="r", linestyle="--")
    plt.axvline(x=0, color="r", linestyle="--")

    for i in range(cox_df_cachexia[cox_df_cachexia["p_adj"] < 0.1].shape[0]):
        plt.annotate(
            cox_df_cachexia[cox_df_cachexia["p_adj"] < 0.1].index[i],
            (
                cox_df_cachexia[cox_df_cachexia["p_adj"] < 0.1].iloc[i]["coef"],
                cox_df_cachexia[cox_df_cachexia["p_adj"] < 0.1].iloc[i]["-log2(p_adj)"],
            ),
            fontsize=5,
        )

    plt.xlabel("log(Hazard Ratio)")
    plt.ylabel("-log2(p_adj)")
    plt.title("Volcano plot for mutation in cachexia")
    plt.savefig(os.path.join(results_dir, "plots", "volcano_plot_ccr_updated_5ct.pdf"))
    plt.close()

    # --------------------------------for significant genes, plot the CIF curves--------------------------------
    try:
        os.makedirs(os.path.join(results_dir, "plots_cif"), exist_ok=True)
    except FileExistsError:
        print("Directory plots already exists.")

    cox_df_significant = cox_df_cachexia[cox_df_cachexia["p_adj"] < 0.1]

    for i in range(cox_df_significant.shape[0]):
        gene = cox_df_significant.iloc[i]["mutation"]
        cancer_type = cox_df_significant.iloc[i]["detailed_cancer_type"]
        df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type]

        ajf_wt = AalenJohansenFitter(calculate_variance=True)
        ajf_mut = AalenJohansenFitter(calculate_variance=True)

        ajf_wt.fit(
            durations=df[df[gene] == 0]["time"],
            event_observed=df[df[gene] == 0]["event"],
            event_of_interest=1,
        )

        ajf_mut.fit(
            durations=df[df[gene] > 0]["time"],
            event_observed=df[df[gene] > 0]["event"],
            event_of_interest=1,
        )

        fig, ax = plt.subplots(figsize=(3, 2.6))

        ajf_wt.plot(ax=ax, label=f"WT {gene}", ci_show=True)
        ajf_mut.plot(ax=ax, label=f"MUT {gene}", ci_show=True)

        WT_COL  = "#6388B4FF"
        MUT_COL = "#EF6F6AFF"

        lines = ax.get_lines()
        if len(lines) >= 2:
            lines[0].set_color(WT_COL)
            lines[1].set_color(MUT_COL)
            lines[0].set_linewidth(2.0)
            lines[1].set_linewidth(2.0)

        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        if len(polys) >= 2:
            polys[0].set_facecolor(WT_COL)
            polys[1].set_facecolor(MUT_COL)
            polys[0].set_alpha(0.18)
            polys[1].set_alpha(0.18)
            polys[0].set_edgecolor("none")
            polys[1].set_edgecolor("none")

        ax.grid(False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.3)

        ax.tick_params(axis="both", which="both",
               top=False, right=False,
               width=0.3, length=3)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0, y=0)  # removes the default padding

        ax.set_title(f"CIF in {cancer_type}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Cumulative Incidence of cachexia")
        ax.legend(frameon=False)
        plt.savefig(os.path.join(results_dir, "plots_cif", f"cif_{gene}_{cancer_type}_5pct.pdf"))
        plt.close()