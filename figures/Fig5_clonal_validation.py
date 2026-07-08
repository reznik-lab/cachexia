import os
import pandas as pd
from lifelines import CoxPHFitter

BASE_REV   = "."
REV_INPUTS = os.path.join(BASE_REV, "rev_inputs")
REV_MUTS   = os.path.join(BASE_REV, "rev_muts")
DATE_STAMP = "20260706"
WL_LABEL   = "WL5_BMIlt20"

facets_fp = os.path.join(REV_INPUTS, "impact_facets_annotated.ccf.maf")
facets = pd.read_csv(facets_fp, sep="\t", header=0, low_memory=False)

facets = facets.loc[(facets["oncogenic"] == "Oncogenic") | (facets["oncogenic"] == "Likely Oncogenic")]
facets["clonality_new"] = (facets["clonality"] == "CLONAL").astype(int)
table = facets.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"])["clonality_new"].max().reset_index()
clonality = table.pivot(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="clonality_new").fillna(0)
clonality.index = clonality.index.str[:9]
clonality = clonality[~clonality.index.duplicated(keep="first")]

mut_run_dir = os.path.join(REV_MUTS, f"results_mutation_{WL_LABEL}_{DATE_STAMP}")
cachexia_fp = os.path.join(mut_run_dir, f"cachexia_data_survival_mutation_{WL_LABEL}_{DATE_STAMP}.csv")
cox_fp = os.path.join(mut_run_dir, f"mutation_cox_cachexia_results_{WL_LABEL}_{DATE_STAMP}.csv")

cachexia_data = pd.read_csv(cachexia_fp, low_memory=False)
cox_df = pd.read_csv(cox_fp, index_col=0)

significant_mutations = cox_df[cox_df["p_adj"] < 0.1][["detailed_cancer_type", "mutation"]]
print(f"Rechecking {len(significant_mutations)} significant discovery-hit pairs using clonal-only calls")

rows = []
for _, row in significant_mutations.iterrows():
    cancer_type = row["detailed_cancer_type"]
    gene = row["mutation"]
    if gene not in clonality.columns:
        print(f"[SKIP] {cancer_type}/{gene}: gene not found in clonality matrix")
        continue

    df = cachexia_data[cachexia_data["CANCER_TYPE_DETAILED"] == cancer_type].copy()
    df[gene] = df["DMP_ID"].map(clonality[gene]).fillna(0)

    if df[gene].sum() == 0:
        print(f"[SKIP] {cancer_type}/{gene}: no clonal-mutant cases")
        continue

    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time_to_cachexia", event_col="cachexia_event", formula=gene)
        s = cph.summary.iloc[0]
        rows.append({
            "detailed_cancer_type": cancer_type, "mutation": gene,
            "coef": s["coef"], "exp(coef)": s["exp(coef)"], "se(coef)": s["se(coef)"],
            "coef lower 95%": s["coef lower 95%"], "coef upper 95%": s["coef upper 95%"],
            "exp(coef) lower 95%": s["exp(coef) lower 95%"], "exp(coef) upper 95%": s["exp(coef) upper 95%"],
            "z": s["z"], "p": s["p"], "n_clonal_mutant": int(df[gene].sum()),
        })
    except Exception as e:
        print(f"[FAIL] {cancer_type}/{gene}: {e}")

out = pd.DataFrame(rows)
out["sig_clonal_p05"] = out["p"] < 0.05

out_dir = os.path.join(BASE_REV, "rev_SI", "tables")
os.makedirs(out_dir, exist_ok=True)
out_fp = os.path.join(out_dir, f"STable15_clonal_validation_{DATE_STAMP}.csv")
out.to_csv(out_fp, index=False)

n_sig = out["sig_clonal_p05"].sum()
n_total = len(significant_mutations)
print(f"\n{n_sig} / {n_total} significant discovery-hit pairs remain significant (p<0.05) using clonal-only mutation calls")
print(f"[SAVED] {out_fp}")
