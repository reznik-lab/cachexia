import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

BASE_REV = "."
REV_MUTS = os.path.join(BASE_REV, "rev_muts")
REV_PLOTS = os.path.join(BASE_REV, "rev_plots", "fearon_definition", "SFig5")
DATE_STAMP = "20260706"
os.makedirs(REV_PLOTS, exist_ok=True)

HORIZONS = (365, 730, 1095)
FORMULA_COVS = ["TP53", "STK11", "STAGE_CDM_DERIVED_GRANULAR", "start_BMI"]

REV_INPUTS = os.path.join(BASE_REV, "rev_inputs")
msk = pd.read_csv(
    os.path.join(REV_MUTS, "results_mutation_WL5_BMIlt20_20260706", "cachexia_data_survival_mutation_WL5_BMIlt20_20260706.csv"),
    low_memory=False,
)
luad = msk[msk["CANCER_TYPE_DETAILED"] == "Lung Adenocarcinoma"].copy()

bmi = pd.read_csv(os.path.join(REV_INPUTS, "bmi_final_20260129.csv"), header=0)
bmi_baseline = bmi.groupby("MRN", group_keys=False).apply(lambda x: x.loc[(x["days_since_anchor"]).abs().idxmin()])
bmi_baseline.reset_index(inplace=True, drop=True)
luad = luad.merge(bmi_baseline[["MRN", "bmi"]], on="MRN", how="inner")
luad = luad.rename(columns={"bmi": "start_BMI"})

use_cols = FORMULA_COVS + ["time_to_cachexia", "cachexia_event"]
luad = luad.dropna(subset=use_cols).reset_index(drop=True)
print(f"MSK LUAD training set: n={len(luad)}, events={int(luad['cachexia_event'].sum())}")

formula = " + ".join(FORMULA_COVS)
cph = CoxPHFitter()
cph.fit(luad, duration_col="time_to_cachexia", event_col="cachexia_event", formula=formula)
print(cph.summary[["coef", "exp(coef)", "p"]])

DFCI_ROOT = os.path.join(REV_MUTS, "FILE_SHARED_WITH_MSKCC_V1", "deidentified_data",
                          "NSCLC_cachexia", "diagdate_1ca_tier1or2or3_bmi", "5WL_BMI20")
dfci = pd.read_csv(os.path.join(DFCI_ROOT, "patient_data_deidentified_with_startBMI.csv"), low_memory=False)
dfci = dfci[dfci["time_to_cachexia"] >= 0].copy()
dfci_use_cols = ["TP53", "STK11", "STAGE_CDM_DERIVED_GRANULAR", "start_BMI", "time_to_cachexia", "cachexia_event"]
dfci = dfci.dropna(subset=[c for c in dfci_use_cols if c in dfci.columns]).reset_index(drop=True)
print(f"\nDFCI NSCLC test set: n={len(dfci)}, events={int(dfci['cachexia_event'].sum())}")

pred_dfci = cph.predict_partial_hazard(dfci)
c_dfci = concordance_index(dfci["time_to_cachexia"], -pred_dfci, dfci["cachexia_event"])
print(f"\nDFCI C-index (MSK-trained reduced model): {c_dfci:.4f}")

from sklearn.metrics import roc_auc_score
auc_vals = []
for h in HORIZONS:
    event_h = (dfci["cachexia_event"] & (dfci["time_to_cachexia"] <= h)).astype(int)
    a = roc_auc_score(event_h, pred_dfci)
    auc_vals.append(a)
    print(f"AUC @ {h}d: {a:.4f}")

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 9
HORIZON_COLORS = {365: "#4FA3A3", 730: "#B5A22A", 1095: "#B0578D"}
HORIZON_MARKERS = {365: "o", 730: "s", 1095: "^"}

sf = cph.predict_survival_function(dfci)
times_avail = sf.index.to_numpy(float)
dfci_cal = dfci.copy()
for h in HORIZONS:
    t_near = times_avail[np.abs(times_avail - h).argmin()]
    dfci_cal[f"pred_{h}"] = 1 - sf.loc[t_near].values

dfci_cal["risk_bin"] = pd.qcut(dfci_cal["pred_365"], q=4, labels=False, duplicates="drop")

cal_rows = []
for h in HORIZONS:
    for grp in sorted(dfci_cal["risk_bin"].dropna().unique()):
        sub = dfci_cal[dfci_cal["risk_bin"] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time_to_cachexia"], event_observed=sub["cachexia_event"])
        obs = 1 - kmf.predict(h)
        cal_rows.append({"horizon": h, "pred_mean": sub[f"pred_{h}"].mean(), "observed": obs})
cal_df = pd.DataFrame(cal_rows)

fig, ax = plt.subplots(figsize=(2.8, 2.8))
for h in HORIZONS:
    sub = cal_df[cal_df.horizon == h]
    ax.plot(sub["pred_mean"], sub["observed"], marker=HORIZON_MARKERS[h], color=HORIZON_COLORS[h],
            label=f"{round(h/365)}y", linewidth=1)
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
ax.set_xlabel("Mean predicted risk")
ax.set_ylabel("Observed event proportion")
ax.set_title("NSCLC DFCI Test Set")
ax.legend(fontsize=7, frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
cal_fp = os.path.join(REV_PLOTS, f"DFCI_external_calibration_retrained_{DATE_STAMP}.pdf")
plt.savefig(cal_fp)
plt.close()
print(f"\n[SAVED] {cal_fp}")

from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(2.8, 2.8))
for h, a in zip(HORIZONS, auc_vals):
    event_h = (dfci["cachexia_event"] & (dfci["time_to_cachexia"] <= h)).astype(int)
    fpr, tpr, _ = roc_curve(event_h, pred_dfci)
    ax.plot(fpr, tpr, color=HORIZON_COLORS[h], label=f"{round(h/365)}y AUC={a:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("NSCLC DFCI Test Set")
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
roc_fp = os.path.join(REV_PLOTS, f"DFCI_external_ROC_retrained_{DATE_STAMP}.pdf")
plt.savefig(roc_fp)
plt.close()
print(f"[SAVED] {roc_fp}")

summary = pd.DataFrame([{
    "n_MSK": len(luad), "events_MSK": int(luad["cachexia_event"].sum()),
    "n_DFCI": len(dfci), "events_DFCI": int(dfci["cachexia_event"].sum()),
    "C_index_DFCI": c_dfci,
    "AUC_1y": auc_vals[0], "AUC_2y": auc_vals[1], "AUC_3y": auc_vals[2],
}])
summary_fp = os.path.join(REV_PLOTS, f"DFCI_reduced_model_retrained_summary_{DATE_STAMP}.csv")
summary.to_csv(summary_fp, index=False)
print(f"[SAVED] {summary_fp}")
