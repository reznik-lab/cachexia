library(ggplot2)
library(data.table)

code_map <- c(
  "Esophageal Adenocarcinoma" = "ESCA",
  "Acute Myeloid Leukemia" = "AML",
  "Stomach Adenocarcinoma" = "STAD",
  "Pancreatic Adenocarcinoma" = "PAAD",
  "Small Cell Lung Cancer" = "SCLC",
  "Intrahepatic Cholangiocarcinoma" = "IHCH",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "High-Grade Serous Ovarian Cancer" = "HGSOC",
  "Cancer of Unknown Primary" = "CUP",
  "Colon Adenocarcinoma" = "COAD",
  "Colorectal Adenocarcinoma" = "COADREAD",
  "Rectal Adenocarcinoma" = "READ",
  "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCLNOS",
  "Lung Squamous Cell Carcinoma" = "LUSC",
  "Non-Small Cell Lung Cancer" = "NSCLC",
  "Bladder Urothelial Carcinoma" = "BLCA",
  "Glioblastoma Multiforme" = "GBM",
  "Lung Adenocarcinoma" = "LUAD",
  "Renal Clear Cell Carcinoma" = "CCRCC",
  "Plasma Cell Myeloma" = "PCM",
  "Invasive Breast Carcinoma" = "BRCA",
  "Myelodysplastic Workup" = "MDSWP",
  "Gastrointestinal Stromal Tumor" = "GIST",
  "Breast Invasive Ductal Carcinoma" = "IDC",
  "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC",
  "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD",
  "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL"
)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)
eps[, MRN := as.integer(MRN)]

msk_clin_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
msk_clin <- fread(msk_clin_fp)
msk_clin[, MRN := as.integer(MRN)]

eps_mrn <- eps[, .(has_cachexia = max(has_cachexia_valid_edemaQC, na.rm = TRUE)), by = MRN]
eps_mrn[is.infinite(has_cachexia) | is.na(has_cachexia), has_cachexia := 0L]

sum_dt <- merge(
  msk_clin[, .(MRN, CANCER_TYPE_DETAILED)],
  eps_mrn,
  by = "MRN",
  all.x = TRUE
)
sum_dt[is.na(has_cachexia), has_cachexia := 0L]

sum_dt[, CANCER_CODE := fifelse(
  CANCER_TYPE_DETAILED %chin% names(code_map),
  code_map[CANCER_TYPE_DETAILED],
  CANCER_TYPE_DETAILED
)]

prev_dt <- sum_dt[
  !is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED)),
  .(
    Total     = uniqueN(MRN),
    Cachectic = uniqueN(MRN[has_cachexia == 1])
  ),
  by = CANCER_CODE
]
prev_dt <- prev_dt[Total >= 500]
prev_dt[, Proportion := 100 * Cachectic / Total]
prev_dt[, SE    := sqrt((Proportion/100) * (1 - Proportion/100) / Total) * 100]
prev_dt[, Lower := pmax(0,   Proportion - 1.96 * SE)]
prev_dt[, Upper := pmin(100, Proportion + 1.96 * SE)]

setorder(prev_dt, Proportion)

cancer_levels <- prev_dt$CANCER_CODE
prev_dt[, CANCER_CODE := factor(CANCER_CODE, levels = cancer_levels)]

p2d <- ggplot(prev_dt, aes(x = Proportion, y = CANCER_CODE)) +
  geom_col(fill = "black", width = 0.85) +
  geom_segment(aes(x = Lower, xend = Proportion, yend = CANCER_CODE),
               color = "white", linewidth = 0.7) +
  geom_segment(aes(x = Proportion, xend = Upper, yend = CANCER_CODE),
               color = "black", linewidth = 0.7) +
  labs(x = "Prevalence (%)", y = NULL, title = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0))) +
  coord_cartesian(xlim = c(0, NA), clip = "off") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    axis.title.x = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line.x  = element_line(color = "black", linewidth = 0.25),
    axis.ticks.x = element_line(color = "black", linewidth = 0.25),
    axis.ticks.y = element_line(color = "black", linewidth = 0.25),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none",
    panel.spacing    = unit(1.2, "lines"),
    strip.text       = element_text(size = 9, face = "plain",
                                    margin = margin(t = 3, r = 6, b = 3, l = 6)),
    strip.background = element_rect(fill = "white", color = "black", linewidth = 0.4),
    strip.placement  = "outside"
  )

print(p2d)

ggsave(file.path(out_dir, "fig2D_prevalence_by_cancertype.pdf"),
       plot = p2d, width = 2.65, height = 5.0, dpi = 300)

fwrite(prev_dt, file.path(out_dir, "fig2D_fearon_cachexia_prevalence_by_cancer.csv"))
