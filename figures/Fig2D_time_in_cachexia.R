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
msk_clin[, anchor_final := as.IDate(anchor_final)]
msk_clin[, PLA_LAST_CONTACT_DTE := as.IDate(PLA_LAST_CONTACT_DTE)]
msk_clin[, `Tumor Diagnosis Date` := anchor_final]
msk_clin[, followup_days := as.numeric(PLA_LAST_CONTACT_DTE - `Tumor Diagnosis Date`)]

eps_mrn <- eps[, .(has_cachexia = max(has_cachexia_valid_edemaQC, na.rm = TRUE)), by = MRN]
eps_mrn[is.infinite(has_cachexia) | is.na(has_cachexia), has_cachexia := 0L]
sum_dt <- merge(msk_clin[, .(MRN, CANCER_TYPE_DETAILED)], eps_mrn, by = "MRN", all.x = TRUE)
sum_dt[is.na(has_cachexia), has_cachexia := 0L]
sum_dt[, CANCER_CODE := fifelse(CANCER_TYPE_DETAILED %chin% names(code_map), code_map[CANCER_TYPE_DETAILED], CANCER_TYPE_DETAILED)]
n_by_type <- sum_dt[!is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED)), .(Total = uniqueN(MRN)), by = CANCER_CODE]
cancer_levels_eligible <- n_by_type[Total >= 500, CANCER_CODE]

eps[, `:=`(start_day = as.numeric(start_day), end_day = as.numeric(end_day))]
eps[, episode_duration := end_day - start_day]
eps[episode_duration < 0, episode_duration := NA_real_]

cachexia_days_per_patient <- eps[, .(
  total_cachexia_days = sum(episode_duration, na.rm = TRUE)
), by = MRN]

cachexia_proportion <- merge(
  msk_clin[, .(MRN, followup_days, CANCER_TYPE_DETAILED)],
  cachexia_days_per_patient,
  by = "MRN",
  all.x = TRUE
)
cachexia_proportion[is.na(total_cachexia_days), total_cachexia_days := 0]
cachexia_proportion[, cachexia_fraction := fifelse(
  followup_days > 0,
  total_cachexia_days / followup_days,
  NA_real_
)]

cachexia_proportion <- cachexia_proportion[
  !is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED))
]
cachexia_proportion[, Cancer_Code := fifelse(
  CANCER_TYPE_DETAILED %chin% names(code_map),
  code_map[CANCER_TYPE_DETAILED],
  CANCER_TYPE_DETAILED
)]

cachexia_proportion <- cachexia_proportion[
  Cancer_Code %in% cancer_levels_eligible & !is.na(cachexia_fraction)
]

ci_summary <- cachexia_proportion[, .(
  n             = .N,
  mean_cachexia = mean(cachexia_fraction, na.rm = TRUE),
  sd_cachexia   = sd(cachexia_fraction,   na.rm = TRUE)
), by = Cancer_Code]

ci_summary[, se := sd_cachexia / sqrt(n)]
ci_summary[, `:=`(
  lower = pmax(0, mean_cachexia - 1.96 * se),
  upper = pmin(1, mean_cachexia + 1.96 * se)
)]
setorder(ci_summary, mean_cachexia)
ci_summary[, Cancer_Code := factor(Cancer_Code, levels = Cancer_Code)]

global_dt   <- cachexia_proportion[!is.na(cachexia_fraction)]
global_mean <- global_dt[, mean(cachexia_fraction, na.rm = TRUE)]
global_se   <- global_dt[, sd(cachexia_fraction, na.rm = TRUE) / sqrt(.N)]
global_lo   <- max(0, global_mean - 1.96 * global_se)
global_hi   <- min(1, global_mean + 1.96 * global_se)

p1d <- ggplot(ci_summary, aes(x = 100 * mean_cachexia, y = Cancer_Code)) +
  annotate("rect", xmin = 100*global_lo, xmax = 100*global_hi,
           ymin = -Inf, ymax = Inf,
           fill = "gray85", alpha = 0.2) +
  geom_vline(xintercept = 100*global_mean, linetype = "dashed",
             linewidth = 0.5, color = "#6388B4FF") +
  geom_col(fill = "black", width = 0.85) +
  geom_segment(aes(x = 100*lower, xend = 100*mean_cachexia, yend = Cancer_Code),
               color = "white", linewidth = 0.6) +
  geom_segment(aes(x = 100*mean_cachexia, xend = 100*upper, yend = Cancer_Code),
               color = "gray10", linewidth = 0.6) +
  labs(x = "Time spent in cachexia (%)", y = NULL, title = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0))) +
  coord_cartesian(xlim = c(0, NA), clip = "off") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    axis.title.x = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line.x  = element_line(color = "black", linewidth = 0.25),
    axis.line.y  = element_line(color = "black", linewidth = 0.25),
    axis.ticks.x = element_line(color = "black", linewidth = 0.25),
    axis.ticks.y = element_line(color = "black", linewidth = 0.25),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

print(p1d)

ggsave(file.path(out_dir, "fig2D_time_in_cachexia.pdf"),
       plot = p1d, width = 2.65, height = 5.0, dpi = 300)

fwrite(ci_summary, file.path(out_dir, "fig2D_cachexia_time_proportion.csv"))

# cohort-wide mean/IQR of time spent in cachectic state (manuscript Para 1, line 37).
# Manuscript text says "median 13.8% (IQR 0-20.9%)" but neither whole-cohort nor
# episode-only median reproduces 13.8% under any definition tested (whole-cohort
# median is 0%, since >50% of patients never have an episode; episode-only median
# is ~25%). The MEAN (not median) of the whole cohort reproduces this almost
# exactly in both the original 0303 data (14.6%) and here - "median" in the
# manuscript text was very likely a mislabeled mean.
global_mean_pct   <- mean(global_dt$cachexia_fraction, na.rm = TRUE)
global_median_pct <- median(global_dt$cachexia_fraction, na.rm = TRUE)
global_iqr <- quantile(global_dt$cachexia_fraction, c(0.25, 0.75), na.rm = TRUE)
cat(sprintf("\nCohort-wide (n=%d): mean %.1f%% (median %.1f%%) of observed time in cachectic state (IQR %.1f-%.1f%%)\n",
            nrow(global_dt), 100*global_mean_pct, 100*global_median_pct, 100*global_iqr[1], 100*global_iqr[2]))
