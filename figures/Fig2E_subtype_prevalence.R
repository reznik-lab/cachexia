library(ggplot2)
library(dplyr)
library(data.table)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig2", "Fig2E_subtypes")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

msk_clin_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
msk_clin <- fread(msk_clin_fp)
msk_clin[, MRN := as.integer(MRN)]

eps_fp <- file.path(
  REV_RESULTS,
  paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv")
)
eps <- fread(eps_fp)
eps[, MRN := as.integer(MRN)]

has_ccx <- eps[, .(has_cachexia = as.integer(any(has_cachexia_valid_edemaQC == 1, na.rm = TRUE))), by = MRN]

cac_clin <- merge(msk_clin, has_ccx, by = "MRN", all.x = TRUE)
cac_clin[is.na(has_cachexia), has_cachexia := 0L]
cac_clin <- cac_clin[MRN %in% has_ccx$MRN]

cat("n =", cac_clin[, .N], " | ccx_rate =", cac_clin[, mean(has_cachexia)], "\n")

cac_clin[, CANCER_TYPE_DETAILED := dplyr::recode(
  CANCER_TYPE_DETAILED,
  "Renal Clear Cell Carcinoma" = "CCRCC",
  "Papillary Renal Cell Carcinoma" = "PRCC",
  "Lung Adenocarcinoma" = "LUAD",
  "Lung Squamous Cell Carcinoma" = "LUSC",
  "Small Cell Lung Cancer" = "SCLC",
  "Non-Small Cell Lung Cancer" = "NSCLC",
  "Breast Invasive Ductal Carcinoma" = "IDC",
  "Breast Invasive Lobular Carcinoma" = "ILC",
  "Uterine Endometrioid Carcinoma" = "UCEC",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCL",
  "Follicular Lymphoma" = "FL",
  "Esophageal Adenocarcinoma" = "ESCA",
  "Adenocarcinoma of the Gastroesophageal Junction" = "GEJ",
  "Hepatocellular Carcinoma" = "LIHC",
  "Intrahepatic Cholangiocarcinoma" = "CHOL",
  "Pancreatic Adenocarcinoma" = "PAAD",
  "Pancreatic Neuroendocrine Tumor" = "PNET",
  .default = CANCER_TYPE_DETAILED
)]

subtype_pairs <- list(
  "Kidney"       = c("CCRCC", "PRCC"),
  "Lung (NSCLC)" = c("LUAD", "LUSC"),
  "Lung"         = c("SCLC", "NSCLC"),
  "Breast"       = c("IDC", "ILC"),
  "Uterus"       = c("UCEC", "USC"),
  "Blood"        = c("DLBCL", "FL"),
  "Esophagus"    = c("ESCA", "GEJ"),
  "Liver"        = c("LIHC", "CHOL"),
  "Pancreas"     = c("PAAD", "PNET")
)

summary_table <- list()

for (tissue in names(subtype_pairs)) {
  subtypes <- subtype_pairs[[tissue]]

  dt <- cac_clin[CANCER_TYPE_DETAILED %chin% subtypes]
  if (nrow(dt) == 0) next

  dt_summary <- dt[, .(
    Cachectic  = sum(has_cachexia, na.rm = TRUE),
    Total      = .N,
    Prevalence = mean(has_cachexia, na.rm = TRUE) * 100
  ), by = CANCER_TYPE_DETAILED]

  dt_summary[, SE := sqrt((Prevalence/100) * (1 - Prevalence/100) / Total) * 100]
  dt_summary[, `:=`(
    Lower = pmax(0, Prevalence - 1.96 * SE),
    Upper = pmin(100, Prevalence + 1.96 * SE)
  )]

  ord <- dt_summary[order(-Prevalence), CANCER_TYPE_DETAILED]
  dt_summary[, CANCER_TYPE_DETAILED := factor(CANCER_TYPE_DETAILED, levels = ord)]

  a <- dt[CANCER_TYPE_DETAILED == subtypes[1], sum(has_cachexia, na.rm = TRUE)]
  b <- dt[CANCER_TYPE_DETAILED == subtypes[1], .N] - a
  c <- dt[CANCER_TYPE_DETAILED == subtypes[2], sum(has_cachexia, na.rm = TRUE)]
  d <- dt[CANCER_TYPE_DETAILED == subtypes[2], .N] - c

  fisher_test <- fisher.test(matrix(c(a, b, c, d), nrow = 2, byrow = TRUE))
  p_value <- fisher_test$p.value

  summary_table[[tissue]] <- data.table(
    Tissue   = tissue,
    Subtype1 = subtypes[1],
    Subtype2 = subtypes[2],
    p_value  = signif(p_value, 3),
    OR       = unname(fisher_test$estimate),
    a = a, b = b, c = c, d = d
  )

  p <- ggplot(dt_summary, aes(x = CANCER_TYPE_DETAILED, y = Prevalence)) +
    geom_bar(stat = "identity", width = 0.9, fill = "black") +
    geom_segment(aes(xend = CANCER_TYPE_DETAILED, y = Prevalence, yend = Upper),
                 color = "gray10", linewidth = 0.6) +
    geom_segment(aes(xend = CANCER_TYPE_DETAILED, y = Prevalence, yend = Lower),
                 color = "white", linewidth = 0.6) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.25) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.5, 0)) +
    labs(title = NULL, subtitle = NULL, x = NULL, y = "Prevalence (%)") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      axis.title.y = element_text(size = 9),
      axis.text.x  = element_text(size = 9),
      axis.text.y  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.subtitle = element_text(hjust = 0.5, size = 8),
      legend.position  = "none"
    )

  plot_file <- file.path(out_dir, paste0("fig2E_subtype_prevalence_", gsub(" ", "_", tissue), ".pdf"))
  ggsave(filename = plot_file, plot = p, width = 2, height = 1.5, dpi = 300)
}

summary_table_dt <- rbindlist(summary_table, fill = TRUE)
fwrite(summary_table_dt, file.path(out_dir, "fig2E_subtype_comparison_summary.csv"))
print(summary_table_dt)
