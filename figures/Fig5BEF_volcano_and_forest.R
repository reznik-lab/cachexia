library(ggplot2)
library(data.table)
library(ggrepel)

# Fig5 MSK discovery volcano + LUAD/COADREAD multivariate forest plots (Panels E/F).
# Adapted from 0303_ccx_revisions/rev_code/Figures/Fig5.R, reading the outputs of
# Fig5_univariate_mutation_cox.py / Fig5_multivariate_mutation_cox.py instead of
# the original (which had a save/read filename mismatch and a broken CRC Sidedness
# merge - both fixed upstream in the parameterized Python scripts).

BASE_REV   <- "."
REV_MUTS   <- file.path(BASE_REV, "rev_muts")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots")
DATE_STAMP <- "20260706"
WL_LABEL   <- "WL5_BMIlt20"

fig_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig5")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

mut_run_dir <- file.path(REV_MUTS, paste0("results_mutation_", WL_LABEL, "_", DATE_STAMP))

code_map <- c(
  "Esophageal Adenocarcinoma" = "ESCA", "Acute Myeloid Leukemia" = "AML",
  "Stomach Adenocarcinoma" = "STAD", "Pancreatic Adenocarcinoma" = "PAAD",
  "Small Cell Lung Cancer" = "SCLC", "Intrahepatic Cholangiocarcinoma" = "IHCH",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "High-Grade Serous Ovarian Cancer" = "HGSOC", "Cancer of Unknown Primary" = "CUP",
  "Colon Adenocarcinoma" = "COAD", "Colorectal Adenocarcinoma" = "COADREAD",
  "Rectal Adenocarcinoma" = "READ", "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCLNOS",
  "Lung Squamous Cell Carcinoma" = "LUSC", "Non-Small Cell Lung Cancer" = "NSCLC",
  "Bladder Urothelial Carcinoma" = "BLCA", "Glioblastoma Multiforme" = "GBM",
  "Lung Adenocarcinoma" = "LUAD", "Renal Clear Cell Carcinoma" = "CCRCC",
  "Plasma Cell Myeloma" = "PCM", "Invasive Breast Carcinoma" = "BRCA",
  "Myelodysplastic Workup" = "MDSWP", "Gastrointestinal Stromal Tumor" = "GIST",
  "Breast Invasive Ductal Carcinoma" = "IDC", "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC", "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD", "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL",
  "Adenocarcinoma of the Gastroesophageal Junction" = "GEJ",
  "Adenocarcinoma, NOS" = "ADNOS", "Endometrial Carcinoma" = "UCEC",
  "Glioblastoma" = "GB", "Hepatocellular Carcinoma" = "HCC",
  "Leiomyosarcoma" = "LM", "Melanoma" = "MEL", "Neuroblastoma" = "NBL",
  "Pancreatic Neuroendocrine Tumor" = "PANET", "Papillary Thyroid Cancer" = "THPA",
  "Sarcoma, NOS" = "SARCNOS",
  "Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma" = "MFH",
  "Upper Tract Urothelial Carcinoma" = "UTUC",
  "Uterine Carcinosarcoma/Uterine Malignant Mixed Mullerian Tumor" = "UCS"
)
oncotree_map <- data.table(detailed_cancer_type = names(code_map), oncotree_code = unname(code_map))

# ============================================================
# MSK discovery volcano
# ============================================================

muts_fp <- file.path(mut_run_dir, paste0("mutation_cox_cachexia_results_", WL_LABEL, "_", DATE_STAMP, ".csv"))
muts <- fread(muts_fp)
setnames(muts, "V1", "cancer_gene_id", skip_absent = TRUE)
setnames(muts, "detailed_cancer_type", "detailed_cancer_type", skip_absent = TRUE)

muts <- merge(muts, oncotree_map, by = "detailed_cancer_type", all.x = TRUE)
muts[, label := paste0(oncotree_code, ":", mutation)]
muts[, log2HR := log2(`exp(coef)`)]
muts[, neg_log10_fdr := -log10(p_adj)]

p_thresh <- 0.05
log2HR_thresh <- 0.5
fdr_thresh <- 0.10

muts[, sig := fifelse(p_adj < fdr_thresh & `exp(coef)` > 1, "Up (HR>1)",
                fifelse(p_adj < fdr_thresh & `exp(coef)` < 1, "Down (HR<1)", "Not Sig"))]
muts[, sig := factor(sig, levels = c("Up (HR>1)", "Down (HR<1)", "Not Sig"))]

top_hits <- muts[order(p_adj)][1:min(12, .N), label]

volcano <- ggplot(muts, aes(x = log2HR, y = neg_log10_fdr, color = sig)) +
  geom_point(alpha = 0.9, size = 1.5) +
  scale_x_continuous(limits = c(-2.3, 1.5), breaks = c(-2, -1, 0, 1, 1.5)) +
  scale_color_manual(values = c("Up (HR>1)" = "#EF6F6AFF", "Down (HR<1)" = "#6388B4", "Not Sig" = "gray80")) +
  geom_vline(xintercept = c(-log2HR_thresh, log2HR_thresh), linetype = "dashed", color = "black", linewidth = 0.3) +
  geom_hline(yintercept = -log10(fdr_thresh), linetype = "dashed", color = "black", linewidth = 0.3) +
  ggrepel::geom_text_repel(
    data = muts[label %in% top_hits], aes(label = label),
    size = 2, max.overlaps = 100, box.padding = 0.4, segment.size = 0.3
  ) +
  labs(x = expression(log[2] * "(HR)"), y = expression(-log[10] * "(FDR)"), color = "Significance", title = NULL) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    legend.text  = element_text(size = 9),
    legend.title = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    panel.grid   = element_blank(),
    axis.ticks   = element_line(color = "black", linewidth = 0.3),
    legend.position = "top"
  )

ggsave(file.path(fig_dir, paste0("Fig5_volcano_mutation_cox_cachexia_", WL_LABEL, "_", DATE_STAMP, ".pdf")),
       volcano, width = 3.7, height = 3.6)

si_table <- muts[, .(
  oncotree_code, mutation,
  `OR (95% CI)` = sprintf("%.3f (%.3f-%.3f)", `exp(coef)`, `exp(coef) lower 95%`, `exp(coef) upper 95%`),
  log2_or = round(log2HR, 2), p = signif(p, 3), FDR = signif(p_adj, 3)
)]
setorder(si_table, oncotree_code, mutation)
fwrite(si_table, file.path(mut_run_dir, paste0("STable_mutation_cox_", WL_LABEL, "_", DATE_STAMP, ".csv")))

cat(sprintf("\nMSK volcano: %d cancer-gene pairs, %d significant at FDR<%.2f\n",
            nrow(muts), sum(muts$p_adj < fdr_thresh), fdr_thresh))

# ============================================================
# LUAD / COADREAD multivariate forest plots
# ============================================================

clean_cov_label <- function(dt, cancer_type) {
  dt[, cov_label := as.character(covariate)]
  dt[, cov_label := gsub("SAMPLE_TYPE\\[T\\.", "", cov_label)]
  dt[, cov_label := gsub("GENDER\\[T\\.(.*)\\]", "\\1", cov_label)]
  dt[, cov_label := gsub("CVR_TMB_SCORE", "CVR TMB Score", cov_label)]
  dt[, cov_label := gsub("STAGE_CDM_DERIVED_GRANULAR", "Stage", cov_label)]
  dt[, cov_label := gsub("ECOG_KPS", "ECOG/KPS", cov_label)]
  dt[, cov_label := gsub("start_BMI", "Start BMI", cov_label)]
  dt[, cov_label := gsub("age_at_diagnosis", "Age at Diagnosis", cov_label)]
  dt[, cov_label := gsub("\\]", "", cov_label)]
  dt <- dt[!is.na(cov_label) & trimws(cov_label) != ""]
  if (cancer_type == "Colorectal Adenocarcinoma") {
    dt <- dt[cov_label != "Unknown"]
    dt[, cov_label := gsub("MSI_TYPE\\[T\\.", "", cov_label)]
    dt[, cov_label := gsub("Sidedness\\[T\\.", "", cov_label)]
  }
  dt
}

assign_group <- function(dt, cancer_type, gene_set = character(0)) {
  if (cancer_type == "Colorectal Adenocarcinoma") {
    dt[, group := fifelse(covariate %like% "SAMPLE_TYPE", "Sample Type",
                    fifelse(covariate %like% "MSI_TYPE", "MSI Type",
                     fifelse(covariate %like% "Sidedness", "Tumor Feature",
                      fifelse(covariate %in% gene_set, "Gene",
                       fifelse(covariate %in% c("CVR_TMB_SCORE","STAGE_CDM_DERIVED_GRANULAR","ECOG_KPS","start_BMI"),
                               "Clinical", "Demographic")))))]
  } else {
    dt[, group := fifelse(covariate %like% "SAMPLE_TYPE", "Sample Type",
                    fifelse(covariate %in% gene_set, "Gene",
                     fifelse(covariate %in% c("CVR_TMB_SCORE","STAGE_CDM_DERIVED_GRANULAR","ECOG_KPS","start_BMI"),
                             "Clinical", "Demographic")))]
  }
  dt[, sig := ifelse(p_adj < 0.1, "Significant", "Not significant")]
  dt
}

plot_mv_forest <- function(cancer_type, title, ordered_labels, gene_set, width, height, out_name) {
  in_fp <- file.path(mut_run_dir, paste0("multivariate_", WL_LABEL, "_", DATE_STAMP), "cancer_types",
                      paste0(cancer_type, ".csv"))
  if (!file.exists(in_fp)) stop("Missing multivariate CSV: ", in_fp)

  dt <- fread(in_fp)
  setnames(dt, "V1", "covariate", skip_absent = TRUE)
  dt <- dt[!is.na(covariate)]

  dt <- clean_cov_label(dt, cancer_type)
  dt <- assign_group(dt, cancer_type, gene_set = gene_set)

  extra <- setdiff(unique(dt$cov_label), ordered_labels)
  if (length(extra) > 0) {
    cat(sprintf("[%s] labels not in ordered_labels (appended at end): %s\n", cancer_type, paste(extra, collapse = ", ")))
    ordered_labels <- c(ordered_labels, extra)
  }

  dt[, cov_label := factor(cov_label, levels = ordered_labels)]
  dt <- dt[!is.na(cov_label)]
  setorder(dt, cov_label)

  dt[, plot_y := .I]
  dt[, block := rleid(group)]
  group_map <- dt[, .(ymin = min(plot_y) - 0.5, ymax = max(plot_y) + 0.5, group = first(group)),
                  by = block][, fill := rep(c("gray90", "#ffffff"), length.out = .N)]

  p <- ggplot() +
    geom_rect(data = group_map, aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax, fill = fill),
              inherit.aes = FALSE, alpha = 0.4) +
    scale_fill_identity() +
    geom_linerange(data = dt, aes(xmin = `coef lower 95%`, xmax = `coef upper 95%`, y = plot_y), linewidth = 0.4) +
    geom_point(data = dt, aes(x = coef, y = plot_y, color = sig), size = 2.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.4) +
    scale_color_manual(values = c("Significant" = "#EF6F6AFF", "Not significant" = "gray40")) +
    scale_y_continuous(breaks = dt$plot_y, labels = dt$cov_label, expand = expansion(mult = c(0.01, 0.01))) +
    labs(x = "log(HR)", y = NULL, title = title, color = NULL) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      axis.title.x = element_text(size = 9),
      axis.text.x  = element_text(size = 9),
      axis.text.y  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom"
    )

  ggsave(file.path(fig_dir, out_name), plot = p, width = width, height = height, units = "in", device = "pdf")
  invisible(p)
}

luad_order <- c(
  "Unknown", "Primary", "Metastasis",
  "STK11", "RBM10", "TP53",
  "ECOG/KPS", "Start BMI", "CVR TMB Score", "Stage",
  "Age at Diagnosis", "MALE"
)

plot_mv_forest(
  cancer_type = "Lung Adenocarcinoma", title = "LUAD",
  ordered_labels = luad_order, gene_set = c("STK11", "RBM10", "TP53"),
  out_name = paste0("Fig5_E_multivariate_luad_", DATE_STAMP, ".pdf"),
  width = 2.9, height = 3.5
)

# PIK3CA is newly significant in the corrected data (wasn't in the 0303 original
# gene set) - added to both the gene classification and the display order.
coad_order <- c(
  "Unknown", "Primary", "Metastasis",
  "Do not report", "Indeterminate", "Instable",
  "ARID1A", "FBXW7", "KMT2B", "KRAS", "PIK3CA", "PTEN", "SMAD4", "SOX9", "TCF7L2", "TP53", "AMER1",
  "ECOG/KPS", "CVR TMB Score", "Stage", "Start BMI",
  "Rectum", "Right",
  "Age at Diagnosis", "MALE"
)

plot_mv_forest(
  cancer_type = "Colorectal Adenocarcinoma", title = "COADREAD",
  ordered_labels = coad_order,
  gene_set = c("ARID1A", "FBXW7", "KMT2B", "KRAS", "PIK3CA", "PTEN", "SMAD4", "SOX9", "TCF7L2", "TP53", "AMER1"),
  out_name = paste0("Fig5_F_multivariate_coadread_", DATE_STAMP, ".pdf"),
  width = 2.8, height = 4.9
)

cat("\nWrote Fig5 volcano + LUAD/COADREAD forest plots to:", fig_dir, "\n")
