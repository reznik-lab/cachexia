library(data.table)
library(dplyr)
library(ggplot2)
library(ggrepel)

# SFig5 C/D: dosage-sensitivity volcanoes. Takes the WL5 discovery hits
# (FDR<0.1) and re-checks those SAME cancer-gene pairs at the 10% and 15%
# weight-loss thresholds using raw p<0.05 (not an independent fresh discovery
# + FDR correction at each threshold - that washes out significance since
# WL15 has far fewer qualifying episodes to begin with).
# Adapted from 0303_ccx_revisions/rev_code/Figures/SFig5_genomics.R (lines 1-191).

BASE_REV   <- "."
REV_MUTS   <- file.path(BASE_REV, "rev_muts")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots")
REV_TABLES <- file.path(BASE_REV, "rev_tables")
DATE_STAMP <- "20260706"

sfig_dir <- file.path(REV_PLOTS, "fearon_definition", "SFig5")
dir.create(sfig_dir, recursive = TRUE, showWarnings = FALSE)
si_dir <- file.path(REV_TABLES, "SI_tables")
dir.create(si_dir, recursive = TRUE, showWarnings = FALSE)

disc_fp <- file.path(REV_MUTS, "results_mutation_WL5_BMIlt20_20260706",
                      paste0("mutation_cox_cachexia_results_WL5_BMIlt20_", DATE_STAMP, ".csv"))
f10_fp  <- file.path(REV_MUTS, "results_mutation_WL10_20260706",
                      paste0("mutation_cox_cachexia_results_WL10_", DATE_STAMP, ".csv"))
f15_fp  <- file.path(REV_MUTS, "results_mutation_WL15_20260706",
                      paste0("mutation_cox_cachexia_results_WL15_", DATE_STAMP, ".csv"))

disc <- fread(disc_fp)
dt10 <- fread(f10_fp)
dt15 <- fread(f15_fp)

disc_sig <- disc[p_adj < 0.10, .(detailed_cancer_type, mutation)]
cat(sprintf("WL5 discovery hits (FDR<0.10): %d cancer-gene pairs\n", nrow(disc_sig)))

dt10_matched <- merge(dt10, disc_sig, by = c("detailed_cancer_type", "mutation"))
dt15_matched <- merge(dt15, disc_sig, by = c("detailed_cancer_type", "mutation"))

cat(sprintf("Of these, still p<0.05 at WL10: %d / %d\n", sum(dt10_matched$p < 0.05, na.rm = TRUE), nrow(disc_sig)))
cat(sprintf("Of these, still p<0.05 at WL15: %d / %d\n", sum(dt15_matched$p < 0.05, na.rm = TRUE), nrow(disc_sig)))

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
  "Breast Invasive Ductal Carcinoma" = "IDC", "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC", "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD", "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL",
  "Endometrial Carcinoma" = "UCEC"
)

prep_for_volcano <- function(dt) {
  dt <- dt[order(p)]
  dt <- dt[!duplicated(dt, by = c("detailed_cancer_type", "mutation"))]
  dt[, log2HR := log2(`exp(coef)`)]
  dt[, neglog10p := -log10(p)]
  dt[, oncotree_code := unname(code_map[detailed_cancer_type])]
  dt[, label := paste0(oncotree_code, ":", mutation)]
  dt[, sig := fifelse(p < 0.05 & `exp(coef)` > 1, "Up (HR>1)",
               fifelse(p < 0.05 & `exp(coef)` < 1, "Down (HR<1)", "Not Sig"))]
  dt[, sig := factor(sig, levels = c("Up (HR>1)", "Down (HR<1)", "Not Sig"))]
  dt
}

make_volcano <- function(dt, title, y_breaks = NULL) {
  dt_top <- dt[sig != "Not Sig"][order(-neglog10p)][1:min(10, .N)]
  p <- ggplot(dt, aes(x = log2HR, y = neglog10p, color = sig)) +
    geom_point(alpha = 0.9, size = 1.5) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black", linewidth = 0.3) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.3) +
    ggrepel::geom_text_repel(data = dt_top, aes(label = label), size = 2, max.overlaps = 100, box.padding = 0.4) +
    scale_color_manual(values = c("Up (HR>1)" = "#EF6F6AFF", "Down (HR<1)" = "#6388B4", "Not Sig" = "gray80")) +
    labs(x = expression(log[2] * "(HR)"), y = expression(-log[10] * "(p)"), color = "Significance", title = title) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      axis.title   = element_text(size = 9),
      axis.text    = element_text(size = 9),
      legend.text  = element_text(size = 9),
      legend.title = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      panel.grid   = element_blank(),
      axis.ticks   = element_line(color = "black", linewidth = 0.3),
      legend.position = "top"
    )
  if (!is.null(y_breaks)) p <- p + scale_y_continuous(breaks = y_breaks)
  p
}

dt10_v <- prep_for_volcano(dt10_matched)
dt15_v <- prep_for_volcano(dt15_matched)

p_c <- make_volcano(dt10_v, "10% WL discovery-hit recheck", y_breaks = seq(0, 8.5, by = 1))
p_d <- make_volcano(dt15_v, "15% WL discovery-hit recheck", y_breaks = seq(0, 4, by = 1))

ggsave(file.path(sfig_dir, paste0("Fig5_C_volcano_10pct_discovery_", DATE_STAMP, ".pdf")), p_c, width = 3.7, height = 3.6)
ggsave(file.path(sfig_dir, paste0("Fig5_D_volcano_15pct_discovery_", DATE_STAMP, ".pdf")), p_d, width = 3.7, height = 3.6)

fwrite(dt10_v[p < 0.05], file.path(si_dir, paste0("STable_uni_10WL_recheck_", DATE_STAMP, ".csv")))
fwrite(dt15_v[p < 0.05], file.path(si_dir, paste0("STable_uni_15WL_recheck_", DATE_STAMP, ".csv")))

cat("\nWrote SFig5 C/D (threshold sensitivity volcanoes) to:", sfig_dir, "\n")
