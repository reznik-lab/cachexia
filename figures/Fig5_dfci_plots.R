library(ggplot2)
library(data.table)
library(ggrepel)

# DFCI validation volcano + NSCLC multivariate forest plot.
# Adapted from 0303_ccx_revisions/rev_code/Figures/Fig5.R (DFCI sections).

BASE_REV   <- "."
REV_MUTS   <- file.path(BASE_REV, "rev_muts")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots")
DATE_STAMP <- "20260706"
THRESHOLD  <- "5WL_BMI20"
COMBINATION <- "diagdate_1ca_tier1or2or3_bmi"

fig_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig5")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

dfci_run_dir <- file.path(REV_MUTS, paste0("results_dfci_", THRESHOLD, "_", DATE_STAMP))

# ============================================================
# DFCI discovery volcano
# ============================================================

dfci_fp <- file.path(dfci_run_dir, paste0("mutation_cox_cachexia_results_DFCI_", COMBINATION, "_", THRESHOLD, "_", DATE_STAMP, ".csv"))
dfci <- fread(dfci_fp)
setnames(dfci, "V1", "cancer_gene_id", skip_absent = TRUE)

dfci_code_map <- c(
  BladderCancer = "BLADDER", Breastcancer = "BREAST", CRC = "CRC",
  EndometrialCancer = "ENDO", EsophagogastricCarcinoma = "ESO-GAST",
  Glioma = "GLIOMA", Melanoma = "MEL", NSCLC = "NSCLC",
  Pancreaticcancer = "PANCREAS", ProstateCancer = "PROSTATE",
  RCC = "RCC", Ovariancancer = "OVARIAN"
)
missing_map <- setdiff(unique(dfci$cancer_type_dfci), names(dfci_code_map))
if (length(missing_map) > 0) stop("Missing dfci_code_map entries for: ", paste(missing_map, collapse = ", "))

dfci[, dfci_code := unname(dfci_code_map[cancer_type_dfci])]
dfci[, log2HR := log2(`exp(coef)`)]
dfci[, neg_log10_fdr := -log10(p_adj)]
dfci[, label := paste0(dfci_code, ": ", mutation)]

fdr_thresh <- 0.10
dfci[, sig := fifelse(p_adj < fdr_thresh & `exp(coef)` > 1, "Up (HR>1)",
               fifelse(p_adj < fdr_thresh & `exp(coef)` < 1, "Down (HR<1)", "Not Sig"))]
dfci[, sig := factor(sig, levels = c("Up (HR>1)", "Down (HR<1)", "Not Sig"))]

# genes shared with the MSK primary discovery hits vs. DFCI-only findings
msk_fp <- file.path(REV_MUTS, "results_mutation_WL5_BMIlt20_20260706",
                     paste0("mutation_cox_cachexia_results_WL5_BMIlt20_", DATE_STAMP, ".csv"))
msk_sig <- fread(msk_fp)[p_adj < 0.10, unique(mutation)]

sig_dfci <- dfci[p_adj < fdr_thresh]
label_pairs <- sig_dfci[mutation %in% msk_sig]
dfci_only_pairs <- sig_dfci[!(mutation %in% msk_sig)]

volcano_dfci <- ggplot(dfci, aes(x = log2HR, y = neg_log10_fdr, color = sig)) +
  geom_point(alpha = 0.9, size = 1.5) +
  scale_color_manual(values = c("Up (HR>1)" = "#EF6F6AFF", "Down (HR<1)" = "#6388B4", "Not Sig" = "gray80")) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "black", linewidth = 0.3) +
  geom_hline(yintercept = -log10(fdr_thresh), linetype = "dashed", color = "black", linewidth = 0.3) +
  ggrepel::geom_text_repel(data = label_pairs, aes(label = label), size = 2, color = "black",
                            max.overlaps = 100, box.padding = 0.4, segment.size = 0.3) +
  ggrepel::geom_text_repel(data = dfci_only_pairs, aes(label = label), size = 2, color = "black", fontface = "italic",
                            max.overlaps = 100, box.padding = 0.4, segment.size = 0.3) +
  labs(x = expression(log[2] * "(HR)"), y = expression(-log[10] * "(FDR)"), color = "Significance",
       title = "DFCI validation") +
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

ggsave(file.path(fig_dir, paste0("Fig5_volcano_mutation_cox_cachexia_DFCI_", THRESHOLD, "_", DATE_STAMP, ".pdf")),
       volcano_dfci, width = 3.7, height = 3.6)

cat(sprintf("DFCI volcano: %d pairs, %d significant at FDR<%.2f (%d shared with MSK, %d DFCI-only)\n",
            nrow(dfci), sum(dfci$p_adj < fdr_thresh), fdr_thresh, nrow(label_pairs), nrow(dfci_only_pairs)))

# ============================================================
# DFCI NSCLC multivariate forest plot
# ============================================================

nsclc_fp <- file.path(dfci_run_dir, paste0("multivariate_", THRESHOLD), "cancer_types",
                       paste0("NSCLC_multivariate_", THRESHOLD, ".csv"))

clean_dfci_label <- function(dt) {
  dt[, cov_label := as.character(covariate)]
  dt[, cov_label := gsub("^GENDER_", "", cov_label)]
  dt[, cov_label := gsub("^SAMPLE_TYPE_", "", cov_label)]
  dt[, cov_label := gsub("^ANCESTRY_LABEL_", "Ancestry: ", cov_label)]
  dt[, cov_label := gsub("^age_at_diagnosis_binned_", "Age ", cov_label)]
  dt[, cov_label := gsub("CVR_TMB_SCORE", "CVR TMB Score", cov_label)]
  dt[, cov_label := gsub("STAGE_CDM_DERIVED_GRANULAR", "Stage", cov_label)]
  dt[, cov_label := tools::toTitleCase(tolower(cov_label))]
  dt[, cov_label := gsub("^Tp53$", "TP53", cov_label)]
  dt[, cov_label := gsub("Cvr Tmb Score", "CVR TMB Score", cov_label)]
  dt
}

assign_group_dfci <- function(dt, gene_set) {
  dt[, group := fifelse(covariate %like% "SAMPLE_TYPE", "Sample Type",
                  fifelse(covariate %like% "ANCESTRY_LABEL", "Ancestry",
                   fifelse(covariate %like% "age_at_diagnosis_binned", "Demographic",
                    fifelse(covariate %in% gene_set, "Gene",
                     fifelse(covariate %in% c("CVR_TMB_SCORE", "STAGE_CDM_DERIVED_GRANULAR"),
                             "Clinical", "Demographic")))))]
  dt
}

if (file.exists(nsclc_fp)) {
  dt <- fread(nsclc_fp)
  setnames(dt, "V1", "covariate", skip_absent = TRUE)
  dt <- dt[!is.na(covariate)]
  dt[, sig := ifelse(p_adj < 0.1, "Significant", "Not significant")]
  dt <- clean_dfci_label(dt)
  dt <- assign_group_dfci(dt, gene_set = "TP53")

  ordered_labels <- c("Ancestry: Eur", "Ancestry: Eas", "Ancestry: Amr",
                       "Unspecified", "Primary", "Metastatic Recurrence",
                       "Male",
                       "Age >=80", "Age <40", "Age 70-79", "Age 60-69", "Age 50-59",
                       "TP53",
                       "CVR TMB Score", "Stage")
  extra <- setdiff(unique(dt$cov_label), ordered_labels)
  if (length(extra) > 0) ordered_labels <- c(ordered_labels, extra)

  dt[, cov_label := factor(cov_label, levels = ordered_labels)]
  dt <- dt[!is.na(cov_label)]
  setorder(dt, cov_label)
  dt[, plot_y := .I]

  dt[, block := rleid(group)]
  group_map <- dt[, .(ymin = min(plot_y) - 0.5, ymax = max(plot_y) + 0.5, group = first(group)),
                  by = block][, fill := rep(c("gray90", "#ffffff"), length.out = .N)]

  p_nsclc <- ggplot() +
    geom_rect(data = group_map, aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax, fill = fill),
              inherit.aes = FALSE, alpha = 0.4) +
    scale_fill_identity() +
    geom_linerange(data = dt, aes(xmin = `coef lower 95%`, xmax = `coef upper 95%`, y = plot_y), linewidth = 0.4) +
    geom_point(data = dt, aes(x = coef, y = plot_y, color = sig), size = 2.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.4) +
    scale_color_manual(values = c("Significant" = "#EF6F6AFF", "Not significant" = "gray40")) +
    scale_y_continuous(breaks = dt$plot_y, labels = dt$cov_label, expand = expansion(mult = c(0.01, 0.01))) +
    labs(x = "log(HR)", y = NULL, title = "NSCLC (DFCI)", color = NULL) +
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

  ggsave(file.path(fig_dir, paste0("Fig5_multivariate_nsclc_DFCI_", DATE_STAMP, ".pdf")),
         p_nsclc, width = 3.5, height = 4.2, units = "in", device = "pdf")
  cat("Wrote DFCI NSCLC multivariate forest plot\n")
} else {
  cat("[SKIP] No NSCLC multivariate CSV found at:", nsclc_fp, "\n")
}

cat("\nWrote DFCI validation plots to:", fig_dir, "\n")
