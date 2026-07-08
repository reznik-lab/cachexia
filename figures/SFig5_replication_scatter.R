library(data.table)
library(ggplot2)
library(ggrepel)


BASE_REV   <- "."
REV_MUTS   <- file.path(BASE_REV, "rev_muts")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots")
DATE_STAMP <- "20260706"

sfig_dir <- file.path(REV_PLOTS, "fearon_definition", "SFig5")
dir.create(sfig_dir, recursive = TRUE, showWarnings = FALSE)

msk_fp  <- file.path(REV_MUTS, "results_mutation_WL5_BMIlt20_20260706",
                      paste0("mutation_cox_cachexia_results_WL5_BMIlt20_", DATE_STAMP, ".csv"))
dfci_fp <- file.path(REV_MUTS, "results_dfci_5WL_BMI20_20260706",
                      paste0("mutation_cox_cachexia_results_DFCI_diagdate_1ca_tier1or2or3_bmi_5WL_BMI20_", DATE_STAMP, ".csv"))

msk  <- fread(msk_fp)
dfci <- fread(dfci_fp)

msk_to_dfci_map <- c(
  "Lung Adenocarcinoma" = "NSCLC",
  "Colorectal Adenocarcinoma" = "CRC",
  "Invasive Breast Carcinoma" = "Breastcancer",
  "Bladder Urothelial Carcinoma" = "BladderCancer",
  "Prostate Adenocarcinoma" = "ProstateCancer",
  "Pancreatic Adenocarcinoma" = "Pancreaticcancer",
  "Renal Clear Cell Carcinoma" = "RCC",
  "Uterine Endometrioid Carcinoma" = "EndometrialCancer",
  "Cutaneous Melanoma" = "Melanoma",
  "Glioblastoma Multiforme" = "Glioma",
  "Esophageal Adenocarcinoma" = "EsophagogastricCarcinoma"
)

msk[, dfci_ct := unname(msk_to_dfci_map[detailed_cancer_type])]
msk <- msk[!is.na(dfci_ct)]
msk[, log2HR_msk := log2(`exp(coef)`)]

dfci[, log2HR_dfci := log2(`exp(coef)`)]
setnames(dfci, "cancer_type_dfci", "dfci_ct", skip_absent = TRUE)

merged <- merge(
  msk[, .(dfci_ct, mutation, log2HR_msk, p_adj_msk = p_adj, detailed_cancer_type)],
  dfci[, .(dfci_ct, mutation, log2HR_dfci, p_adj_dfci = p_adj)],
  by = c("dfci_ct", "mutation")
)

merged[, category := fifelse(p_adj_msk < 0.1 & p_adj_dfci < 0.1, "Both FDR<0.1",
                      fifelse(p_adj_msk < 0.1, "Discovery",
                      fifelse(p_adj_dfci < 0.1, "Validation", "Neither")))]
merged[, category := factor(category, levels = c("Both FDR<0.1", "Discovery", "Validation", "Neither"))]

cat(sprintf("Matched cancer-gene pairs: %d | replicated (both FDR<0.1): %d\n",
            nrow(merged), sum(merged$category == "Both FDR<0.1")))

r_val <- cor(merged$log2HR_msk, merged$log2HR_dfci, use = "complete.obs")

cancer_code_map <- c(
  "Lung Adenocarcinoma" = "LUAD", "Colorectal Adenocarcinoma" = "COADREAD",
  "Invasive Breast Carcinoma" = "BRCA", "Bladder Urothelial Carcinoma" = "BLCA",
  "Prostate Adenocarcinoma" = "PRAD", "Pancreatic Adenocarcinoma" = "PAAD",
  "Renal Clear Cell Carcinoma" = "CCRCC", "Uterine Endometrioid Carcinoma" = "UEC",
  "Cutaneous Melanoma" = "SKCM", "Glioblastoma Multiforme" = "GBM",
  "Esophageal Adenocarcinoma" = "ESO"
)
merged[, label := paste0(unname(cancer_code_map[detailed_cancer_type]), ":", mutation)]

both_labels <- merged[category == "Both FDR<0.1"][order(p_adj_msk + p_adj_dfci)][1:min(5, .N)]
discovery_labels <- merged[category == "Discovery"][order(p_adj_msk)][1:min(2, .N)]
validation_labels <- merged[category == "Validation"][order(p_adj_dfci)][1:min(2, .N)]
label_pts <- rbindlist(list(both_labels, discovery_labels, validation_labels))

p_scatter <- ggplot(merged, aes(x = log2HR_msk, y = log2HR_dfci, color = category)) +
  geom_point(alpha = 0.8, size = 2.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.3) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "gray40", linewidth = 0.3) +
  ggrepel::geom_text_repel(data = label_pts, aes(label = label), color = "black", size = 2.2,
                            max.overlaps = 100, box.padding = 0.4, segment.size = 0.3) +
  scale_color_manual(values = c(
    "Both FDR<0.1" = "#8CC2CAFF", "Discovery" = "#6388B4FF",
    "Validation" = "#FFAE34FF", "Neither" = "gray70"
  )) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.1, vjust = 1.5,
           label = sprintf("r = %.2f", r_val), size = 3) +
  labs(x = expression("MSK " * log[2] * "(HR)"), y = expression("DFCI " * log[2] * "(HR)"),
       color = NULL, title = "MSK vs DFCI replication") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title   = element_text(size = 9),
    axis.text    = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    panel.grid   = element_blank(),
    axis.ticks   = element_line(color = "black", linewidth = 0.3),
    legend.position = "none"
  )

out_fp <- file.path(sfig_dir, paste0("replication_scatter_log2HR_MSK_vs_DFCI_", DATE_STAMP, ".pdf"))
ggsave(out_fp, p_scatter, width = 2.9, height = 2.8, units = "in", device = "pdf")

fwrite(merged, file.path(sfig_dir, paste0("replication_scatter_data_", DATE_STAMP, ".csv")))

cat("\nWrote replication scatter to:", out_fp, "\n")
