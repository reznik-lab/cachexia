library(data.table)
library(dplyr)
library(ggplot2)
library(ggpubr)

# SFig5: for a given cancer type + gene, % change in a lab value (episode start
# to episode end) for the patient's FIRST cachexia episode, split by MUT vs WT
# status for that gene. Needs RAW (non-z-scored) lab values.
# Adapted from 0303_ccx_revisions/rev_code/Figures/SFig5_genomics.R
# (run_sfig5_gene_serology_pctchange section, ~lines 340-537).

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

sfig_dir <- file.path(REV_PLOTS, "fearon_definition", "SFig5")
dir.create(sfig_dir, recursive = TRUE, showWarnings = FALSE)

labs_fp  <- file.path(REV_INPUTS, "labs_flattened_20260202.csv")
spans_fp <- file.path(REV_RESULTS, "spans_fearon_WL5_BMIlt20_20260706.csv")
msk_fp   <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
mutation_fp <- file.path(REV_INPUTS, "IMPACT_Oncogenic_Table_0919_2024.csv")

labs  <- fread(labs_fp)
spans <- fread(spans_fp)
msk   <- fread(msk_fp)
mutation <- fread(mutation_fp)
setnames(mutation, 1, "DMP_ID")
mutation[, DMP_ID := substr(DMP_ID, 1, 9)]
mutation <- mutation[!duplicated(DMP_ID)]

labs[, Date := as.IDate(Date)]
msk[, anchor_final := as.IDate(anchor_final)]

labs <- merge(labs, msk[, .(MRN, DMP_ID, anchor_final, CANCER_TYPE_DETAILED)], by = "MRN", all.x = TRUE)
labs <- labs[!is.na(anchor_final) & !is.na(Date)]
labs[, Days_Since_Dx := as.integer(Date - anchor_final)]

spans_keep <- unique(spans[, .(MRN, start_day, end_day, span)])
setkey(spans_keep, MRN); setkey(labs, MRN)
spans_labs <- merge(spans_keep, labs, by = "MRN", allow.cartesian = TRUE)

run_sfig5_gene_serology_pctchange <- function(cancer_type, gene, labs_to_pull = c("Albumin", "ALK", "HGB", "WBC"), win = 30) {
  if (!gene %in% names(mutation)) {
    cat(sprintf("[SKIP] Gene %s not found in mutation table\n", gene))
    return(invisible(NULL))
  }

  first_ep <- spans_labs[span == 1 & CANCER_TYPE_DETAILED == cancer_type,
                          .(start_day = min(start_day, na.rm = TRUE)), by = MRN]
  first_ep <- merge(first_ep, unique(spans_labs[span == 1, .(MRN, start_day, end_day)]),
                     by = c("MRN", "start_day"))
  first_ep <- first_ep[!duplicated(MRN)]

  gene_status <- merge(first_ep, msk[, .(MRN, DMP_ID)], by = "MRN", all.x = TRUE)
  gene_status <- merge(gene_status, mutation[, .(DMP_ID, mut = get(gene))], by = "DMP_ID", all.x = TRUE)
  gene_status[, mut := fifelse(is.na(mut), 0, fifelse(mut > 1, 1, mut))]
  gene_status[, gene_group := fifelse(mut == 1, paste0(gene, "_MUT"), paste0(gene, "_WT"))]

  ct_labs <- labs[CANCER_TYPE_DETAILED == cancer_type]

  results <- list()
  for (lab in labs_to_pull) {
    if (!lab %in% names(ct_labs)) next
    lab_vals <- ct_labs[!is.na(get(lab)), .(MRN, Days_Since_Dx, value = get(lab))]

    pull_nearest <- function(mrn, target_day) {
      sub <- lab_vals[MRN == mrn & abs(Days_Since_Dx - target_day) <= win]
      if (nrow(sub) == 0) return(NA_real_)
      sub[which.min(abs(Days_Since_Dx - target_day)), value]
    }

    dt <- copy(gene_status)
    dt[, start_val := mapply(pull_nearest, MRN, start_day)]
    dt[, end_val   := mapply(pull_nearest, MRN, end_day)]
    dt <- dt[!is.na(start_val) & !is.na(end_val) & start_val != 0]
    dt[, pct_change := 100 * (end_val - start_val) / start_val]
    dt[, test := lab]
    results[[lab]] <- dt[, .(MRN, gene_group, test, pct_change)]
  }

  out <- rbindlist(results, use.names = TRUE, fill = TRUE)
  if (nrow(out) == 0) {
    cat(sprintf("[SKIP] %s / %s: no matched lab pulls\n", cancer_type, gene))
    return(invisible(NULL))
  }
  out[, gene_group := factor(gene_group, levels = c(paste0(gene, "_WT"), paste0(gene, "_MUT")))]

  p <- ggplot(out, aes(x = gene_group, y = pct_change, fill = gene_group)) +
    geom_boxplot(outlier.size = 0.6, width = 0.6) +
    ggpubr::stat_compare_means(method = "wilcox.test", size = 2.8, label.y.npc = 0.95) +
    facet_wrap(~test, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = setNames(c("#6388B4FF", "#EF6F6AFF"), c(paste0(gene, "_WT"), paste0(gene, "_MUT")))) +
    labs(x = NULL, y = "% change (episode start to end)",
         title = paste0(gsub("[/ ]+", " ", cancer_type), ": ", gene)) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      strip.text   = element_text(size = 8),
      axis.title.y = element_text(size = 8),
      axis.text.x  = element_text(size = 7, angle = 30, hjust = 1),
      axis.text.y  = element_text(size = 8),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      panel.grid   = element_blank(),
      axis.ticks   = element_line(color = "black", linewidth = 0.3),
      legend.position = "none"
    )

  safe_ct <- gsub("[^A-Za-z0-9]", "_", cancer_type)
  out_fp <- file.path(sfig_dir, paste0("SFig5_", safe_ct, "_", gene, "_lab_pct_change_", DATE_STAMP, ".pdf"))
  ggsave(out_fp, p, width = 5, height = 2.6)
  fwrite(out, file.path(sfig_dir, paste0("SFig5_", safe_ct, "_", gene, "_lab_pct_change_data_", DATE_STAMP, ".csv")))
  cat(sprintf("[SAVED] %s (n=%d rows)\n", out_fp, nrow(out)))
}

run_sfig5_gene_serology_pctchange(cancer_type = "Bladder Urothelial Carcinoma", gene = "TP53")

cat("\nWrote SFig5 gene-serology %change boxplot to:", sfig_dir, "\n")
