library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(forcats)

# SFig4A (ALK, high>147 U/L), SFig4B (NLR, out of cohort-derived non-episode
# reference interval), SFig4C (CRP, high>10 mg/L) - pan-cancer Outside/During
# abnormal-value proportions. Needs RAW (non-z-scored) values.
# ALK/NLR from routine labs (raw flattened merge, same pattern as Fig4E);
# CRP from additional labs (already raw in the additional_labs_merge.py output).
# Adapted from 0303_ccx_revisions/rev_code/Figures/Fig4.R (ALK/CRP sections).
# NLR's original reference-interval computation code was lost from the 0303
# codebase (only the plotting snippet survived, referencing an undefined
# df_nlr) - reconstructed here using the standard clinical-lab convention: a
# 95% reference interval (2.5th-97.5th percentile) built from the "Outside"
# (non-episode) NLR distribution, per the manuscript legend's own description
# ("NLR values outside the non-episode reference interval").

BASE_REV    <- "."
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "SFig4")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fill_abn <- c("Above threshold" = "#EF6F6AFF", "In-range" = "#A9B5AEFF")
fill_oor <- c("Out of range" = "#EF6F6AFF", "In-range" = "#A9B5AEFF")

make_prop_plot <- function(df, title, fill_values, legend_labels) {
  ggplot(df, aes(x = span_lab, y = prop, fill = label)) +
    geom_bar(stat = "identity", position = position_stack(reverse = TRUE), width = 0.9) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                        labels = number_format(accuracy = 0.01), expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.5, 0)) +
    scale_fill_manual(values = fill_values) +
    guides(fill = guide_legend(reverse = TRUE)) +
    labs(title = title, x = "", y = "Proportion", fill = "") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_text(hjust = 0.5, size = 9),
      axis.title   = element_text(size = 9),
      axis.text    = element_text(size = 9),
      legend.text  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      panel.grid   = element_blank(),
      axis.ticks   = element_line(color = "black", linewidth = 0.3),
      legend.position = "top"
    )
}

# ------------------ Raw routine-labs merge (ALK + NLR components) ------------------

labs_fp  <- file.path(REV_INPUTS, "labs_flattened_20260202.csv")
spans_fp <- file.path(REV_RESULTS, "spans_fearon_WL5_BMIlt20_20260706.csv")
msk_fp   <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")

labs  <- fread(labs_fp)
spans <- fread(spans_fp)
msk   <- fread(msk_fp)

labs[, Date := as.IDate(Date)]
msk[, anchor_final := as.IDate(anchor_final)]

labs <- merge(labs, msk[, .(MRN, anchor_final, CANCER_TYPE_DETAILED, GENDER)], by = "MRN", all.x = TRUE)
labs <- labs[!is.na(anchor_final) & !is.na(Date)]
labs[, Days_Since_Dx := as.integer(Date - anchor_final)]

spans_keep <- unique(spans[, .(MRN, start_day, end_day, span)])
setkey(spans_keep, MRN)
setkey(labs, MRN)

spans_labs <- merge(spans_keep, labs, by = "MRN", allow.cartesian = TRUE)
spans_labtests <- spans_labs[!is.na(Days_Since_Dx) & Days_Since_Dx >= start_day & Days_Since_Dx <= end_day]

spans_labtests[, span_lab := factor(fifelse(span == 0, "Outside", "During"), levels = c("Outside", "During"))]

# ------------------ SFig4A: ALK (high > 147 U/L) ------------------

alk_high <- 147

df_alk <- spans_labtests[!is.na(ALK), .(span_lab, label = fifelse(ALK > alk_high, "Above threshold", "In-range"))
][, .(n = .N), by = .(span_lab, label)][, prop := n / sum(n), by = span_lab]
df_alk[, label := factor(label, levels = c("Above threshold", "In-range"))]

p_alk <- make_prop_plot(df_alk, "Pan-cancer: ALK", fill_abn)
ggsave(file.path(out_dir, paste0("sfig4a_alk_pan_cancer_", DATE_STAMP, ".pdf")), p_alk, width = 2.3, height = 2.4, device = "pdf")
fwrite(df_alk, file.path(out_dir, paste0("sfig4a_alk_pan_cancer_data_", DATE_STAMP, ".csv")))

# ------------------ SFig4B: NLR (out of cohort non-episode reference interval) ------------------

nlr_dt <- spans_labtests[!is.na(Neut) & !is.na(Lymph) & Lymph != 0, .(MRN, span_lab, NLR = Neut / Lymph)]

ref_bounds <- nlr_dt[span_lab == "Outside", .(lo = quantile(NLR, 0.025, na.rm = TRUE),
                                               hi = quantile(NLR, 0.975, na.rm = TRUE))]
cat(sprintf("NLR non-episode 95%% reference interval: [%.3f, %.3f]\n", ref_bounds$lo, ref_bounds$hi))

df_nlr <- nlr_dt[, .(span_lab, label = fifelse(NLR < ref_bounds$lo | NLR > ref_bounds$hi, "Out of range", "In-range"))
][, .(n = .N), by = .(span_lab, label)][, prop := n / sum(n), by = span_lab]
df_nlr[, label := factor(label, levels = c("Out of range", "In-range"))]

p_nlr <- make_prop_plot(df_nlr, "Pan-cancer: NLR", fill_oor)
ggsave(file.path(out_dir, paste0("sfig4b_nlr_pan_cancer_", DATE_STAMP, ".pdf")), p_nlr, width = 2.3, height = 2.4, device = "pdf")
fwrite(df_nlr, file.path(out_dir, paste0("sfig4b_nlr_pan_cancer_data_", DATE_STAMP, ".csv")))

# ------------------ SFig4C: CRP (high > 10 mg/L) ------------------

crp_hi <- 10
addlabs_fp <- file.path(REV_RESULTS, "serology_spans_labs", paste0("spans_additionallabs_WL5_BMIlt20_", DATE_STAMP, ".csv"))
labs_span <- fread(addlabs_fp)

labs_span[, span01 := suppressWarnings(as.integer(as.character(span)))]
labs_span[is.na(span01), span01 := suppressWarnings(as.integer(span))]
labs_span[, span_lab := factor(fifelse(span01 == 0, "Outside", "During"), levels = c("Outside", "During"))]

df_crp <- labs_span[
  CLEANED_TEST_NAME == "C-Reactive Protein" & is.finite(LR_RESULT_VALUE) & !is.na(span01),
  .(span_lab, label = fifelse(LR_RESULT_VALUE > crp_hi, "Above threshold", "In-range"))
][, .(n = .N), by = .(span_lab, label)][, prop := n / sum(n), by = span_lab]
df_crp[, label := factor(label, levels = c("Above threshold", "In-range"))]

p_crp <- make_prop_plot(df_crp, "Pan-cancer: CRP", fill_abn)
ggsave(file.path(out_dir, paste0("sfig4c_crp_pan_cancer_", DATE_STAMP, ".pdf")), p_crp, width = 2.3, height = 2.4, device = "pdf")
fwrite(df_crp, file.path(out_dir, paste0("sfig4c_crp_pan_cancer_data_", DATE_STAMP, ".csv")))

cat("\nWrote SFig4 A-C (pan-cancer abnormal-value proportions) to:", out_dir, "\n")
