library(ggplot2)
library(data.table)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
REV_INPUTS  <- file.path(BASE_REV, "rev_inputs")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "SFig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if ("showtext" %in% loadedNamespaces()) showtext::showtext_auto(FALSE)

msk_clin_fp <- file.path(REV_INPUTS, "dx_cohort_metadata_20260126_v2.csv")
msk_clin <- fread(msk_clin_fp)
msk_clin[, MRN := as.integer(MRN)]

fp10 <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL10_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
fp15 <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL15_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))

eps10 <- fread(fp10); eps10[, MRN := as.integer(MRN)]
eps15 <- fread(fp15); eps15[, MRN := as.integer(MRN)]

cohort_mrns <- unique(msk_clin$MRN)
total_patients <- uniqueN(cohort_mrns)

has10 <- eps10[MRN %in% cohort_mrns & has_cachexia_valid_edemaQC == 1 & !is.na(start_day), uniqueN(MRN)]
has15 <- eps15[MRN %in% cohort_mrns & has_cachexia_valid_edemaQC == 1 & !is.na(start_day), uniqueN(MRN)]

plot_df <- data.table(
  Group = factor(c("≥10% WL", "≥15% WL"), levels = c("≥10% WL", "≥15% WL")),
  Count = c(has10, has15)
)
plot_df[, Proportion := Count / total_patients * 100]
plot_df[, SE := sqrt((Proportion/100) * (1 - Proportion/100) / total_patients) * 100]
plot_df[, Lower := pmax(0,   Proportion - 1.96 * SE)]
plot_df[, Upper := pmin(100, Proportion + 1.96 * SE)]

p_wl10_15 <- ggplot(plot_df, aes(x = Group, y = Proportion)) +
  geom_col(width = 0.9, fill = "black") +
  geom_segment(aes(xend = Group, y = Proportion, yend = Upper),
               color = "gray10", linewidth = 0.7) +
  geom_segment(aes(xend = Group, y = Proportion, yend = Lower),
               color = "white", linewidth = 0.7) +
  labs(x = NULL, y = "Percentage (%)", title = NULL) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    axis.ticks.x = element_line(color = "black", linewidth = 0.3),
    axis.ticks.y = element_line(color = "black", linewidth = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

print(p_wl10_15)

ggsave(file.path(out_dir, "fig2B_cachexia_WL10_WL15_prevalence.pdf"),
       plot = p_wl10_15, width = 2, height = 2.1, dpi = 300)
fwrite(plot_df, file.path(out_dir, "fig2B_cachexia_WL10_WL15_prevalence.csv"))

cat(sprintf("WL10: %d/%d (%.2f%%)\n", has10, total_patients, 100*has10/total_patients))
cat(sprintf("WL15: %d/%d (%.2f%%)\n", has15, total_patients, 100*has15/total_patients))

col_map <- c("10% WL" = "#D9A3B5FF", "15% WL" = "#8F3F63FF")

make_prev_dt <- function(eps_dt, threshold_label) {
  eps_mrn <- eps_dt[, .(has_cachexia = as.integer(max(has_cachexia_valid_edemaQC, na.rm = TRUE))), by = MRN]
  eps_mrn[is.infinite(has_cachexia) | is.na(has_cachexia), has_cachexia := 0L]

  dt <- merge(msk_clin[, .(MRN, CANCER_TYPE_DETAILED, ONCOTREE_CODE)], eps_mrn, by = "MRN", all.x = TRUE)
  dt[is.na(has_cachexia), has_cachexia := 0L]
  dt <- dt[!is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED))]
  dt <- dt[!is.na(ONCOTREE_CODE) & nzchar(trimws(ONCOTREE_CODE))]

  dt[, ONCOTREE_GROUP := ONCOTREE_CODE]
  dt[ONCOTREE_GROUP %chin% c("COAD", "READ"), ONCOTREE_GROUP := "COADREAD"]

  prev_dt <- dt[, .(Total = uniqueN(MRN), Cachectic = uniqueN(MRN[has_cachexia == 1])), by = ONCOTREE_GROUP]
  prev_dt[, Proportion := 100 * Cachectic / Total]
  prev_dt[, SE := sqrt((Proportion/100) * (1 - Proportion/100) / Total) * 100]
  prev_dt[, Lower := pmax(0,   Proportion - 1.96 * SE)]
  prev_dt[, Upper := pmin(100, Proportion + 1.96 * SE)]
  prev_dt[, Threshold := threshold_label]
  prev_dt[]
}

prev10 <- make_prev_dt(eps10, "10% WL")
prev15 <- make_prev_dt(eps15, "15% WL")

base_dt <- msk_clin[, .(MRN, CANCER_TYPE_DETAILED, ONCOTREE_CODE)]
base_dt <- base_dt[!is.na(CANCER_TYPE_DETAILED) & nzchar(trimws(CANCER_TYPE_DETAILED))]
base_dt <- base_dt[!is.na(ONCOTREE_CODE) & nzchar(trimws(ONCOTREE_CODE))]
base_dt[, ONCOTREE_GROUP := ONCOTREE_CODE]
base_dt[ONCOTREE_GROUP %chin% c("COAD", "READ"), ONCOTREE_GROUP := "COADREAD"]

eligible <- base_dt[, .(Total = uniqueN(MRN)), by = ONCOTREE_GROUP][Total >= 500]
eligible_ct <- eligible$ONCOTREE_GROUP

prev10 <- prev10[ONCOTREE_GROUP %chin% eligible_ct]
prev15 <- prev15[ONCOTREE_GROUP %chin% eligible_ct]

setorder(prev10, -Proportion)
x_levels <- prev10$ONCOTREE_GROUP

prev_all <- rbindlist(list(prev10, prev15), use.names = TRUE, fill = TRUE)
prev_all[, ONCOTREE_GROUP := factor(ONCOTREE_GROUP, levels = x_levels)]
prev_all[, Threshold := factor(Threshold, levels = c("10% WL", "15% WL"))]

pd <- position_dodge(width = 0.8)

p_sfig2h <- ggplot(prev_all, aes(x = ONCOTREE_GROUP, y = Proportion, fill = Threshold)) +
  geom_col(position = pd, width = 0.8, color = NA) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = pd, width = 0, linewidth = 0.45, color = "black") +
  scale_fill_manual(values = col_map) +
  labs(x = NULL, y = "Prevalence (%)", title = NULL, fill = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9, angle = 90, hjust = 1, vjust = 1),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    axis.ticks   = element_line(color = "black", linewidth = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "top",
    legend.text      = element_text(size = 9),
    legend.key.height = unit(0.35, "cm"),
    legend.key.width  = unit(0.55, "cm"),
    plot.margin = margin(5.5, 5.5, 5.5, 5.5)
  )

print(p_sfig2h)

ggsave(filename = file.path(out_dir, paste0("SFig2H_prevalence_WL10_vs_WL15_", DATE_STAMP, ".pdf")),
       plot = p_sfig2h, width = 5.1, height = 2.8, dpi = 300, useDingbats = FALSE)
fwrite(prev_all[order(ONCOTREE_GROUP, Threshold)], file.path(out_dir, paste0("SFig2H_prevalence_WL10_vs_WL15_", DATE_STAMP, ".csv")))
