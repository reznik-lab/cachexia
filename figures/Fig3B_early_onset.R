source("rev_code/Figures/figure_panel/Fig3_setup.R")
library(ggplot2)

SFIG3_DIR <- file.path(dirname(FIG3_DIR), "SFig3")
dir.create(SFIG3_DIR, recursive = TRUE, showWarnings = FALSE)


run_early_onset <- function(eps, wl_label, out_dir, title_suffix) {
  first_ep <- merge(eps, clin_mrn[, .(MRN, CANCER_TYPE_DETAILED, early_stage)], by = "MRN", all.x = TRUE)
  first_ep <- merge(first_ep, tx_first[, .(MRN, first_tx_day)], by = "MRN", all.x = TRUE)

  first_ep[, Cachexia_180 := (has_cachexia_valid_edemaQC == 1L) & !is.na(start_day) & start_day <= 180L]

  first_ep <- first_ep[early_stage == TRUE & !is.na(first_tx_day) & first_tx_day > 0L]
  first_ep <- first_ep[CANCER_TYPE_DETAILED %in% valid_ctypes]

  first_ep[, Cancer_Code := code_map[CANCER_TYPE_DETAILED]]
  first_ep[is.na(Cancer_Code), Cancer_Code := CANCER_TYPE_DETAILED]

  first_ep_pt <- first_ep[, .(
    Cachexia_180 = as.integer(any(Cachexia_180 == 1L, na.rm = TRUE))
  ), by = .(MRN, Cancer_Code)]

  plot_data <- first_ep_pt[, .(
    n_patients     = .N,
    n_cachexia_180 = sum(Cachexia_180),
    prop           = mean(Cachexia_180),
    se             = sqrt(mean(Cachexia_180) * (1 - mean(Cachexia_180)) / .N)
  ), by = Cancer_Code]

  plot_data[, `:=`(lower = pmax(0, prop - 1.96 * se), upper = pmin(1, prop + 1.96 * se))]
  plot_data <- plot_data[order(-prop)]
  plot_data[, Cancer_Code := factor(Cancer_Code, levels = plot_data$Cancer_Code)]
  plot_data[, `:=`(prop = 100 * prop, lower = 100 * lower, upper = 100 * upper)]

  p_dx <- ggplot(plot_data, aes(x = Cancer_Code, y = prop)) +
    geom_bar(stat = "identity", fill = "black", width = 0.9) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = upper), color = "gray10", linewidth = 0.6) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = lower), color = "white", linewidth = 0.6) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.025, 0)) +
    labs(title = paste0("Episodes within the first 180 days\n(Stage 1-3, untreated at diagnosis)", title_suffix),
         x = "", y = "Percentage (%)") +
    theme_minimal(base_size = 9, base_family = "ArialMT") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 9),
      axis.text.x = element_text(size = 9, angle = 90, hjust = 1),
      axis.text.y = element_text(size = 9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(linewidth = 0.3)
    )

  ggsave(file.path(out_dir, paste0("fig3_ccx_180_", wl_label, "_stage13_untreated_", DATE_STAMP, ".pdf")),
         plot = p_dx, width = 4.8, height = 2.6, dpi = 300, device = pdf)

  fwrite(plot_data[, .(Cancer_Code, n_patients, n_cachexia_180, prop, lower, upper)],
         file.path(REV_TABLES, paste0("fig3_ccx_180_", wl_label, "_stage13_untreated_per_cancer_summary_", DATE_STAMP, ".csv")))

  fwrite(data.table(
    n_patients_eligible = uniqueN(first_ep_pt$MRN),
    n_cachexia_le180    = sum(first_ep_pt$Cachexia_180 == 1L),
    pct_cachexia_le180  = 100 * mean(first_ep_pt$Cachexia_180 == 1L)
  ), file.path(REV_TABLES, paste0("fig3_ccx_180_", wl_label, "_stage13_untreated_overall_prevalence_", DATE_STAMP, ".csv")))

  first_ep_pt2 <- merge(
    first_ep_pt,
    unique(first_ep[, .(MRN, CANCER_TYPE_DETAILED, early_stage, first_tx_day)]),
    by = "MRN", all.x = TRUE
  )
  fwrite(first_ep_pt2[, .(MRN, CANCER_TYPE_DETAILED, Cancer_Code, early_stage, first_tx_day, Cachexia_180)],
         file.path(REV_TABLES, paste0("fig3_ccx_180_", wl_label, "_stage13_untreated_patient_level_", DATE_STAMP, ".csv")))

  cat(sprintf("\n=== EARLY-ONSET (<=180d) | stage 1-3 | untreated | WL%s ===\n", wl_label))
  cat("Eligible patients (unique MRN):", uniqueN(first_ep_pt$MRN), "\n")
  cat("Early-onset positive (MRN):",    sum(first_ep_pt$Cachexia_180 == 1L, na.rm = TRUE), "\n")
  cat(sprintf("Overall rate: %.2f%%\n", 100 * mean(first_ep_pt$Cachexia_180 == 1L, na.rm = TRUE)))

  invisible(plot_data)
}

plot_data_5 <- run_early_onset(eps5,  "5",  FIG3_DIR,  "")
run_early_onset(eps10, "10", SFIG3_DIR, " (>=10% WL)")
run_early_onset(eps15, "15", SFIG3_DIR, " (>=15% WL)")


build_first_ep_full_allcancers <- function(ep) {
  fe <- merge(
    clin_mrn[, .(MRN, CANCER_TYPE_DETAILED, early_stage)],
    tx_first[, .(MRN, first_tx_day)],
    by = "MRN", all.x = TRUE
  )
  fe <- fe[early_stage == TRUE & !is.na(first_tx_day) & first_tx_day > 0L]
  add_cancer_code(fe)

  ep_any <- ep[has_cachexia_valid_edemaQC == 1L & !is.na(start_day),
               .(Early_180 = as.integer(any(as.integer(start_day) <= 180L))),
               by = MRN]

  fe <- merge(fe, ep_any, by = "MRN", all.x = TRUE)
  fe[is.na(Early_180), Early_180 := 0L]
  fe
}

prev_by_cancer <- function(fe_dt) {
  fe_dt[, .(
    N_eligible = .N,
    N_pos      = sum(Early_180 == 1L, na.rm = TRUE),
    Prev_pct   = 100 * mean(Early_180 == 1L, na.rm = TRUE)
  ), by = Cancer_Code]
}

fe5_all  <- build_first_ep_full_allcancers(eps5)
fe10_all <- build_first_ep_full_allcancers(eps10)
fe15_all <- build_first_ep_full_allcancers(eps15)

or5_all  <- build_or_table(fe5_all,  "Early_180")
or10_all <- build_or_table(fe10_all, "Early_180")
or15_all <- build_or_table(fe15_all, "Early_180")

pr5_all  <- prev_by_cancer(fe5_all);  setnames(pr5_all,  c("N_eligible","N_pos","Prev_pct"), c("N_elig_5","N_pos_5","Prev_5_pct"))
pr10_all <- prev_by_cancer(fe10_all); setnames(pr10_all, c("N_eligible","N_pos","Prev_pct"), c("N_elig_10","N_pos_10","Prev_10_pct"))
pr15_all <- prev_by_cancer(fe15_all); setnames(pr15_all, c("N_eligible","N_pos","Prev_pct"), c("N_elig_15","N_pos_15","Prev_15_pct"))

or5_all  <- or5_all [, .(Cancer_Code, OR_5=OR,  LCL_5=LCL,  UCL_5=UCL,  p_5=P,  FDR_5=FDR,  a_5=a,b_5=b,c_5=c,d_5=d)]
or10_all <- or10_all[, .(Cancer_Code, OR_10=OR, LCL_10=LCL, UCL_10=UCL, p_10=P, FDR_10=FDR, a_10=a,b_10=b,c_10=c,d_10=d)]
or15_all <- or15_all[, .(Cancer_Code, OR_15=OR, LCL_15=LCL, UCL_15=UCL, p_15=P, FDR_15=FDR, a_15=a,b_15=b,c_15=c,d_15=d)]

early_tab_all <- Reduce(function(x, y) merge(x, y, by = "Cancer_Code", all = TRUE),
                        list(pr5_all, pr10_all, pr15_all, or5_all, or10_all, or15_all))

early_tab_rep <- early_tab_all[Cancer_Code %in% report_codes]

early_tab_rep[, `:=`(
  Prev_5_pct  = round(Prev_5_pct,  1),
  Prev_10_pct = round(Prev_10_pct, 1),
  Prev_15_pct = round(Prev_15_pct, 1),
  OR_5  = round(OR_5,  2), LCL_5  = round(LCL_5,  2), UCL_5  = round(UCL_5,  2),
  OR_10 = round(OR_10, 2), LCL_10 = round(LCL_10, 2), UCL_10 = round(UCL_10, 2),
  OR_15 = round(OR_15, 2), LCL_15 = round(LCL_15, 2), UCL_15 = round(UCL_15, 2),
  p_5   = signif(p_5,   3), FDR_5  = signif(FDR_5,  3),
  p_10  = signif(p_10,  3), FDR_10 = signif(FDR_10, 3),
  p_15  = signif(p_15,  3), FDR_15 = signif(FDR_15, 3)
)]

setorder(early_tab_rep, -Prev_5_pct)

fwrite(early_tab_rep, file.path(REV_TABLES, paste0("si_table3_early180_OR_prev_WL5_WL10_WL15_", DATE_STAMP, ".csv")))

cat("\nWrote early-onset outputs to:\n  ", FIG3_DIR, "\n  ", SFIG3_DIR, "\n  ", REV_TABLES, "\n")
