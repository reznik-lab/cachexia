source("rev_code/Figures/figure_panel/Fig3_setup.R")
library(ggplot2)

SFIG3_DIR <- file.path(dirname(FIG3_DIR), "SFig3")
dir.create(SFIG3_DIR, recursive = TRUE, showWarnings = FALSE)

# ------------------ Fig3D / SFig3A-B: cachexia near death (+/-30d), anchored timeline ------------------ #
# Fig3D (main) = consensus/WL5+modifier definition; SFig3 repeats this for WL10 and WL15

run_near_death <- function(eps, wl_label, out_dir, title_suffix) {
  cohort <- merge(
    clin_mrn[, .(MRN, anchor_final, CANCER_TYPE_DETAILED, PT_DEATH_DTE)],
    last_bmi,
    by = "MRN", all.x = TRUE
  )

  cohort <- cohort[
    !is.na(PT_DEATH_DTE) &
      !is.na(last_bmi_date) &
      last_bmi_date <= PT_DEATH_DTE &
      (PT_DEATH_DTE - last_bmi_date <= 30)
  ]

  cohort[, death_day := as.integer(PT_DEATH_DTE - anchor_final)]

  cohort <- cohort[trimws(as.character(CANCER_TYPE_DETAILED)) %in% valid_ctypes]
  cohort[, CANCER_TYPE_DETAILED := trimws(as.character(CANCER_TYPE_DETAILED))]
  cohort[, Cancer_Code := unname(code_map[CANCER_TYPE_DETAILED])]
  cohort[is.na(Cancer_Code), Cancer_Code := CANCER_TYPE_DETAILED]

  death_pt <- unique(cohort[, .(MRN, Cancer_Code, death_day)])

  ep_rows <- eps[
    has_cachexia_valid_edemaQC == 1L & !is.na(start_day) & !is.na(end_day),
    .(MRN, start_day = as.integer(start_day), end_day = as.integer(end_day))
  ]

  ep_rows <- merge(ep_rows, death_pt[, .(MRN, death_day)], by = "MRN", all.x = TRUE)

  ep_rows[, flag := as.integer(
    !is.na(death_day) &
      death_day >= start_day &
      death_day <= (end_day + 30L)
  )]

  ep_flag <- ep_rows[, .(Cachexia_At_Death = max(flag, na.rm = TRUE)), by = MRN]
  ep_flag[is.infinite(Cachexia_At_Death) | is.na(Cachexia_At_Death), Cachexia_At_Death := 0L]

  death_pt <- merge(death_pt, ep_flag, by = "MRN", all.x = TRUE)
  death_pt[is.na(Cachexia_At_Death), Cachexia_At_Death := 0L]

  plot_data_death <- death_pt[, .(
    n_patients       = .N,
    n_cachexia_death = sum(Cachexia_At_Death == 1L, na.rm = TRUE),
    prop             = mean(Cachexia_At_Death == 1L, na.rm = TRUE),
    se               = sqrt(mean(Cachexia_At_Death == 1L, na.rm = TRUE) *
                              (1 - mean(Cachexia_At_Death == 1L, na.rm = TRUE)) / .N)
  ), by = Cancer_Code]

  plot_data_death[, `:=`(
    lower = pmax(0, prop - 1.96 * se),
    upper = pmin(1, prop + 1.96 * se)
  )]

  plot_data_death <- plot_data_death[order(-prop)]
  plot_data_death[, Cancer_Code := factor(Cancer_Code, levels = plot_data_death$Cancer_Code)]
  plot_data_death[, `:=`(prop = 100 * prop, lower = 100 * lower, upper = 100 * upper)]

  p_death <- ggplot(plot_data_death, aes(x = Cancer_Code, y = prop)) +
    geom_bar(stat = "identity", fill = "black", width = 0.9) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = upper), color = "gray10", linewidth = 0.6) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = lower), color = "gray70", linewidth = 0.6) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.025, 0)) +
    labs(title = paste0("Episodes near death (+/-30 days)", title_suffix), x = "", y = "Percentage (%)") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 9),
      axis.text.x = element_text(size = 9, angle = 90, hjust = 1),
      axis.text.y = element_text(size = 9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(linewidth = 0.3)
    )

  ggsave(file.path(out_dir, paste0("fig3_ccx_near_death_", wl_label, "_", DATE_STAMP, ".pdf")),
         plot = p_death, width = 4.8, height = 2.6, dpi = 300, device = pdf)

  fwrite(plot_data_death[, .(Cancer_Code, n_patients, n_cachexia_death, prop, lower, upper)],
         file.path(REV_TABLES, paste0("fig3_ccx_near_death_", wl_label, "_per_cancer_summary_", DATE_STAMP, ".csv")))

  fwrite(data.table(
    n_deceased_with_lastBMI_le30d = uniqueN(death_pt$MRN),
    n_cachexia_near_death         = sum(death_pt$Cachexia_At_Death == 1L),
    pct_cachexia_near_death       = 100 * mean(death_pt$Cachexia_At_Death == 1L)
  ), file.path(REV_TABLES, paste0("fig3_ccx_near_death_", wl_label, "_overall_prevalence_", DATE_STAMP, ".csv")))

  death_dt <- merge(cohort, ep_flag, by = "MRN", all.x = TRUE)
  death_dt[is.na(Cachexia_At_Death), Cachexia_At_Death := 0L]
  fwrite(unique(death_dt[, .(MRN, CANCER_TYPE_DETAILED, Cancer_Code, last_bmi_date, death_day, Cachexia_At_Death)]),
         file.path(REV_TABLES, paste0("fig3_ccx_near_death_", wl_label, "_patient_level_", DATE_STAMP, ".csv")))

  cat(sprintf("\n=== NEAR-DEATH (+/-30d) | last BMI <=30d | WL%s ===\n", wl_label))
  cat("Eligible deceased (unique MRN):", uniqueN(death_pt$MRN), "\n")
  cat("Near-death positive (MRN):",      sum(death_pt$Cachexia_At_Death == 1L, na.rm = TRUE), "\n")
  cat(sprintf("Overall rate: %.2f%%\n",  100 * mean(death_pt$Cachexia_At_Death == 1L, na.rm = TRUE)))

  invisible(plot_data_death)
}

# Fig3D (main): consensus / WL5+modifier
plot_data_death_5 <- run_near_death(eps5,  "5",  FIG3_DIR,  "")
# SFig3 A/B: WL10 and WL15
run_near_death(eps10, "10", SFIG3_DIR, " (>=10% WL)")
run_near_death(eps15, "15", SFIG3_DIR, " (>=15% WL)")

# ------------------ Fig3 combined (WL5): near-death panel ordered by early-onset (dx) panel ------------------ #

fig3B_summary_fp <- file.path(REV_TABLES, paste0("fig3_ccx_180_5_stage13_untreated_per_cancer_summary_", DATE_STAMP, ".csv"))
if (file.exists(fig3B_summary_fp)) {
  dx_summary <- fread(fig3B_summary_fp)
  target_codes <- dx_summary[order(-prop)]$Cancer_Code

  plot_data_death2 <- copy(plot_data_death_5)
  plot_data_death2 <- plot_data_death2[as.character(Cancer_Code) %in% target_codes]
  plot_data_death2[, Cancer_Code := factor(as.character(Cancer_Code), levels = target_codes)]

  p2 <- ggplot(plot_data_death2, aes(x = Cancer_Code, y = prop)) +
    geom_bar(stat = "identity", fill = "black", width = 0.9) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = upper), color = "gray10", linewidth = 0.6) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = lower), color = "white", linewidth = 0.6) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.025, 0)) +
    labs(title = "Episodes near death (+/-30 days)", x = "", y = "Percentage (%)") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 9),
      axis.text.x = element_text(angle = 90, hjust = 1, size = 9),
      axis.text.y = element_text(size = 9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.3)
    )

  ggsave(file.path(FIG3_DIR, paste0("fig3_combined_near_death_ordered_by_dx_", DATE_STAMP, ".pdf")),
         plot = p2, width = 3.8, height = 2.6, dpi = 300, device = pdf)
}

# ------------------ SI Table: per-cancer prevalence + Fisher OR, across WL5/WL10/WL15 ------------------ #

build_neardeath_full_allcancers <- function(ep) {
  dt <- merge(
    clin_mrn[, .(MRN, anchor_final, CANCER_TYPE_DETAILED, PT_DEATH_DTE)],
    last_bmi,
    by = "MRN", all.x = TRUE
  )

  dt <- dt[
    !is.na(PT_DEATH_DTE) &
      !is.na(last_bmi_date) &
      last_bmi_date <= PT_DEATH_DTE &
      (PT_DEATH_DTE - last_bmi_date <= 30)
  ]

  dt[, death_day := as.integer(PT_DEATH_DTE - anchor_final)]
  add_cancer_code(dt)

  base <- unique(dt[, .(MRN, Cancer_Code, death_day)])

  ep_rows <- ep[has_cachexia_valid_edemaQC == 1L & !is.na(start_day) & !is.na(end_day),
                .(MRN, ep_start = as.integer(start_day), ep_end = as.integer(end_day))]

  ep_rows <- merge(ep_rows, base[, .(MRN, death_day)], by = "MRN", all.x = TRUE)
  ep_rows[, flag := as.integer(!is.na(death_day) & death_day >= ep_start & death_day <= (ep_end + 30L))]

  ep_flag <- ep_rows[, .(Cachexia_At_Death = max(flag, na.rm = TRUE)), by = MRN]
  ep_flag[is.infinite(Cachexia_At_Death) | is.na(Cachexia_At_Death), Cachexia_At_Death := 0L]

  out <- merge(base[, .(MRN, Cancer_Code)], ep_flag, by = "MRN", all.x = TRUE)
  out[is.na(Cachexia_At_Death), Cachexia_At_Death := 0L]
  out
}

prev_by_cancer_nd <- function(dt) {
  dt[, .(
    N_eligible = .N,
    N_pos      = sum(Cachexia_At_Death == 1L, na.rm = TRUE),
    Prev_pct   = 100 * mean(Cachexia_At_Death == 1L, na.rm = TRUE)
  ), by = Cancer_Code]
}

nd5_all  <- build_neardeath_full_allcancers(eps5)
nd10_all <- build_neardeath_full_allcancers(eps10)
nd15_all <- build_neardeath_full_allcancers(eps15)

or5_all  <- build_or_table(nd5_all,  "Cachexia_At_Death")
or10_all <- build_or_table(nd10_all, "Cachexia_At_Death")
or15_all <- build_or_table(nd15_all, "Cachexia_At_Death")

pr5_all  <- prev_by_cancer_nd(nd5_all);  setnames(pr5_all,  c("N_eligible","N_pos","Prev_pct"), c("N_elig_5","N_pos_5","Prev_5_pct"))
pr10_all <- prev_by_cancer_nd(nd10_all); setnames(pr10_all, c("N_eligible","N_pos","Prev_pct"), c("N_elig_10","N_pos_10","Prev_10_pct"))
pr15_all <- prev_by_cancer_nd(nd15_all); setnames(pr15_all, c("N_eligible","N_pos","Prev_pct"), c("N_elig_15","N_pos_15","Prev_15_pct"))

or5_all  <- or5_all [, .(Cancer_Code, OR_5=OR,  LCL_5=LCL,  UCL_5=UCL,  p_5=P,  FDR_5=FDR,  a_5=a,b_5=b,c_5=c,d_5=d)]
or10_all <- or10_all[, .(Cancer_Code, OR_10=OR, LCL_10=LCL, UCL_10=UCL, p_10=P, FDR_10=FDR, a_10=a,b_10=b,c_10=c,d_10=d)]
or15_all <- or15_all[, .(Cancer_Code, OR_15=OR, LCL_15=LCL, UCL_15=UCL, p_15=P, FDR_15=FDR, a_15=a,b_15=b,c_15=c,d_15=d)]

death_tab_all <- Reduce(function(x, y) merge(x, y, by = "Cancer_Code", all = TRUE),
                        list(pr5_all, pr10_all, pr15_all, or5_all, or10_all, or15_all))

death_tab_rep <- death_tab_all[Cancer_Code %in% report_codes]

death_tab_rep[, `:=`(
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

setorder(death_tab_rep, -Prev_5_pct)

fwrite(death_tab_rep, file.path(REV_TABLES, paste0("si_table3b_neardeath_OR_prev_WL5_WL10_WL15_", DATE_STAMP, ".csv")))

cat("\nNEAR-DEATH eligible (ALL cancers) n:", uniqueN(nd5_all$MRN), "\n")
cat(sprintf("WL5 overall:  %.2f%%\n", 100 * mean(nd5_all$Cachexia_At_Death == 1L)))
cat(sprintf("WL10 overall: %.2f%%\n", 100 * mean(nd10_all$Cachexia_At_Death == 1L)))
cat(sprintf("WL15 overall: %.2f%%\n", 100 * mean(nd15_all$Cachexia_At_Death == 1L)))
cat("\nWrote near-death outputs to:\n  ", FIG3_DIR, "\n  ", SFIG3_DIR, "\n  ", REV_TABLES, "\n")
