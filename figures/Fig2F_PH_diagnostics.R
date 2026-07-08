# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
source("rev_code/Figures/figure_panel/Fig2F_setup.R")

vanilla_cox <- coxph(Surv(time_mt, event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
                     data = cac_fixed)
print(summary(vanilla_cox))

time_dep_cox <- coxph(
  Surv(start_mt, end_mt, span_event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
  data = cac_time_dep,
  cluster = MRN
)
print(summary(time_dep_cox))

ph_td <- cox.zph(time_dep_cox, transform = "km")
print(ph_td)

ph_td_df <- data.frame(
  term  = rownames(ph_td$table),
  chisq = ph_td$table[, "chisq"],
  df    = ph_td$table[, "df"],
  p     = ph_td$table[, "p"]
)
write.csv(ph_td_df, file.path(TABLE_ROOT, "PHcheck_time_dep_cox_pan_cancer.csv"), row.names = FALSE)

pdf(file.path(PLOT_ROOT, "PHcheck_time_dep_cox_pan_cancer_schoenfeld.pdf"), width = 7, height = 6)
plot(ph_td)
dev.off()
