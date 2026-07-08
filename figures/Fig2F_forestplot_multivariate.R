# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
source("rev_code/Figures/figure_panel/Fig2F_setup.R")
library(ggplot2)
library(scales)

multivariate_fixed_df <- get_all_coefficients(
  data = cac_fixed,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(time_mt, event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
  method = "fixed"
)

err_multivariate_timedep_df <- get_all_coefficients(
  data = cac_spans,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(start_mt, end_mt, span_event) ~ cachexia_status + GENDER_NUM + age_at_diagnosis + stage_gran,
  method = "err_time_dep_cac"
)

multivariate_timedep_df <- get_all_coefficients(
  data = cac_time_dep,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(start_mt, end_mt, span_event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
  method = "time_dep_cac"
)

write.csv(multivariate_fixed_df,       file.path(TABLE_ROOT, "multivariable_fixed_coefficients_raw.csv"), row.names = FALSE)
write.csv(multivariate_timedep_df,     file.path(TABLE_ROOT, "multivariable_time_dep_coefficients_raw.csv"), row.names = FALSE)
write.csv(err_multivariate_timedep_df, file.path(TABLE_ROOT, "multivariable_time_dep_cachexia_status_coefficients_raw.csv"), row.names = FALSE)

table1 <- multivariate_timedep_df %>%
  dplyr::filter(Coefficient.Name == "cachexia") %>%
  dplyr::mutate(
    CANCER_CODE = dplyr::recode(Cancer.Type, !!!code_map, .default = Cancer.Type),
    HR  = exp(Coefficient),
    LCL = exp(Lower.CI),
    UCL = exp(Upper.CI),
    HR_95CI = sprintf("%.2f (%.2f–%.2f)", HR, LCL, UCL),
    P   = signif(P.Value, 3),
    FDR = signif(P.Adjust, 3)
  ) %>% select(CANCER_CODE, HR_95CI, P, FDR) %>% arrange(CANCER_CODE)
write.csv(table1, file.path(TABLE_ROOT, "SI_Table2_OS_time_dep_multivariate.csv"), row.names = FALSE)

multivariate_fixed_df$Method   <- "Fixed"
multivariate_timedep_df$Method <- "Time-Dependent"

multi_combined <- rbind(multivariate_fixed_df, multivariate_timedep_df) %>%
  filter(Coefficient.Name == "cachexia") %>%
  mutate(
    Hazard.Ratio = exp(Coefficient),
    Lower.CI.HR  = exp(Lower.CI),
    Upper.CI.HR  = exp(Upper.CI),
    Shape        = ifelse(Significant, 16, 1))

multi_combined$Cancer.Type <- code_map[multi_combined$Cancer.Type]

p_multi <- ggplot(multi_combined,
                  aes(x = Hazard.Ratio,
                      y = reorder(Cancer.Type, Hazard.Ratio),
                      xmin = Lower.CI.HR,
                      xmax = Upper.CI.HR,
                      color = Method)) +
  geom_vline(xintercept = c(0.5, 1, 2, 5, 10), linetype = "dotted", color = "grey80") +
  geom_vline(xintercept = 1, linetype = "solid", color = "grey40") +
  geom_pointrange(position = position_dodge(width = 0.3), size = 0.5) +
  geom_point(aes(shape = as.factor(Shape)), size = 0.5,
             position = position_dodge(width = 0.6)) +
  scale_shape_manual(values = c("1" = 1, "16" = 16), guide = "none") +
  scale_color_manual(values = c("Fixed" = "#8CC2CAFF", "Time-Dependent" = "#BB7693FF")) +
  scale_x_log10(breaks = c(0.5, 1, 2, 5, 10), limits = c(0.3, 15)) +
  labs(title = "Multivariate Cox Models ", x = "HR", y = "Cancer Type", color = "Model Type") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", size = 0.2),
    axis.ticks.x = element_line(color = "black", size = 0.3),
    axis.ticks.y = element_line(color = "black", size = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )
p_multi

ggsave(file.path(PLOT_ROOT, "time_dep_cox_forestplot_multi.pdf"),
       plot = p_multi, width = 3.2, height = 3.8, dpi = 300)
