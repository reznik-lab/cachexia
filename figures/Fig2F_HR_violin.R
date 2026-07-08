source("rev_code/Figures/figure_panel/Fig2F_setup.R")
library(ggplot2)
library(ggpubr)

multivariate_fixed_df <- get_all_coefficients(
  data = cac_fixed,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(time_mt, event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
  method = "fixed"
)

multivariate_timedep_df <- get_all_coefficients(
  data = cac_time_dep,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(start_mt, end_mt, span_event) ~ cachexia + GENDER_NUM + age_at_diagnosis + stage_gran,
  method = "time_dep_cac"
)

multivariate_fixed_df$Method    <- "Fixed"
multivariate_timedep_df$Method  <- "Time-Dependent"

hazard_df <- rbind(multivariate_fixed_df, multivariate_timedep_df) %>% filter(Coefficient.Name == "cachexia")
hazard_df$Hazard.Ratio <- exp(hazard_df$Coefficient)

p_compare <- ggplot(hazard_df, aes(x = Method, y = Hazard.Ratio, color = Method)) +
  geom_violin() + geom_boxplot(width = 0.1) +
  scale_y_continuous(breaks = c(0, 1, 5, 10, 15)) +
  stat_compare_means(method = "wilcox.test", paired = TRUE, size = 8) +
  labs(title = "Fixed vs Time Dependent Hazard Ratio", y = "Hazard Ratio") +
  theme(legend.position = "none", text = element_text(size = 20))

p_compare

ggsave(file.path(PLOT_ROOT, "fixed_vs_time_dep_HR_violin.pdf"),
       plot = p_compare, width = 4.5, height = 3.5, dpi = 300)
