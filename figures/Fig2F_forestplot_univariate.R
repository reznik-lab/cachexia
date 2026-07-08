source("rev_code/Figures/figure_panel/Fig2F_setup.R")
library(ggplot2)
library(scales)

univariate_fixed_df <- get_all_coefficients(
  data = cac_fixed,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(time_mt, event) ~ cachexia,
  method = "univariate_fixed")

univariate_td_df <- get_all_coefficients(
  data = cac_time_dep,
  cancer_types_list = large_n_cancer_types$CANCER_TYPE_DETAILED,
  formula = Surv(start_mt, end_mt, span_event) ~ cachexia,
  method = "univariate_time_dep")

write.csv(univariate_fixed_df, file.path(TABLE_ROOT, "univariate_fixed_coefficients_raw.csv"), row.names = FALSE)
write.csv(univariate_td_df,    file.path(TABLE_ROOT, "univariate_time_dep_coefficients_raw.csv"), row.names = FALSE)

table_uni_td <- univariate_td_df %>%
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
write.csv(table_uni_td, file.path(TABLE_ROOT, "SI_Table3_OS_time_dep_univariate.csv"), row.names = FALSE)

univariate_fixed_df$Method <- "Fixed"
univariate_td_df$Method    <- "Time-Dependent"

uni_combined <- rbind(univariate_fixed_df, univariate_td_df) %>%
  dplyr::filter(Coefficient.Name == "cachexia") %>%
  dplyr::mutate(
    Hazard.Ratio = exp(Coefficient),
    Lower.CI.HR  = exp(Lower.CI),
    Upper.CI.HR  = exp(Upper.CI),
    Shape        = ifelse(Significant, 16, 1),
    Cancer.Code  = dplyr::recode(Cancer.Type, !!!code_map, .default = Cancer.Type)
  )

p_uni <- ggplot(uni_combined,
                aes(x = Hazard.Ratio,
                    y = reorder(Cancer.Code, Hazard.Ratio),
                    xmin = Lower.CI.HR, xmax = Upper.CI.HR,
                    color = Method)) +
  geom_vline(xintercept = c(0.5, 1, 2, 5, 10), linetype = "dotted", color = "grey80") +
  geom_vline(xintercept = 1, linetype = "solid", color = "grey40") +
  geom_pointrange(position = position_dodge(width = 0.3), size = 0.5) +
  geom_point(aes(shape = as.factor(Shape)), size = 0.5,
             position = position_dodge(width = 0.6)) +
  scale_shape_manual(values = c("1" = 1, "16" = 16), guide = "none") +
  scale_color_manual(values = c("Fixed" = "#8CC2CAFF", "Time-Dependent" = "#BB7693FF")) +
  scale_x_log10(breaks = c(0.5, 1, 2, 5, 10), limits = c(0.3, 15)) +
  labs(title = "Univariate", x = "HR", y = NULL, color = "Model") +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_text(hjust = 0.5, size = 9),
    axis.title.x = element_text(size = 9, margin = margin(t = 2)),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", size = 0.2),
    axis.ticks.x = element_line(color = "black", size = 0.3),
    axis.ticks.y = element_line(color = "black", size = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.title = element_text(size = 9),
    legend.text  = element_text(size = 9),
    legend.key.height = unit(3, "mm"),
    legend.key.width  = unit(6, "mm"),
    legend.box.margin = margin(t = -6, r = 0, b = 0, l = 0),
    legend.margin     = margin(t = -4, r = 0, b = 0, l = 0),
    plot.margin = margin(t = 5, r = 5, b = 0, l = 5)
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE))
p_uni

ggsave(file.path(PLOT_ROOT, "time_dep_cox_forestplot_uni.pdf"),
       plot = p_uni, width = 3.0, height = 3.2, dpi = 300)
