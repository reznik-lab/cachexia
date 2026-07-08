source("rev_code/Figures/figure_panel/Fig2F_setup.R")
library(ggplot2)

DATE_STAMP <- "20260706"

eps10_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL10_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps15_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL15_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))

eps10 <- fread(eps10_fp)
eps15 <- fread(eps15_fp)
eps10[, MRN := as.integer(MRN)]
eps15[, MRN := as.integer(MRN)]

PLOT_ROOT <- file.path(REV_PLOTS, "fearon_definition", "SFig2", "time_dep_cox_WL10_WL15")
dir.create(PLOT_ROOT, recursive = TRUE, showWarnings = FALSE)


get_first_start <- function(eps_dt, tag_label) {
  eps_dt <- copy(eps_dt)
  stopifnot(all(c("MRN", "has_cachexia_valid_edemaQC", "start_day") %chin% names(eps_dt)))
  eps_valid <- eps_dt[has_cachexia_valid_edemaQC == 1 & !is.na(start_day)]
  first_dt <- eps_valid[, .(t_first = suppressWarnings(min(start_day, na.rm = TRUE))), by = MRN]
  first_dt[!is.finite(t_first), t_first := NA_real_]
  setnames(first_dt, "t_first", paste0("t_first_", tag_label))
  first_dt
}

first10 <- get_first_start(eps10, "WL10")
first15 <- get_first_start(eps15, "WL15")

cac_spans <- merge(cac_spans, first10, by = "MRN", all.x = TRUE)
cac_spans <- merge(cac_spans, first15, by = "MRN", all.x = TRUE)

cac_spans[, exp_WL10 := as.integer(!is.na(t_first_WL10) & start_time >= t_first_WL10)]
cac_spans[, exp_WL15 := as.integer(!is.na(t_first_WL15) & start_time >= t_first_WL15)]


run_per_cancer <- function(dt, ct_list, formula) {
  out <- data.frame()
  for (ct in ct_list) {
    d <- dt[dt$CANCER_TYPE_DETAILED == ct, , drop = FALSE]
    if (nrow(d) == 0) next

    fit <- coxph(formula, data = d, cluster = MRN)
    sm <- summary(fit)$coefficients
    ci_log <- confint(fit)

    tmp <- data.frame(
      Cancer.Type      = ct,
      Coefficient.Name = rownames(sm),
      Coefficient      = as.numeric(sm[, "coef"]),
      Lower.CI         = as.numeric(ci_log[, 1]),
      Upper.CI         = as.numeric(ci_log[, 2]),
      P.Value          = as.numeric(sm[, "Pr(>|z|)"]),
      stringsAsFactors = FALSE
    )
    out <- rbind(out, tmp)
  }

  out <- out %>%
    group_by(Coefficient.Name) %>%
    mutate(P.Adjust = p.adjust(P.Value, method = "BH")) %>%
    ungroup() %>%
    mutate(Significant = (P.Adjust < 0.05))

  out
}

ct_list <- large_n_cancer_types$CANCER_TYPE_DETAILED

uni_df <- run_per_cancer(
  dt      = as.data.frame(cac_spans),
  ct_list = ct_list,
  formula = Surv(start_mt, end_mt, span_event) ~ exp_WL10 + exp_WL15
)

multi_df <- run_per_cancer(
  dt      = as.data.frame(cac_spans),
  ct_list = ct_list,
  formula = Surv(start_mt, end_mt, span_event) ~ exp_WL10 + exp_WL15 + GENDER_NUM + age_at_diagnosis + stage_gran
)

write.csv(uni_df,   file.path(TABLE_ROOT, paste0("SFig2_time_dep_univariate_WL10_WL15_", DATE_STAMP, ".csv")), row.names = FALSE)
write.csv(multi_df, file.path(TABLE_ROOT, paste0("SFig2_time_dep_multivariable_WL10_WL15_", DATE_STAMP, ".csv")), row.names = FALSE)


y_levels_hr10 <- uni_df[uni_df$Coefficient.Name == "exp_WL10", ]
y_levels_hr10 <- y_levels_hr10$Cancer.Type[order(-exp(y_levels_hr10$Coefficient))]
y_levels_hr10 <- rev(y_levels_hr10)

prep_forest <- function(df, y_levels) {
  df %>%
    filter(Coefficient.Name %in% c("exp_WL10", "exp_WL15")) %>%
    mutate(
      Hazard.Ratio = exp(Coefficient),
      Lower.CI.HR  = exp(Lower.CI),
      Upper.CI.HR  = exp(Upper.CI),
      Threshold    = recode(Coefficient.Name, exp_WL10 = ">=10%", exp_WL15 = ">=15%"),
      Shape        = ifelse(Significant, 16, 1),
      Cancer.Type  = factor(Cancer.Type, levels = y_levels),
      Threshold    = factor(Threshold, levels = c(">=10%", ">=15%"))
    ) %>%
    filter(!is.na(Cancer.Type)) %>%
    arrange(Cancer.Type, Threshold)
}

make_breaks <- function(xmin, xmax) {
  cand <- c(0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30, 50)
  cand[cand >= xmin & cand <= xmax]
}

plot_forest <- function(df, outname, width = 2.5, height = 3.7) {
  xmin <- max(min(df$Lower.CI.HR, na.rm = TRUE) * 0.95, 0.05)
  xmax <- max(df$Upper.CI.HR, na.rm = TRUE) * 1.05
  brks <- make_breaks(xmin, xmax)
  if (length(brks) < 3) brks <- sort(unique(c(xmin, 1, xmax)))

  col_map <- c(">=10%" = "#D9A3B5FF", ">=15%" = "#8F3F63FF")

  p <- ggplot(df, aes(
    x = Hazard.Ratio,
    y = Cancer.Type,
    xmin = Lower.CI.HR, xmax = Upper.CI.HR,
    color = Threshold
  )) +
    geom_vline(xintercept = brks, linetype = "dotted", color = "grey80") +
    geom_vline(xintercept = 1,    linetype = "solid",  color = "grey40") +
    geom_pointrange(position = position_dodge(width = 0.45), size = 0.45) +
    geom_point(aes(shape = as.factor(Shape)), size = 0.6,
               position = position_dodge(width = 0.45)) +
    scale_shape_manual(values = c("1" = 1, "16" = 16), guide = "none") +
    scale_y_discrete(labels = function(x) recode(x, !!!code_map, .default = x)) +
    scale_x_log10(breaks = brks, limits = c(xmin, xmax)) +
    scale_color_manual(values = col_map) +
    labs(title = NULL, x = "HR", y = NULL, color = NULL) +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_blank(),
      axis.title.x = element_text(size = 9),
      axis.text.x  = element_text(size = 9),
      axis.text.y  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.text      = element_text(size = 9),
      legend.key.height= unit(2, "mm"),
      legend.key.width = unit(2, "mm"),
      plot.margin      = margin(t = 1, r = 1, b = 0, l = 1)
    ) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE))

  ggsave(file.path(PLOT_ROOT, outname),
         plot = p, width = width, height = height, dpi = 300, useDingbats = FALSE)
  p
}

df_uni   <- prep_forest(uni_df,   y_levels = y_levels_hr10)
df_multi <- prep_forest(multi_df, y_levels = y_levels_hr10)

plot_forest(df_uni,   paste0("SFig2_time_dep_cox_forestplot_uni_WL10_WL15_", DATE_STAMP, ".pdf"))
plot_forest(df_multi, paste0("SFig2_time_dep_cox_forestplot_multi_WL10_WL15_", DATE_STAMP, ".pdf"))

cat("\nWrote WL10/WL15 time-dependent Cox outputs to:\n  ", TABLE_ROOT, "\n  ", PLOT_ROOT, "\n")
