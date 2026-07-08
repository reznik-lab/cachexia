# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
source("rev_code/Figures/figure_panel/Fig2F_setup.R")
source("rev_code/Figures/figure_panel/SFig2_WLGS_wholecohort_setup.R")
library(ggplot2)

DATE_STAMP <- "20260706"

PLOT_ROOT <- file.path(REV_PLOTS, "fearon_definition", "SFig2", "time_dep_cox_WLGfirst")
dir.create(PLOT_ROOT, recursive = TRUE, showWarnings = FALSE)

# ------------------ grade of each patient's FIRST cachexia episode ------------------ #
# eps_wlgs_episode (from SFig2_WLGS_wholecohort_setup.R) is episode-level, corrected grading

epg_first <- eps_wlgs_episode[, .(t_first = suppressWarnings(min(start_day, na.rm = TRUE))), by = MRN]
epg_first[!is.finite(t_first), t_first := NA_real_]

epg_first_wlg <- merge(eps_wlgs_episode, epg_first, by = "MRN", all.x = FALSE)
epg_first_wlg <- epg_first_wlg[start_day == t_first,
                                .(t_first = first(t_first),
                                  wlgs_first = suppressWarnings(max(wlgs, na.rm = TRUE))),
                                by = MRN]
epg_first_wlg[!is.finite(wlgs_first), wlgs_first := NA_integer_]

epg_first_wlg[, wlg_grp := fifelse(!is.na(wlgs_first) & wlgs_first <= 2, "WLG0-2",
                             fifelse(!is.na(wlgs_first) & wlgs_first >= 3, "WLG3-4", NA_character_))]
epg_first_wlg[, wlg_grp := factor(wlg_grp, levels = c("WLG0-2", "WLG3-4"))]

cac_spans <- merge(cac_spans, epg_first_wlg[, .(MRN, t_first, wlg_grp)], by = "MRN", all.x = TRUE)

cac_spans[, low  := as.integer(!is.na(t_first) & start_time >= t_first & wlg_grp == "WLG0-2")]
cac_spans[, high := as.integer(!is.na(t_first) & start_time >= t_first & wlg_grp == "WLG3-4")]

# ------------------ y-axis order: reuse WL10 HR ordering from the WL10/WL15 script, for visual consistency ------------------ #

wl10_fp <- file.path(TABLE_ROOT, paste0("SFig2_time_dep_univariate_WL10_WL15_", DATE_STAMP, ".csv"))
stopifnot(file.exists(wl10_fp))
wl10_df <- fread(wl10_fp)

y_levels_hr10 <- wl10_df[Coefficient.Name == "exp_WL10"][order(-exp(Coefficient))]$Cancer.Type
y_levels_hr10 <- rev(y_levels_hr10)

ct_list <- large_n_cancer_types$CANCER_TYPE_DETAILED
ct_list <- y_levels_hr10[y_levels_hr10 %chin% ct_list]

# ------------------ per-cancer Cox runner (low + high jointly modeled) ------------------ #

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

uni_df <- run_per_cancer(
  dt      = as.data.frame(cac_spans),
  ct_list = ct_list,
  formula = Surv(start_mt, end_mt, span_event) ~ low + high
)

multi_df <- run_per_cancer(
  dt      = as.data.frame(cac_spans),
  ct_list = ct_list,
  formula = Surv(start_mt, end_mt, span_event) ~ low + high + GENDER_NUM + age_at_diagnosis + stage_gran
)

write.csv(uni_df,   file.path(TABLE_ROOT, paste0("SFig2_time_dep_univariate_WLGfirst_", DATE_STAMP, ".csv")), row.names = FALSE)
write.csv(multi_df, file.path(TABLE_ROOT, paste0("SFig2_time_dep_multivariable_WLGfirst_", DATE_STAMP, ".csv")), row.names = FALSE)

# ------------------ forest plots ------------------ #

prep_forest <- function(df, y_levels) {
  df %>%
    filter(Coefficient.Name %in% c("low", "high")) %>%
    mutate(
      Hazard.Ratio = exp(Coefficient),
      Lower.CI.HR  = exp(Lower.CI),
      Upper.CI.HR  = exp(Upper.CI),
      Threshold    = recode(Coefficient.Name, low = "WLG0-2", high = "WLG3-4"),
      Shape        = ifelse(Significant, 16, 1),
      Cancer.Code  = recode(Cancer.Type, !!!code_map, .default = Cancer.Type),
      Cancer.Type  = factor(Cancer.Type, levels = y_levels),
      Threshold    = factor(Threshold, levels = c("WLG0-2", "WLG3-4"))
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

  col_map <- c("WLG0-2" = "#4681c2", "WLG3-4" = "#845ea9")

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

plot_forest(df_uni,   paste0("SFig2_time_dep_cox_forestplot_uni_WLGfirst_", DATE_STAMP, ".pdf"))
plot_forest(df_multi, paste0("SFig2_time_dep_cox_forestplot_multi_WLGfirst_", DATE_STAMP, ".pdf"))

cat("\nWrote WLGfirst time-dependent Cox outputs to:\n  ", TABLE_ROOT, "\n  ", PLOT_ROOT, "\n")
