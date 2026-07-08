library(data.table)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(patchwork)

# SFig4D: combined sensitivity summary - 3 rows (Routine / Derived / Additional
# labs), each faceted by the 4 sensitivity conditions (WL5 Stage 1-3, WL5 Stage
# 4, WL10, WL15), point size = # cancer types contributing, segment from 0 to
# mean log2(OR). Adapted from 0303_ccx_revisions/rev_code/Figures/SFig4.R
# (SFig4_stratified_glmm_ALLPANELS section). Requires all 12 sensitivity GLMM
# runs (routine/derived/additional x 4 conditions) to already be complete.

BASE_REV   <- "."
REV_TABLES <- file.path(BASE_REV, "rev_tables")
REV_PLOTS  <- file.path(BASE_REV, "rev_plots", "fearon_definition")
DATE_STAMP <- "20260706"

out_dir <- file.path(REV_PLOTS, "SFig4")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

analysis_levels_plot <- c("WL5 Stage 1-3", "WL5 Stage 4", "WL10", "WL15")

col_map <- c(
  "WL5 Stage 4"   = "#6388B4FF",
  "WL10"          = "#FFAE34FF",
  "WL15"          = "#EF6F6AFF",
  "WL5 Stage 1-3" = "#8CC2CAFF"
)

theme_core <- theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 9),
    strip.text = element_text(size = 9),
    axis.title.x = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.ticks = element_line(color = "black", linewidth = 0.3),
    axis.line.x = element_line(color = "black", linewidth = 0.25),
    axis.line.y = element_line(color = "black", linewidth = 0.25),
    panel.spacing.x = unit(1.1, "lines"),
    plot.margin = margin(6, 6, 6, 6),
    strip.background = element_blank()
  )

summarise_mean_from_results <- function(df, logor_col) {
  df %>%
    mutate(log_or_use = suppressWarnings(as.numeric(.data[[logor_col]]))) %>%
    filter(!is.na(test), is.finite(log_or_use)) %>%
    group_by(analysis, test) %>%
    summarise(
      n_eff       = n(),
      mean_log_or = mean(log_or_use),
      mean_log2or = mean_log_or / log(2),
      .groups = "drop"
    )
}

make_panel_plot <- function(mean_df, panel_title, test_levels, show_strips = TRUE, show_x_title = TRUE) {
  mean_df <- mean_df %>%
    mutate(
      analysis = factor(analysis, levels = analysis_levels_plot),
      test     = factor(test, levels = test_levels),
      size_val = sqrt(pmax(n_eff, 0))
    )

  p <- ggplot(mean_df %>% filter(!is.na(mean_log2or)), aes(x = mean_log2or, y = test)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.35, color = "gray45") +
    geom_segment(aes(x = 0, xend = mean_log2or, yend = test, color = analysis), linewidth = 0.55, alpha = 0.95) +
    geom_point(aes(size = size_val, color = analysis), alpha = 0.85) +
    facet_wrap(~analysis, nrow = 1, scales = "free_x") +
    scale_color_manual(values = col_map) +
    scale_size_continuous(range = c(0.75, 2.0), name = "# cancers") +
    scale_x_continuous(
      breaks = scales::breaks_pretty(n = 3),
      labels = scales::label_number(accuracy = 0.1, drop0trailing = TRUE)
    ) +
    labs(title = panel_title, x = if (show_x_title) "Mean log2(OR)" else NULL, y = NULL, color = NULL) +
    theme_core

  if (!show_strips)  p <- p + theme(strip.text = element_blank())
  if (!show_x_title) p <- p + theme(axis.title.x = element_blank())
  p
}

# ------------------ Routine blood labs ------------------

read_routine <- function(tag, label) {
  fp <- file.path(REV_TABLES, "serology_glmm", paste0(tag, "_", DATE_STAMP),
                   paste0("glmm_results_labs_", tag, "_", DATE_STAMP, ".csv"))
  read_csv(fp, show_col_types = FALSE) %>%
    mutate(analysis = label, test = as.character(test),
           log_or_use = dplyr::coalesce(suppressWarnings(as.numeric(log_or)), suppressWarnings(as.numeric(logor))))
}

core_base <- read_csv(file.path(REV_TABLES, "serology_glmm", paste0("WL5_", DATE_STAMP),
                                 paste0("glmm_results_labs_5_", DATE_STAMP, ".csv")), show_col_types = FALSE) %>%
  mutate(test = as.character(test), log_or_use = suppressWarnings(as.numeric(log_or))) %>%
  filter(!is.na(test), is.finite(log_or_use))

core_order <- core_base %>% group_by(test) %>% summarise(m = mean(log_or_use, na.rm = TRUE), .groups = "drop") %>%
  arrange(m) %>% pull(test)

core_all <- bind_rows(
  read_routine("WL5_BMIlt20_Stage1-3", "WL5 Stage 1-3"),
  read_routine("WL5_BMIlt20_Stage4",   "WL5 Stage 4"),
  read_routine("WL10",                 "WL10"),
  read_routine("WL15",                 "WL15")
)

core_mean <- core_all %>%
  summarise_mean_from_results("log_or_use") %>%
  complete(analysis, test, fill = list(n_eff = 0, mean_log2or = NA_real_)) %>%
  mutate(analysis = factor(analysis, levels = analysis_levels_plot), test = factor(test, levels = core_order))

# ------------------ Derived labs ------------------

read_derived <- function(tag, label) {
  fp <- file.path(REV_TABLES, "derived_labs_glmm", paste0(tag, "_", DATE_STAMP),
                   paste0("glmm_derived_labs_results_z_", tag, "_", DATE_STAMP, ".csv"))
  fread(fp) %>% as.data.frame() %>% mutate(analysis = label, test = as.character(test))
}

der_raw <- bind_rows(
  read_derived("WL5_BMIlt20_Stage1-3", "WL5 Stage 1-3"),
  read_derived("WL5_BMIlt20_Stage4",   "WL5 Stage 4"),
  read_derived("WL10",                 "WL10"),
  read_derived("WL15",                 "WL15")
)

der_mean <- der_raw %>%
  summarise_mean_from_results("logor") %>%
  complete(analysis, test, fill = list(n_eff = 0, mean_log2or = NA_real_)) %>%
  mutate(analysis = factor(analysis, levels = analysis_levels_plot))

derived_order <- der_mean %>% filter(analysis == "WL5 Stage 1-3") %>% arrange(desc(mean_log2or)) %>%
  pull(test) %>% as.character()
derived_order <- rev(derived_order)
derived_order <- c(derived_order, setdiff(unique(der_mean$test), derived_order))

# ------------------ Additional labs ------------------

read_additional <- function(tag, label) {
  fp <- file.path(REV_TABLES, "serology_additional_labs", paste0(tag, "_", DATE_STAMP),
                   paste0("table_additionallabs_glmm_z500_", tag, "_", DATE_STAMP, ".csv"))
  fread(fp) %>% as.data.frame() %>% mutate(analysis = label, test = as.character(test))
}

add_raw <- bind_rows(
  read_additional("WL5_BMIlt20_Stage1-3", "WL5 Stage 1-3"),
  read_additional("WL5_BMIlt20_Stage4",   "WL5 Stage 4"),
  read_additional("WL10",                 "WL10"),
  read_additional("WL15",                 "WL15")
)

add_mean <- add_raw %>%
  summarise_mean_from_results("logor") %>%
  complete(analysis, test, fill = list(n_eff = 0, mean_log2or = NA_real_)) %>%
  mutate(analysis = factor(analysis, levels = analysis_levels_plot))

add_order <- add_mean %>% filter(analysis == "WL5 Stage 1-3") %>% arrange(desc(mean_log2or)) %>%
  pull(test) %>% as.character()
add_order <- rev(add_order)
add_order <- c(add_order, setdiff(unique(add_mean$test), add_order))

# ------------------ Combine ------------------

p_core <- make_panel_plot(core_mean, "Routine Blood Labs", core_order, show_strips = TRUE,  show_x_title = FALSE)
p_der  <- make_panel_plot(der_mean,  "Derived Labs",       derived_order, show_strips = FALSE, show_x_title = FALSE)
p_add  <- make_panel_plot(add_mean,  "Additional Labs",    add_order,     show_strips = FALSE, show_x_title = TRUE)

p_all <- (p_core / p_der / p_add) +
  plot_layout(heights = c(3.5, 0.6, 0.7), guides = "collect") &
  theme(legend.position = "none")

out_fp <- file.path(out_dir, paste0("SFig4D_sensitivity_lollipop_", DATE_STAMP, ".pdf"))
ggsave(out_fp, p_all, width = 7.2, height = 8, device = "pdf")

cat("\nWrote SFig4D (combined sensitivity lollipop) to:", out_fp, "\n")
