library(data.table)
library(dplyr)
library(ggplot2)
library(patchwork)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_TABLES  <- file.path(BASE_REV, "rev_tables")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

TABLE_ROOT <- file.path(REV_TABLES, "fearon_definition")
PLOT_ROOT  <- file.path(REV_PLOTS, "fearon_definition", "SFig2", "time_dep_cox_combined")
dir.create(PLOT_ROOT, recursive = TRUE, showWarnings = FALSE)

code_map <- c(
  "Esophageal Adenocarcinoma" = "ESCA", "Acute Myeloid Leukemia" = "AML",
  "Stomach Adenocarcinoma" = "STAD", "Pancreatic Adenocarcinoma" = "PAAD",
  "Small Cell Lung Cancer" = "SCLC", "Intrahepatic Cholangiocarcinoma" = "IHCH",
  "Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma" = "USC",
  "High-Grade Serous Ovarian Cancer" = "HGSOC", "Cancer of Unknown Primary" = "CUP",
  "Colon Adenocarcinoma" = "COAD", "Colorectal Adenocarcinoma" = "COADREAD",
  "Rectal Adenocarcinoma" = "READ", "Diffuse Large B-Cell Lymphoma, NOS" = "DLBCLNOS",
  "Lung Squamous Cell Carcinoma" = "LUSC", "Non-Small Cell Lung Cancer" = "NSCLC",
  "Bladder Urothelial Carcinoma" = "BLCA", "Glioblastoma Multiforme" = "GBM",
  "Lung Adenocarcinoma" = "LUAD", "Renal Clear Cell Carcinoma" = "CCRCC",
  "Plasma Cell Myeloma" = "PCM", "Invasive Breast Carcinoma" = "BRCA",
  "Myelodysplastic Workup" = "MDSWP", "Gastrointestinal Stromal Tumor" = "GIST",
  "Breast Invasive Ductal Carcinoma" = "IDC", "Cutaneous Melanoma" = "SKCM",
  "Breast Invasive Lobular Carcinoma" = "ILC", "Uterine Endometrioid Carcinoma" = "UEC",
  "Prostate Adenocarcinoma" = "PRAD", "Follicular Lymphoma" = "FL",
  "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma" = "CLLSLL"
)

wl_uni    <- fread(file.path(TABLE_ROOT, paste0("SFig2_time_dep_univariate_WL10_WL15_", DATE_STAMP, ".csv")))
wl_multi  <- fread(file.path(TABLE_ROOT, paste0("SFig2_time_dep_multivariable_WL10_WL15_", DATE_STAMP, ".csv")))
wlg_uni   <- fread(file.path(TABLE_ROOT, paste0("SFig2_time_dep_univariate_WLGfirst_", DATE_STAMP, ".csv")))
wlg_multi <- fread(file.path(TABLE_ROOT, paste0("SFig2_time_dep_multivariable_WLGfirst_", DATE_STAMP, ".csv")))

# shared y-axis order (by exp_WL10 HR, ascending bottom-to-top) -- used by BOTH panels
y_levels <- wl_uni[Coefficient.Name == "exp_WL10"][order(-exp(Coefficient))]$Cancer.Type
y_levels <- rev(y_levels)

wl_col_map  <- c(">=10%" = "#D9A3B5FF", ">=15%" = "#8F3F63FF")
wlg_col_map <- c("WLG0-2" = "#4681c2", "WLG3-4" = "#845ea9")

prep_wl <- function(df) {
  df %>%
    filter(Coefficient.Name %in% c("exp_WL10", "exp_WL15")) %>%
    mutate(
      Hazard.Ratio = exp(Coefficient),
      Lower.CI.HR  = exp(Lower.CI),
      Upper.CI.HR  = exp(Upper.CI),
      Shape        = ifelse(Significant, 16, 1),
      Series       = recode(Coefficient.Name, exp_WL10 = ">=10%", exp_WL15 = ">=15%"),
      Series       = factor(Series, levels = c(">=10%", ">=15%")),
      Cancer.Type  = factor(Cancer.Type, levels = y_levels)
    ) %>%
    filter(!is.na(Cancer.Type)) %>%
    arrange(Cancer.Type, Series)
}

prep_wlg <- function(df) {
  df %>%
    filter(Coefficient.Name %in% c("low", "high")) %>%
    mutate(
      Hazard.Ratio = exp(Coefficient),
      Lower.CI.HR  = exp(Lower.CI),
      Upper.CI.HR  = exp(Upper.CI),
      Shape        = ifelse(Significant, 16, 1),
      Series       = recode(Coefficient.Name, low = "WLG0-2", high = "WLG3-4"),
      Series       = factor(Series, levels = c("WLG0-2", "WLG3-4")),
      Cancer.Type  = factor(Cancer.Type, levels = y_levels)
    ) %>%
    filter(!is.na(Cancer.Type)) %>%
    arrange(Cancer.Type, Series)
}

make_breaks <- function(xmin, xmax) {
  cand <- c(0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30, 50)
  cand[cand >= xmin & cand <= xmax]
}

# side = "left" keeps y-axis text (cancer codes); side = "right" hides it (shares the left panel's y-axis)
plot_panel <- function(df, col_map, side = c("left", "right")) {
  side <- match.arg(side)
  xmin <- max(min(df$Lower.CI.HR, na.rm = TRUE) * 0.95, 0.05)
  xmax <- max(df$Upper.CI.HR, na.rm = TRUE) * 1.05
  brks <- make_breaks(xmin, xmax)
  if (length(brks) < 3) brks <- sort(unique(c(xmin, 1, xmax)))

  p <- ggplot(df, aes(
    x = Hazard.Ratio,
    y = Cancer.Type,
    xmin = Lower.CI.HR, xmax = Upper.CI.HR,
    color = Series
  )) +
    geom_vline(xintercept = brks, linetype = "dotted", color = "grey80") +
    geom_vline(xintercept = 1,    linetype = "solid",  color = "grey40") +
    geom_pointrange(position = position_dodge(width = 0.45), size = 0.45) +
    geom_point(aes(shape = as.factor(Shape)), size = 0.6,
               position = position_dodge(width = 0.45)) +
    scale_shape_manual(values = c("1" = 1, "16" = 16), guide = "none") +
    scale_y_discrete(labels = function(x) recode(x, !!!code_map, .default = x), drop = FALSE) +
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
      legend.text      = element_text(size = 8),
      legend.key.height= unit(2, "mm"),
      legend.key.width = unit(2, "mm"),
      plot.margin      = margin(t = 1, r = 1, b = 0, l = 1)
    ) +
    guides(color = guide_legend(nrow = 1, byrow = TRUE))

  if (side == "right") {
    p <- p + theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  }
  p
}

build_combined <- function(wl_df, wlg_df, outname, width = 4.6, height = 3.9) {
  df_wl  <- prep_wl(wl_df)
  df_wlg <- prep_wlg(wlg_df)

  p_left  <- plot_panel(df_wl,  wl_col_map,  side = "left")
  p_right <- plot_panel(df_wlg, wlg_col_map, side = "right")

  p_combined <- p_left + p_right + plot_layout(widths = c(1, 1))

  ggsave(file.path(PLOT_ROOT, outname), plot = p_combined,
         width = width, height = height, dpi = 300, useDingbats = FALSE)

  list(plot = p_combined, df_wl = df_wl, df_wlg = df_wlg)
}

res_uni   <- build_combined(wl_uni,   wlg_uni,   paste0("SFig2_time_dep_cox_forestplot_combined_uni_",   DATE_STAMP, ".pdf"))
res_multi <- build_combined(wl_multi, wlg_multi, paste0("SFig2_time_dep_cox_forestplot_combined_multi_", DATE_STAMP, ".pdf"))

write.csv(bind_rows(res_uni$df_wl,   res_uni$df_wlg),   file.path(TABLE_ROOT, paste0("SFig2_time_dep_combined_univariate_", DATE_STAMP, ".csv")),   row.names = FALSE)
write.csv(bind_rows(res_multi$df_wl, res_multi$df_wlg), file.path(TABLE_ROOT, paste0("SFig2_time_dep_combined_multivariable_", DATE_STAMP, ".csv")), row.names = FALSE)

cat("\nWrote combined (shared-y, independent-x) time-dependent Cox plots to:", PLOT_ROOT, "\n")
