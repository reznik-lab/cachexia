library(data.table)
library(ggplot2)
library(patchwork)

BASE_REV   <- "."
REV_PLOTS  <- file.path(BASE_REV, "rev_plots", "fearon_definition")
REV_TABLES <- file.path(BASE_REV, "rev_tables", "fearon_definition")
DATE_STAMP <- "20260706"

FIG3_DIR  <- file.path(REV_PLOTS, "Fig3")
SFIG3_DIR <- file.path(REV_PLOTS, "SFig3")
dir.create(FIG3_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(SFIG3_DIR, recursive = TRUE, showWarnings = FALSE)

make_bar <- function(dt, title, show_x_text) {
  p <- ggplot(dt, aes(x = Cancer_Code, y = prop)) +
    geom_bar(stat = "identity", fill = "black", width = 0.9) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = upper), color = "gray10", linewidth = 0.6) +
    geom_segment(aes(xend = Cancer_Code, y = prop, yend = lower), color = "white", linewidth = 0.6) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.025, 0)) +
    labs(title = title, x = "", y = "Percentage (%)") +
    theme_minimal(base_size = 9, base_family = "ArialMT") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 9),
      axis.text.x = if (show_x_text) element_text(size = 9, angle = 90, hjust = 1) else element_blank(),
      axis.ticks.x = if (show_x_text) element_line(linewidth = 0.3) else element_blank(),
      axis.text.y = element_text(size = 9),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks.y = element_line(linewidth = 0.3)
    )
  p
}

build_combined <- function(wl_label, out_dir, title_top, title_bottom, outname, plot_width = 4.5, plot_height = 4.5) {
  early_fp <- file.path(REV_TABLES, paste0("fig3_ccx_180_", wl_label, "_stage13_untreated_per_cancer_summary_", DATE_STAMP, ".csv"))
  death_fp <- file.path(REV_TABLES, paste0("fig3_ccx_near_death_", wl_label, "_per_cancer_summary_", DATE_STAMP, ".csv"))

  early_dt <- fread(early_fp)
  death_dt <- fread(death_fp)

  early_dt <- early_dt[order(-prop)]
  target_codes <- early_dt$Cancer_Code

  early_dt[, Cancer_Code := factor(Cancer_Code, levels = target_codes)]
  death_dt <- death_dt[Cancer_Code %in% target_codes]
  death_dt[, Cancer_Code := factor(as.character(Cancer_Code), levels = target_codes)]

  p_top    <- make_bar(early_dt, title_top,    show_x_text = FALSE)
  p_bottom <- make_bar(death_dt, title_bottom, show_x_text = TRUE)

  p_combined <- p_top / p_bottom + plot_layout(heights = c(1, 1))

  ggsave(file.path(out_dir, outname), plot = p_combined, width = plot_width, height = plot_height, dpi = 300, device = pdf)
  p_combined
}

build_combined("5", FIG3_DIR,
                "Episodes within the first 180 days\n(Stage 1-3, untreated at diagnosis)",
                "Episodes near death (+/-30 days)",
                paste0("fig3_combined_stacked_WL5_", DATE_STAMP, ".pdf"))

build_combined("10", SFIG3_DIR,
                "Episodes within the first 180 days\n(Stage 1-3, untreated at diagnosis) (>=10% WL)",
                "Episodes near death (+/-30 days) (>=10% WL)",
                paste0("sfig3A_combined_stacked_WL10_", DATE_STAMP, ".pdf"),
                plot_width = 4.0, plot_height = 4.5)

build_combined("15", SFIG3_DIR,
                "Episodes within the first 180 days\n(Stage 1-3, untreated at diagnosis) (>=15% WL)",
                "Episodes near death (+/-30 days) (>=15% WL)",
                paste0("sfig3B_combined_stacked_WL15_", DATE_STAMP, ".pdf"),
                plot_width = 4.0, plot_height = 4.5)

cat("\nWrote combined stacked Fig3/SFig3 plots to:\n  ", FIG3_DIR, "\n  ", SFIG3_DIR, "\n")
