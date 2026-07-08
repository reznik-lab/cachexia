library(data.table)
library(ggplot2)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

plot_dir  <- file.path(REV_PLOTS,   "fearon_definition", "SFig2")
table_dir <- file.path(REV_RESULTS, "fearon_definition", "SFig2")
dir.create(plot_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

fp_6mo <- file.path(REV_RESULTS, paste0("episode_regain_WL5_BMIlt20rule_", DATE_STAMP, "_recover5pct_6mo.csv"))
fp_all <- file.path(REV_RESULTS, paste0("episode_regain_WL5_BMIlt20rule_", DATE_STAMP, "_recover5pct_notimelimit.csv"))

dt_6mo <- fread(fp_6mo)
dt_all <- fread(fp_all)

per_episode <- function(dt, label) {
  data.table(Window = label, total_episodes = nrow(dt), has_n = sum(dt$regained))
}

summary_dt <- rbind(
  per_episode(dt_6mo, "Within 6mo"),
  per_episode(dt_all, "Overall")
)
summary_dt[, Proportion := 100 * has_n / total_episodes]

add_wald_ci <- function(dt, prop_col = "Proportion", n_col = "total_episodes") {
  dt[, SE := sqrt((get(prop_col)/100) * (1 - get(prop_col)/100) / get(n_col)) * 100]
  dt[, Lower := pmax(0,   get(prop_col) - 1.96 * SE)]
  dt[, Upper := pmin(100, get(prop_col) + 1.96 * SE)]
  dt
}

summary_dt <- add_wald_ci(summary_dt)
summary_dt[, Window := factor(Window, levels = c("Within 6mo", "Overall"))]

print(summary_dt)
fwrite(summary_dt, file.path(table_dir, paste0("SFig2_regain_5pct_6mo_vs_notimelimit_", DATE_STAMP, ".csv")))

theme_msk_style <- function() {
  theme_minimal(base_family = "ArialMT") +
    theme(
      plot.title   = element_blank(),
      axis.title.x = element_blank(),
      axis.title.y = element_text(size = 9),
      axis.text.x  = element_text(size = 9),
      axis.text.y  = element_text(size = 9),
      axis.line    = element_line(color = "black", linewidth = 0.2),
      axis.ticks.x = element_line(color = "black", linewidth = 0.3),
      axis.ticks.y = element_line(color = "black", linewidth = 0.3),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position  = "none"
    )
}

p <- ggplot(summary_dt, aes(x = Window, y = Proportion)) +
  geom_col(width = 0.9, fill = "black") +
  geom_segment(aes(xend = Window, y = Proportion, yend = Upper), color = "gray10", linewidth = 0.7) +
  geom_segment(aes(xend = Window, y = Proportion, yend = Lower), color = "white", linewidth = 0.7) +
  labs(x = NULL, y = "Episodes (%)") +
  scale_y_continuous(expand = c(0, 0), limits = c(0, max(summary_dt$Upper) * 1.1)) +
  theme_msk_style()

ggsave(file.path(plot_dir, paste0("SFig2_regain_5pct_6mo_vs_notimelimit_", DATE_STAMP, ".pdf")),
       plot = p, width = 2, height = 1.7, dpi = 300)

cat("\nWrote regain comparison outputs to:\n  ", table_dir, "\n  ", plot_dir, "\n")
