source("rev_code/Figures/figure_panel/SFig2_WLGS_wholecohort_setup.R")
library(ggplot2)

make_grade_bar <- function(summary_dt, y_col) {
  summary_dt[, end_class := factor(wlgs, levels = sort(unique(wlgs)))]
  ggplot(summary_dt, aes(x = end_class, y = .data[[y_col]])) +
    geom_bar(stat = "identity", width = 0.8, fill = "black") +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.25) +
    scale_y_continuous(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0.25, 0)) +
    labs(title = NULL, x = NULL, y = "Percentage (%)") +
    theme_minimal(base_family = "ArialMT") +
    theme(
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      panel.spacing = unit(0.05, "lines"),
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

summary_dt <- patient_worst[, .N, by = wlgs][order(wlgs)]
summary_dt[, prop := N / sum(N)]
summary_dt[, pct_patients := 100 * prop]
summary_dt[, DATE_STAMP := DATE_STAMP]

print(summary_dt)

write.csv(summary_dt, file.path(table_dir, paste0("SFig2D_WLGS_patient_grade_distribution_", DATE_STAMP, ".csv")), row.names = FALSE)
write.csv(patient_worst, file.path(plot_dir, paste0("SFig2D_WLGS_patient_level_", DATE_STAMP, ".csv")), row.names = FALSE)

p_patient <- make_grade_bar(summary_dt, "pct_patients")
ggsave(filename = file.path(plot_dir, paste0("SFig2D_WLGS_patient_grade_distribution_", DATE_STAMP, ".pdf")),
       plot = p_patient, width = 1.7, height = 1.5, units = "in", useDingbats = FALSE)

cat(sprintf("Whole cohort WLG%d: %.1f%% (n=%d)\n", summary_dt$wlgs, summary_dt$pct_patients, summary_dt$N))

summary_ep <- eps_wlgs_episode[, .N, by = wlgs][order(wlgs)]
summary_ep[, prop := N / sum(N)]
summary_ep[, pct_episodes := 100 * prop]
summary_ep[, DATE_STAMP := DATE_STAMP]

print(summary_ep)

write.csv(summary_ep, file.path(table_dir, paste0("SFig2D_WLGS_episode_grade_distribution_", DATE_STAMP, ".csv")), row.names = FALSE)
write.csv(eps_wlgs_episode, file.path(plot_dir, paste0("SFig2D_WLGS_episode_level_", DATE_STAMP, ".csv")), row.names = FALSE)

p_episode <- make_grade_bar(summary_ep, "pct_episodes")
ggsave(filename = file.path(plot_dir, paste0("SFig2D_WLGS_episode_grade_distribution_", DATE_STAMP, ".pdf")),
       plot = p_episode, width = 1.7, height = 1.5, units = "in", useDingbats = FALSE)

cat(sprintf("Cachexia episodes only WLG%d: %.1f%% (n=%d)\n", summary_ep$wlgs, summary_ep$pct_episodes, summary_ep$N))
