library(ggplot2)
library(data.table)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if ("showtext" %in% loadedNamespaces()) showtext::showtext_auto(FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)
eps[, MRN := as.integer(MRN)]

total_patients <- eps[, uniqueN(MRN)]
has_n <- eps[has_cachexia_valid_edemaQC == 1, uniqueN(MRN)]
no_n  <- total_patients - has_n

plot_df <- data.table(
  Group = factor(c("No cachexia", "Has cachexia"),
                 levels = c("No cachexia", "Has cachexia")),
  Count = c(no_n, has_n)
)
plot_df[, Proportion := Count / total_patients * 100]

plot_df[, SE := sqrt((Proportion/100) * (1 - Proportion/100) / total_patients) * 100]
plot_df[, Lower := pmax(0,   Proportion - 1.96 * SE)]
plot_df[, Upper := pmin(100, Proportion + 1.96 * SE)]

p2b <- ggplot(plot_df, aes(x = Group, y = Proportion)) +
  geom_col(width = 0.9, fill = "black") +
  geom_segment(aes(xend = Group, y = Proportion, yend = Upper),
               color = "gray10", linewidth = 0.7) +
  geom_segment(aes(xend = Group, y = Proportion, yend = Lower),
               color = "white", linewidth = 0.7) +
  labs(x = NULL, y = "Percentage (%)", title = NULL) +
  scale_y_continuous(expand = c(0, 0)) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    plot.title   = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 8),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", size = 0.2),
    axis.ticks.x = element_line(color = "black", size = 0.3),
    axis.ticks.y = element_line(color = "black", size = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none"
  )

print(p2b)

ggsave(file.path(out_dir, "fig2B_cachexia.pdf"),
       plot = p2b, width = 2.2, height = 2.1, dpi = 300)

fwrite(plot_df, file.path(out_dir, "fig2B_cachexia_summary.csv"))

cat("Has cachexia:", has_n, "(", round(plot_df[Group=="Has cachexia", Proportion], 2), "%)\n")
cat("No cachexia :", no_n,  "(", round(plot_df[Group=="No cachexia",  Proportion], 2), "%)\n")
