library(data.table)
library(ggplot2)

BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "SFig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)

eps_valid <- eps[has_cachexia_valid_edemaQC == 1]
mu_wl <- mean(eps_valid$weight_loss, na.rm = TRUE)
cat(sprintf("n episodes: %d | mean weight loss: %.2f%%\n", nrow(eps_valid), mu_wl))

p_sfig2a <- ggplot(eps_valid, aes(x = weight_loss)) +
  geom_histogram(bins = 30, fill = "#6388B4FF", color = "black", linewidth = 0.2) +
  geom_vline(xintercept = mu_wl, color = "#EF6F6AFF", linetype = "dotted", linewidth = 0.8) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "% Weight loss", y = "Episodes", title = NULL) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.2),
    axis.ticks   = element_line(color = "black", linewidth = 0.3),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(out_dir, paste0("SFig2A_weight_loss_magnitude_histogram_", DATE_STAMP, ".pdf")),
       p_sfig2a, width = 2.2, height = 2.1, dpi = 300, useDingbats = FALSE)

fwrite(eps_valid[, .(MRN, weight_loss)], file.path(out_dir, paste0("SFig2A_weight_loss_magnitude_data_", DATE_STAMP, ".csv")))

cat("\nWrote SFig2A weight loss magnitude histogram to:", out_dir, "\n")
