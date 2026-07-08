library(data.table)
library(ggplot2)

BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)
eps[, MRN := as.integer(MRN)]

epi_counts <- eps[has_cachexia_valid_edemaQC == 1, .(n_episodes = .N), by = MRN]

all_mrn <- unique(eps[, .(MRN)])
epi_counts_all <- merge(all_mrn, epi_counts, by = "MRN", all.x = TRUE)
epi_counts_all[is.na(n_episodes), n_episodes := 0L]

epi_counts_all[, epi_bin := fifelse(n_episodes == 0, "0",
                                    fifelse(n_episodes == 1, "1",
                                            fifelse(n_episodes == 2, "2", "3+")))]
epi_counts_all[, epi_bin := factor(epi_bin, levels = c("0","1","2","3+"))]

epi_summary <- epi_counts_all[, .N, by = epi_bin]
setnames(epi_summary, "N", "Count")
epi_summary[, Total := sum(Count)]
epi_summary[, Proportion := 100 * Count / Total]
epi_summary[, SE    := sqrt((Proportion/100) * (1 - Proportion/100) / Total) * 100]
epi_summary[, Lower := pmax(0,   Proportion - 1.96 * SE)]
epi_summary[, Upper := pmin(100, Proportion + 1.96 * SE)]

print(epi_summary)

fwrite(epi_summary, file.path(out_dir, "fig2C_episode_recurrence_summary.csv"))

cat(sprintf("0 episodes : %d (%.2f%%)\n", epi_summary[epi_bin=="0",  Count], epi_summary[epi_bin=="0",  Proportion]))
cat(sprintf("1 episode  : %d (%.2f%%)\n", epi_summary[epi_bin=="1",  Count], epi_summary[epi_bin=="1",  Proportion]))
cat(sprintf("2 episodes : %d (%.2f%%)\n", epi_summary[epi_bin=="2",  Count], epi_summary[epi_bin=="2",  Proportion]))
cat(sprintf("3+ episodes: %d (%.2f%%)\n", epi_summary[epi_bin=="3+", Count], epi_summary[epi_bin=="3+", Proportion]))

p2c <- ggplot(epi_summary, aes(x = epi_bin, y = Proportion)) +
  geom_col(fill = "black", width = 0.7) +
  geom_segment(aes(x = epi_bin, xend = epi_bin, y = Lower, yend = Proportion),
               color = "white", linewidth = 0.7) +
  geom_segment(aes(x = epi_bin, xend = epi_bin, y = Proportion, yend = Upper),
               color = "black", linewidth = 0.7) +
  labs(x = "Episodes per patient", y = "% of patients", title = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme_minimal(base_family = "ArialMT") +
  theme(
    axis.title.x = element_text(size = 9),
    axis.title.y = element_text(size = 9),
    axis.text.x  = element_text(size = 9),
    axis.text.y  = element_text(size = 9),
    axis.line    = element_line(color = "black", linewidth = 0.25),
    axis.ticks   = element_line(color = "black", linewidth = 0.25),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(out_dir, paste0("fig2C_episode_recurrence_", DATE_STAMP, ".pdf")),
       p2c, width = 2.5, height = 2.6, dpi = 300)
cat("\nWrote Fig2C episode recurrence plot to:", out_dir, "\n")
