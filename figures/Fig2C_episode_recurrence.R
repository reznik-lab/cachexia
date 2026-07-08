library(data.table)

# run with working directory set to the project root (containing rev_inputs/, rev_results/, rev_plots/, rev_tables/, rev_code/)
BASE_REV    <- "."
REV_RESULTS <- file.path(BASE_REV, "rev_results")
REV_PLOTS   <- file.path(BASE_REV, "rev_plots")
DATE_STAMP  <- "20260706"

out_dir <- file.path(REV_PLOTS, "fearon_definition", "Fig2")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

eps_fp <- file.path(REV_RESULTS, paste0("episode_summary_valid_WL5_BMIlt20rule_", DATE_STAMP, "_alpha0.2_dur15_wl2_edemaQC_LOG_w30_up5_ret2.csv"))
eps <- fread(eps_fp)
eps[, MRN := as.integer(MRN)]

# count episodes per patient (only valid cachexia episodes)
epi_counts <- eps[has_cachexia_valid_edemaQC == 1, .(n_episodes = .N), by = MRN]

# include patients with 0 episodes
all_mrn <- unique(eps[, .(MRN)])
epi_counts_all <- merge(all_mrn, epi_counts, by = "MRN", all.x = TRUE)
epi_counts_all[is.na(n_episodes), n_episodes := 0L]

# bin into 0, 1, 2, 3+
epi_counts_all[, epi_bin := fifelse(n_episodes == 0, "0",
                                    fifelse(n_episodes == 1, "1",
                                            fifelse(n_episodes == 2, "2", "3+")))]
epi_counts_all[, epi_bin := factor(epi_bin, levels = c("0","1","2","3+"))]

epi_summary <- epi_counts_all[, .N, by = epi_bin]
setnames(epi_summary, "N", "Count")
epi_summary[, Total := sum(Count)]
epi_summary[, Proportion := 100 * Count / Total]

print(epi_summary)

fwrite(epi_summary, file.path(out_dir, "fig2C_episode_recurrence_summary.csv"))

cat(sprintf("0 episodes : %d (%.2f%%)\n", epi_summary[epi_bin=="0",  Count], epi_summary[epi_bin=="0",  Proportion]))
cat(sprintf("1 episode  : %d (%.2f%%)\n", epi_summary[epi_bin=="1",  Count], epi_summary[epi_bin=="1",  Proportion]))
cat(sprintf("2 episodes : %d (%.2f%%)\n", epi_summary[epi_bin=="2",  Count], epi_summary[epi_bin=="2",  Proportion]))
cat(sprintf("3+ episodes: %d (%.2f%%)\n", epi_summary[epi_bin=="3+", Count], epi_summary[epi_bin=="3+", Proportion]))
